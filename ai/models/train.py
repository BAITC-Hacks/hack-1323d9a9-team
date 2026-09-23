"""Train production candidates from archived, issue-time-safe weather forecasts.

The archive is populated by :mod:`ai.services.weather_client`.  This trainer
never calls a weather observation API and never substitutes measured SCADA
weather for a missing forecast. The source SCADA clock is station local time
(Asia/Almaty, confirmed by the data owner); forecast times are UTC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo
from importlib.metadata import version

import joblib
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from ai.models.baseline_forecasting import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRAIN_END_EXCLUSIVE,
    VALIDATION_END_EXCLUSIVE,
    VALIDATION_START,
    clip_predictions,
    train_turbine as train_scada_turbine,
    validation_metrics,
)
from ai.services.weather_client import (
    HOURLY_VARIABLES,
    MODEL,
    PROVIDER,
    OpenMeteoArchivedWeatherClient,
    WeatherClientError,
)


if __name__ == "__main__":
    sys.modules["ai.models.train"] = sys.modules[__name__]


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]
WEATHER_FEATURE_COLUMNS = (*HOURLY_VARIABLES, "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos")
LEAD_TIME_HOURS = (24, 48)
LEAD_TIME_TOLERANCE = timedelta(hours=3)
SCADA_TIMEZONE = "Asia/Almaty"
# Earlier Single Runs IFS data contains 49r1 hindcasts reconstructed before
# that cycle became operational on 2024-11-12. Exclude that segment.
EARLIEST_OPERATIONAL_RUN = datetime(2024, 11, 13, tzinfo=UTC)
ARCHIVED_SOURCE_DESCRIPTION = (
    "Open-Meteo Single Runs ECMWF IFS cached forecast values only. Every row "
    "uses a run available by issue_time under a 6-hour publication delay policy; "
    "actual weather and pre-operational cycle hindcasts are never substituted."
)


@dataclass(frozen=True)
class ForecastTrainingRow:
    valid_time: datetime
    issue_time: datetime
    weather_run_time: datetime
    lead_time_hours: int
    features: list[float]
    target: float
    scada_time: datetime | None = None
    weather_input_hash: str | None = None


class WeatherFeatureRegressor(RegressorMixin, BaseEstimator):
    """A serializable regressor that accepts named forecast-weather features."""

    def __init__(self, estimator: object, feature_columns: tuple[str, ...] = WEATHER_FEATURE_COLUMNS) -> None:
        self.estimator = estimator
        self.feature_columns = feature_columns

    def fit(self, rows: Iterable[dict[str, float]] | list[list[float]], targets: list[float]) -> "WeatherFeatureRegressor":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._matrix(rows), targets)
        return self

    def predict(self, rows: Iterable[dict[str, float]] | list[list[float]]) -> list[float]:
        return clip_predictions(self.estimator_.predict(self._matrix(rows)))

    def _matrix(self, rows: Iterable[dict[str, float]] | list[list[float]]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in rows:
            if isinstance(row, dict):
                values = [row[column] for column in self.feature_columns]
            else:
                values = list(row)
            if len(values) != len(self.feature_columns) or not all(math.isfinite(float(value)) for value in values):
                raise ValueError("Each weather feature row must contain finite configured feature values")
            matrix.append([float(value) for value in values])
        return matrix


WeatherFeatureRegressor.__module__ = "ai.models.train"


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("forecast timestamps must include an explicit UTC offset")
    return parsed.astimezone(UTC)


def _weather_values(hourly: dict[str, Any], index: int) -> list[float] | None:
    values: list[float] = []
    for column in HOURLY_VARIABLES:
        try:
            value = float(hourly[column][index])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def _cyclical_features(valid_time: datetime) -> list[float]:
    hour_angle = 2 * math.pi * valid_time.hour / 24
    day_angle = 2 * math.pi * valid_time.timetuple().tm_yday / 365.25
    return [math.sin(hour_angle), math.cos(hour_angle), math.sin(day_angle), math.cos(day_angle)]


def _scada_timestamp(value: str, zone: ZoneInfo) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        # During Kazakhstan's 2024 offset change one local hour repeats. A
        # naive timestamp cannot establish which UTC hour supplied that target.
        if timestamp.replace(tzinfo=zone, fold=0).utcoffset() != timestamp.replace(tzinfo=zone, fold=1).utcoffset():
            raise ValueError("Ambiguous local SCADA timestamp requires an explicit offset")
        timestamp = timestamp.replace(tzinfo=zone)
    return timestamp.astimezone(zone)


def _load_scada_targets(path: Path, scada_timezone: str = SCADA_TIMEZONE) -> dict[datetime, float]:
    targets: dict[datetime, float] = {}
    zone = ZoneInfo(scada_timezone)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                source_time = _scada_timestamp(row["timestamp"], zone)
                target = float(row[TARGET_COLUMN])
            except (KeyError, TypeError, ValueError):
                continue
            if source_time.replace(tzinfo=None) >= VALIDATION_END_EXCLUSIVE:
                continue
            valid_time = source_time.astimezone(UTC)
            if math.isfinite(target) and 0 <= target <= 1:
                if valid_time in targets and targets[valid_time] != target:
                    raise ValueError(f"Conflicting duplicate SCADA target: {source_time}")
                targets[valid_time] = target
    return targets


def _scada_training_range(path: Path, scada_timezone: str = SCADA_TIMEZONE) -> dict[str, str | int | None]:
    zone = ZoneInfo(scada_timezone)
    training_times = [
        timestamp.astimezone(zone) for timestamp in _load_scada_targets(path, scada_timezone)
        if timestamp.astimezone(zone).replace(tzinfo=None) < TRAIN_END_EXCLUSIVE
    ]
    return {
        "start": min(training_times).isoformat() if training_times else None,
        "end": max(training_times).isoformat() if training_times else None,
        "rows": len(training_times),
    }


def _cache_entries(cache_dir: Path, turbine_id: str, turbines_path: Path | None = None) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    if not cache_dir.is_dir():
        return [], 0
    entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    client = OpenMeteoArchivedWeatherClient(cache_dir=cache_dir, **({"turbines_path": turbines_path} if turbines_path else {}))
    rejected = 0
    for meta_path in sorted(cache_dir.glob("*.meta.json")):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("cache metadata must be an object")
            if metadata.get("turbine_id") != turbine_id:
                continue
            forecast = client.load_cached_forecast(meta_path)
            if forecast.metadata.weather_run_time < EARLIEST_OPERATIONAL_RUN:
                raise ValueError("pre-operational IFS hindcast cannot represent an available forecast")
        except (OSError, WeatherClientError, ValueError, TypeError, KeyError):
            rejected += 1
            continue
        entries.append((forecast.metadata.to_json(), forecast.raw_response))
    return entries, rejected


def load_archived_weather_rows(processed_path: Path, cache_dir: Path, turbine_id: str, *, scada_timezone: str = SCADA_TIMEZONE, turbines_path: Path | None = None) -> tuple[dict[int, list[ForecastTrainingRow]], dict[str, Any]]:
    """Join power only to safe cache records near 24h/48h issue-time leads."""
    targets = _load_scada_targets(processed_path, scada_timezone)
    zone = ZoneInfo(scada_timezone)
    candidates: dict[tuple[int, datetime], list[ForecastTrainingRow]] = {}
    rejected = 0
    entries, rejected_entries = _cache_entries(cache_dir, turbine_id, turbines_path)
    for metadata, raw in entries:
        issue_time = _parse_utc(metadata["issue_time"])
        weather_run_time = _parse_utc(metadata["weather_run_time"])
        hourly = raw["hourly"]
        times = hourly.get("time", [])
        if not isinstance(times, list):
            continue
        for index, epoch in enumerate(times):
            try:
                valid_time = datetime.fromtimestamp(float(epoch), UTC)
            except (TypeError, ValueError, OSError, OverflowError):
                rejected += 1
                continue
            target = targets.get(valid_time)
            values = _weather_values(hourly, index)
            if target is None or values is None or valid_time <= issue_time:
                rejected += 1
                continue
            lead = valid_time - issue_time
            for requested_hours in LEAD_TIME_HOURS:
                if abs(lead - timedelta(hours=requested_hours)) <= LEAD_TIME_TOLERANCE:
                    row = ForecastTrainingRow(
                        valid_time, issue_time, weather_run_time, requested_hours,
                        values + _cyclical_features(valid_time), target, valid_time.astimezone(zone), metadata["input_hash"],
                    )
                    candidates.setdefault((requested_hours, valid_time), []).append(row)
    result: dict[int, list[ForecastTrainingRow]] = {lead: [] for lead in LEAD_TIME_HOURS}
    for (lead, _), choices in candidates.items():
        # Prefer the issue time closest to requested lead; ties use the newest safe issue.
        result[lead].append(min(
            choices,
            key=lambda row: (abs((row.valid_time - row.issue_time).total_seconds() / 3600 - lead), -row.issue_time.timestamp()),
        ))
    for rows in result.values():
        rows.sort(key=lambda row: row.valid_time)
    return result, {
        "cache_entries": len(entries), "rejected_cache_entries": rejected_entries, "rejected_rows": rejected,
        "verified_cache_input_hashes": sorted({metadata["input_hash"] for metadata, _ in entries}),
    }


def _split_rows(rows: Iterable[ForecastTrainingRow]) -> tuple[list[ForecastTrainingRow], list[ForecastTrainingRow]]:
    train, validation = [], []
    for row in rows:
        naive_time = (row.scada_time or row.valid_time).replace(tzinfo=None)
        if naive_time < TRAIN_END_EXCLUSIVE:
            train.append(row)
        elif VALIDATION_START <= naive_time < VALIDATION_END_EXCLUSIVE:
            validation.append(row)
    return train, validation


def _weather_candidates() -> dict[str, WeatherFeatureRegressor]:
    return {
        "extra_trees": WeatherFeatureRegressor(ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1,
        )),
        "hist_gradient_boosting": WeatherFeatureRegressor(HistGradientBoostingRegressor(
            max_iter=300, l2_regularization=0.1, random_state=RANDOM_SEED, early_stopping=False,
        )),
    }


def _row_dict(row: ForecastTrainingRow) -> dict[str, float]:
    return dict(zip(WEATHER_FEATURE_COLUMNS, row.features, strict=True))


def _software_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{package: version(package) for package in ("scikit-learn", "joblib", "numpy")}}


def _save_training_rows(path: Path, train_rows: list[ForecastTrainingRow], validation_rows: list[ForecastTrainingRow], selection_available_at: datetime) -> dict[str, Any]:
    """Save the exact joined rows so issue/run/valid provenance can be audited."""
    fields = ["phase", "issue_time", "weather_run_time", "valid_time", "scada_time", "lead_group_hours", "actual_lead_hours", "weather_input_hash", TARGET_COLUMN, *WEATHER_FEATURE_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for training, rows in ((True, train_rows), (False, validation_rows)):
            for row in rows:
                phase = "training" if training else "validation_selection" if row.valid_time + timedelta(hours=1) <= selection_available_at else "validation_reporting_only"
                writer.writerow({
                    "phase": phase,
                    "issue_time": row.issue_time.isoformat(),
                    "weather_run_time": row.weather_run_time.isoformat(),
                    "valid_time": row.valid_time.isoformat(),
                    "scada_time": row.scada_time.isoformat() if row.scada_time else "",
                    "lead_group_hours": row.lead_time_hours,
                    "actual_lead_hours": (row.valid_time - row.issue_time).total_seconds() / 3600,
                    "weather_input_hash": row.weather_input_hash or "",
                    TARGET_COLUMN: row.target,
                    **_row_dict(row),
                })
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(train_rows) + len(validation_rows),
        "used_weather_cache_input_hashes": sorted({row.weather_input_hash for row in train_rows + validation_rows if row.weather_input_hash}),
    }


def train_archived_turbine(rows_by_lead: dict[int, list[ForecastTrainingRow]], artifact_path: Path, *, scada_timezone: str = SCADA_TIMEZONE) -> dict[str, Any] | None:
    """Train on pooled 24/48h rows and compare models on the same holdout."""
    eligible: dict[int, tuple[list[ForecastTrainingRow], list[ForecastTrainingRow]]] = {}
    for lead, rows in sorted(rows_by_lead.items()):
        train_rows, validation_rows = _split_rows(rows)
        if len(train_rows) >= 2 and len(validation_rows) >= 2:
            eligible[lead] = train_rows, validation_rows
    if not eligible:
        return None
    train_rows = sorted([row for training, _ in eligible.values() for row in training], key=lambda row: (row.valid_time, row.lead_time_hours))
    validation_rows = sorted([row for _, validation in eligible.values() for row in validation], key=lambda row: (row.valid_time, row.lead_time_hours))
    # The first 24h-ahead February forecast is issued Jan 31 at 00:00 local.
    # January targets ending after that instant may be reported as validation
    # metrics, but must not influence the selected production model.
    selection_available_at = (VALIDATION_END_EXCLUSIVE - timedelta(days=1)).replace(tzinfo=ZoneInfo(scada_timezone)).astimezone(UTC)
    selection_rows = [row for row in validation_rows if row.valid_time + timedelta(hours=1) <= selection_available_at]
    if len(selection_rows) < 2:
        return None
    candidates = _weather_candidates()
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for name, model in candidates.items():
        model.fit([_row_dict(row) for row in train_rows], [row.target for row in train_rows])
        metrics = validation_metrics([row.target for row in validation_rows], model.predict([_row_dict(row) for row in validation_rows]))
        selection_metrics = validation_metrics([row.target for row in selection_rows], model.predict([_row_dict(row) for row in selection_rows]))
        by_lead = {
            str(lead): {
                "rows": len(validation),
                "metrics": validation_metrics([row.target for row in validation], model.predict([_row_dict(row) for row in validation])),
            }
            for lead, (_, validation) in eligible.items()
        }
        candidate_metrics[name] = {"metrics": metrics, "selection_metrics": selection_metrics, "validation_by_lead": by_lead}
    name = min(candidates, key=lambda key: (candidate_metrics[key]["selection_metrics"]["mae"], key))
    selected = candidates[name]
    selected.model_source_ = "archived_open_meteo_single_runs"
    selected.available_at_ = selection_available_at.isoformat()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, artifact_path, compress=3)
    audit = _save_training_rows(artifact_path.with_name(f"{artifact_path.stem}_training_rows.csv"), train_rows, validation_rows, selection_available_at)
    return {
        "model_source": selected.model_source_,
        "artifact": artifact_path.name,
        "available_at": selected.available_at_,
        "forecast_source": {"provider": PROVIDER, "model": MODEL, "api": "single_runs"},
        "lead_time_hours": sorted(eligible),
        "lead_time_semantics": "valid_time - issue_time within +/-3h of 24h or 48h; model-run age is separately recorded. Both lead groups are pooled, not selected against unequal holdouts.",
        "training_date_range": {"start": min(row.valid_time for row in train_rows).isoformat(), "end": max(row.valid_time for row in train_rows).isoformat(), "rows": len(train_rows), "unique_target_hours": len({row.valid_time for row in train_rows})},
        "validation_period": {"start": VALIDATION_START.replace(tzinfo=ZoneInfo(scada_timezone)).isoformat(), "end_exclusive": VALIDATION_END_EXCLUSIVE.replace(tzinfo=ZoneInfo(scada_timezone)).isoformat(), "rows": len(validation_rows), "unique_target_hours": len({row.valid_time for row in validation_rows})},
        "features": list(WEATHER_FEATURE_COLUMNS),
        "feature_timezone": "UTC",
        "random_seed": RANDOM_SEED,
        "software_versions": _software_versions(),
        "candidate_parameters": {candidate_name: model.estimator.get_params(deep=False) for candidate_name, model in candidates.items()},
        "training_rows_audit": audit,
        "selected_model": name,
        "selection_basis": "lowest clipped January MAE on identical pooled rows available by Jan31 00:00 station time; later January targets are reporting-only; January targets are never used for fitting",
        "selection_validation_range": {"start": min(row.valid_time for row in selection_rows).isoformat(), "end": max(row.valid_time for row in selection_rows).isoformat(), "rows": len(selection_rows)},
        "selection_metrics": candidate_metrics[name]["selection_metrics"],
        "metrics": candidate_metrics[name]["metrics"],
        "candidate_metrics": candidate_metrics,
        "inference_limitation": "Validation covers approximately 24h and 48h issue-time leads; other hours within the 1-48h production horizon have not been separately validated.",
    }


def _baseline_report(processed_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Reuse existing measured-weather artifacts; never overwrite them in this trainer."""
    report_path = artifacts_dir / "validation_metrics.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if all((artifacts_dir / f"turbine_{number}.joblib").is_file() and f"turbine_{number}" in report.get("turbines", {}) for number in (1, 2)):
            return report
    results = {}
    for number in (1, 2):
        turbine_id = f"turbine_{number}"
        artifact_path = artifacts_dir / f"{turbine_id}.joblib"
        if artifact_path.exists():
            raise ValueError("Existing baseline artifacts lack complete validation_metrics.json; run baseline_forecasting explicitly to regenerate their metrics")
        results[turbine_id] = train_scada_turbine(processed_dir / f"{turbine_id}_hourly.csv", artifact_path)
    report = {"baseline_type": "historical_measured_scada_weather", "turbines": results}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _matched_baseline_metrics(processed_path: Path, artifact_path: Path, rows_by_lead: dict[int, list[ForecastTrainingRow]], scada_timezone: str) -> dict[str, Any]:
    """Compare the measured-weather baseline on exactly the archive holdout rows."""
    feature_lookup = {}
    zone = ZoneInfo(scada_timezone)
    with processed_path.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            try:
                timestamp = _scada_timestamp(record["timestamp"], zone).astimezone(UTC)
                values = [float(record[column]) for column in FEATURE_COLUMNS]
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values):
                feature_lookup[timestamp] = values
    validation = []
    for rows in rows_by_lead.values():
        training, holdout = _split_rows(rows)
        if len(training) >= 2 and len(holdout) >= 2:
            validation.extend(holdout)
    common = [row for row in validation if row.valid_time in feature_lookup]
    if len(common) != len(validation) or len(common) < 2:
        return {"limitation": "Matched comparison unavailable: measured-weather features are missing for archive validation targets", "rows": len(common)}
    baseline_model = joblib.load(artifact_path)
    return {
        "rows": len(common),
        "unique_target_hours": len({row.valid_time for row in common}),
        "metrics": validation_metrics([row.target for row in common], baseline_model.predict([feature_lookup[row.valid_time] for row in common])),
        "interpretation": "Same targets and lead weighting as archived model; baseline uses measured future SCADA weather and is an optimistic diagnostic, not a deployable historical forecast.",
    }


def train_all(processed_dir: Path, cache_dir: Path, artifacts_dir: Path, metrics_path: Path, *, scada_timezone: str = SCADA_TIMEZONE, turbines_path: Path | None = None) -> dict[str, Any]:
    """Preserve measured baselines and train separate archived-weather artifacts."""
    baseline = _baseline_report(processed_dir, artifacts_dir)
    zone = ZoneInfo(scada_timezone)
    available_at = VALIDATION_END_EXCLUSIVE.replace(tzinfo=zone).astimezone(UTC).isoformat()
    selected: dict[str, Any] = {}
    archived: dict[str, Any] = {}
    training_ranges: dict[str, dict[str, str | int | None]] = {}
    for number in (1, 2):
        turbine_id = f"turbine_{number}"
        processed_path = processed_dir / f"{turbine_id}_hourly.csv"
        training_ranges[turbine_id] = _scada_training_range(processed_path, scada_timezone)
        rows_by_lead, cache_stats = load_archived_weather_rows(
            processed_path, cache_dir, turbine_id, scada_timezone=scada_timezone, turbines_path=turbines_path,
        )
        trained = train_archived_turbine(rows_by_lead, artifacts_dir / f"{turbine_id}_archived.joblib", scada_timezone=scada_timezone)
        if trained:
            trained["input_processed_csv"] = {"file": processed_path.name, "sha256": hashlib.sha256(processed_path.read_bytes()).hexdigest()}
        archived[turbine_id] = {
            "cache_statistics": cache_stats,
            "rows_by_lead": {str(lead): len(rows) for lead, rows in rows_by_lead.items()},
            "result": trained,
            "matched_scada_weather_baseline": _matched_baseline_metrics(processed_path, artifacts_dir / f"{turbine_id}.joblib", rows_by_lead, scada_timezone) if trained else None,
        }
        selected[turbine_id] = trained or {
            "model_source": "scada_weather_baseline_fallback",
            "artifact": f"{turbine_id}.joblib",
            "available_at": available_at,
            "selected_model": baseline["turbines"][turbine_id]["selected_model"],
            "metrics": baseline["turbines"][turbine_id]["metrics"][baseline["turbines"][turbine_id]["selected_model"]],
            "training_date_range": training_ranges[turbine_id],
            "validation_period": {"start": VALIDATION_START.replace(tzinfo=zone).isoformat(), "end_exclusive": VALIDATION_END_EXCLUSIVE.replace(tzinfo=zone).isoformat()},
            "features": list(FEATURE_COLUMNS),
            "feature_timezone": scada_timezone,
            "limitation": "No archived forecast rows cover both training and January 2026 validation; measured-SCADA-weather baseline retained as fallback.",
        }
    report = {
        "training_timestamp_convention": f"Naive SCADA timestamps are station local time in {scada_timezone} (confirmed by data owner); explicit offsets are respected; archive joins use UTC.",
        "scada_timezone": scada_timezone,
        "random_seed": RANDOM_SEED,
        "software_versions": _software_versions(),
        "forecast_source": {"provider": PROVIDER, "model": MODEL, "api": "single_runs", "cache_dir": str(cache_dir)},
        "archived_weather_description": ARCHIVED_SOURCE_DESCRIPTION,
        "lead_time_semantics": "Forecast lead is valid_time - issue_time. Only cached rows within +/-3h of 24h or 48h are eligible; both available lead groups are pooled.",
        "earliest_operational_weather_run": EARLIEST_OPERATIONAL_RUN.isoformat(),
        "model_selection_available_at": max(result["available_at"] for result in selected.values()),
        "training_date_range": training_ranges,
        "features": list(WEATHER_FEATURE_COLUMNS),
        "validation_period": {"start": VALIDATION_START.replace(tzinfo=zone).isoformat(), "end_exclusive": VALIDATION_END_EXCLUSIVE.replace(tzinfo=zone).isoformat()},
        "scada_weather_baseline": baseline,
        "archived_weather": archived,
        "selected_models": selected,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe models from cached Open-Meteo Single Runs.")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--weather-cache", type=Path, default=REPO_ROOT / "data" / "weather_cache")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "ai" / "models" / "artifacts")
    parser.add_argument("--metrics", type=Path, default=REPO_ROOT / "ai" / "models" / "artifacts" / "metrics.json")
    parser.add_argument("--scada-timezone", default=SCADA_TIMEZONE, help="IANA timezone for naive station-local SCADA timestamps")
    parser.add_argument("--turbines", type=Path, default=REPO_ROOT / "data" / "turbines.json")
    args = parser.parse_args()
    report = train_all(args.processed_dir, args.weather_cache, args.artifacts_dir, args.metrics, scada_timezone=args.scada_timezone, turbines_path=args.turbines)
    for turbine, result in report["selected_models"].items():
        print(f"{turbine}: {result['model_source']}; {result['selected_model']}")


if __name__ == "__main__":
    main()
