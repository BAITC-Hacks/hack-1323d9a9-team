"""Train production candidates from archived, issue-time-safe weather forecasts.

The archive is populated by :mod:`ai.services.weather_client`.  This trainer
never calls a weather observation API and never substitutes measured SCADA
weather for a missing forecast.  Processed SCADA timestamps are interpreted as
UTC because Open-Meteo cached valid times are UTC and the source has no
timezone field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

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
    train_all as train_scada_baseline,
    validation_metrics,
)
from ai.services.weather_client import (
    DEFAULT_PUBLICATION_SAFETY_DELAY,
    HOURLY_VARIABLES,
    MODEL,
    PROVIDER,
    assert_no_leakage,
)


if __name__ == "__main__":
    sys.modules["ai.models.train"] = sys.modules[__name__]


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]
WEATHER_FEATURE_COLUMNS = (*HOURLY_VARIABLES, "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos")
LEAD_TIME_HOURS = (24, 48)
LEAD_TIME_TOLERANCE = timedelta(hours=3)
ARCHIVED_SOURCE_DESCRIPTION = (
    "Open-Meteo Previous Runs cached forecast values only. Every row uses a "
    "weather_run_time available by issue_time; actual weather is never substituted."
)


@dataclass(frozen=True)
class ForecastTrainingRow:
    valid_time: datetime
    issue_time: datetime
    weather_run_time: datetime
    lead_time_hours: int
    features: list[float]
    target: float


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
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


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


def _load_scada_targets(path: Path) -> dict[datetime, float]:
    targets: dict[datetime, float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                valid_time = _parse_utc(row["timestamp"])
                target = float(row[TARGET_COLUMN])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(target):
                targets[valid_time] = min(1.0, max(0.0, target))
    return targets


def _scada_training_range(path: Path) -> dict[str, str | int | None]:
    training_times = [
        timestamp for timestamp in _load_scada_targets(path)
        if timestamp.replace(tzinfo=None) < TRAIN_END_EXCLUSIVE
    ]
    return {
        "start": min(training_times).isoformat() if training_times else None,
        "end": max(training_times).isoformat() if training_times else None,
        "rows": len(training_times),
    }


def _cache_entries(cache_dir: Path, turbine_id: str) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    if not cache_dir.is_dir():
        return ()
    entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for meta_path in sorted(cache_dir.glob("*.meta.json")):
        raw_path = cache_dir / f"{meta_path.name[:-10]}.json"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("turbine_id") != turbine_id or not raw_path.is_file():
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            issue_time = _parse_utc(metadata["issue_time"])
            weather_run_time = _parse_utc(metadata["weather_run_time"])
            assert_no_leakage(weather_run_time, issue_time, DEFAULT_PUBLICATION_SAFETY_DELAY)
            if metadata.get("provider") != PROVIDER or metadata.get("model") != MODEL:
                continue
            if not isinstance(raw.get("hourly"), dict):
                continue
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            continue
        entries.append((metadata, raw))
    return entries


def load_archived_weather_rows(processed_path: Path, cache_dir: Path, turbine_id: str) -> tuple[dict[int, list[ForecastTrainingRow]], dict[str, int]]:
    """Join power only to safe cache records near 24h/48h issue-time leads."""
    targets = _load_scada_targets(processed_path)
    candidates: dict[tuple[int, datetime], list[ForecastTrainingRow]] = {}
    rejected = 0
    for metadata, raw in _cache_entries(cache_dir, turbine_id):
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
                        values + _cyclical_features(valid_time), target,
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
    return result, {"cache_entries": sum(1 for _ in _cache_entries(cache_dir, turbine_id)), "rejected_rows": rejected}


def _split_rows(rows: Iterable[ForecastTrainingRow]) -> tuple[list[ForecastTrainingRow], list[ForecastTrainingRow]]:
    train, validation = [], []
    for row in rows:
        naive_time = row.valid_time.replace(tzinfo=None)
        if naive_time < TRAIN_END_EXCLUSIVE:
            train.append(row)
        elif VALIDATION_START <= naive_time < VALIDATION_END_EXCLUSIVE:
            validation.append(row)
    return train, validation


def _weather_candidates() -> dict[str, WeatherFeatureRegressor]:
    return {
        "extra_trees": WeatherFeatureRegressor(ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1,
        )),
        "hist_gradient_boosting": WeatherFeatureRegressor(HistGradientBoostingRegressor(
            max_iter=300, l2_regularization=0.1, random_state=RANDOM_SEED,
        )),
    }


def _row_dict(row: ForecastTrainingRow) -> dict[str, float]:
    return dict(zip(WEATHER_FEATURE_COLUMNS, row.features, strict=True))


def train_archived_turbine(rows_by_lead: dict[int, list[ForecastTrainingRow]], artifact_path: Path) -> dict[str, Any] | None:
    """Evaluate each available lead/model pair; save the lowest-MAE safe model."""
    candidates: list[tuple[float, int, str, WeatherFeatureRegressor, dict[str, float], list[ForecastTrainingRow], list[ForecastTrainingRow]]] = []
    for lead, rows in rows_by_lead.items():
        train_rows, validation_rows = _split_rows(rows)
        if len(train_rows) < 2 or len(validation_rows) < 2:
            continue
        train_features, train_targets = [_row_dict(row) for row in train_rows], [row.target for row in train_rows]
        validation_features, validation_targets = [_row_dict(row) for row in validation_rows], [row.target for row in validation_rows]
        for name, model in _weather_candidates().items():
            model.fit(train_features, train_targets)
            metrics = validation_metrics(validation_targets, model.predict(validation_features))
            candidates.append((metrics["mae"], lead, name, model, metrics, train_rows, validation_rows))
    if not candidates:
        return None
    _, lead, name, selected, metrics, train_rows, validation_rows = min(candidates, key=lambda candidate: candidate[0])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, artifact_path, compress=3)
    return {
        "model_source": "archived_open_meteo_previous_runs",
        "artifact": str(artifact_path),
        "forecast_source": {"provider": PROVIDER, "model": MODEL},
        "lead_time_hours": lead,
        "lead_time_semantics": "valid_time - issue_time; issue_time is approximately 24h or 48h before valid_time (±3h).",
        "training_date_range": {"start": min(row.valid_time for row in train_rows).isoformat(), "end": max(row.valid_time for row in train_rows).isoformat(), "rows": len(train_rows)},
        "validation_period": {"start": VALIDATION_START.isoformat(sep=" "), "end_exclusive": VALIDATION_END_EXCLUSIVE.isoformat(sep=" "), "rows": len(validation_rows)},
        "features": list(WEATHER_FEATURE_COLUMNS),
        "selected_model": name,
        "metrics": metrics,
        "candidate_metrics": [
            {"lead_time_hours": candidate_lead, "model": candidate_name, "metrics": candidate_metrics}
            for _, candidate_lead, candidate_name, _, candidate_metrics, _, _ in candidates
        ],
    }


def train_all(processed_dir: Path, cache_dir: Path, artifacts_dir: Path, metrics_path: Path) -> dict[str, Any]:
    """Train baselines and, when cache coverage exists, forecast-safe production models."""
    baseline = train_scada_baseline(processed_dir, artifacts_dir, artifacts_dir / "validation_metrics.json")
    selected: dict[str, Any] = {}
    archived: dict[str, Any] = {}
    training_ranges: dict[str, dict[str, str | int | None]] = {}
    for number in (1, 2):
        turbine_id = f"turbine_{number}"
        processed_path = processed_dir / f"{turbine_id}_hourly.csv"
        training_ranges[turbine_id] = _scada_training_range(processed_path)
        rows_by_lead, cache_stats = load_archived_weather_rows(
            processed_path, cache_dir, turbine_id
        )
        trained = train_archived_turbine(rows_by_lead, artifacts_dir / f"{turbine_id}.joblib")
        archived[turbine_id] = {
            "cache_statistics": cache_stats,
            "rows_by_lead": {str(lead): len(rows) for lead, rows in rows_by_lead.items()},
            "result": trained,
        }
        selected[turbine_id] = trained or {
            "model_source": "scada_weather_baseline_fallback",
            "artifact": baseline["turbines"][turbine_id]["artifact"],
            "selected_model": baseline["turbines"][turbine_id]["selected_model"],
            "metrics": baseline["turbines"][turbine_id]["metrics"][baseline["turbines"][turbine_id]["selected_model"]],
            "training_date_range": training_ranges[turbine_id],
            "validation_period": {"start": VALIDATION_START.isoformat(sep=" "), "end_exclusive": VALIDATION_END_EXCLUSIVE.isoformat(sep=" ")},
            "features": list(FEATURE_COLUMNS),
            "limitation": "No archived forecast rows cover both training and January 2026 validation; measured-SCADA-weather baseline retained as fallback.",
        }
    report = {
        "training_timestamp_convention": "Processed SCADA timestamps are treated as UTC.",
        "forecast_source": {"provider": PROVIDER, "model": MODEL, "cache_dir": str(cache_dir)},
        "archived_weather_description": ARCHIVED_SOURCE_DESCRIPTION,
        "lead_time_semantics": "Forecast lead is valid_time - issue_time. Only cached rows within ±3h of 24h or 48h are eligible.",
        "training_date_range": training_ranges,
        "features": list(WEATHER_FEATURE_COLUMNS),
        "validation_period": {"start": VALIDATION_START.isoformat(sep=" "), "end_exclusive": VALIDATION_END_EXCLUSIVE.isoformat(sep=" ")},
        "scada_weather_baseline": baseline,
        "archived_weather": archived,
        "selected_models": selected,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train leakage-safe models from cached Open-Meteo Previous Runs.")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--weather-cache", type=Path, default=REPO_ROOT / "data" / "weather_cache")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "ai" / "models" / "artifacts")
    parser.add_argument("--metrics", type=Path, default=REPO_ROOT / "ai" / "models" / "artifacts" / "metrics.json")
    args = parser.parse_args()
    report = train_all(args.processed_dir, args.weather_cache, args.artifacts_dir, args.metrics)
    for turbine, result in report["selected_models"].items():
        print(f"{turbine}: {result['model_source']}; {result['selected_model']}")


if __name__ == "__main__":
    main()
