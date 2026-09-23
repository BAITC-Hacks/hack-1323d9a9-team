"""Issue-time-safe, daily rolling forecasts for the February 2026 test month.

No February SCADA target or measured weather is read by this module. The full
per-issue forecasts and the one-row-per-valid-hour evaluation view are separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from ai.services.forecast_agent import ForecastAgent, ForecastResult, TURBINE_IDS
from ai.services.weather_client import (
    DEFAULT_PUBLICATION_SAFETY_DELAY,
    FORECAST_SOURCE,
    OpenMeteoArchivedWeatherClient,
    REPO_ROOT,
)
from backend.app.services.model_loader import ArtifactModelLoader


UTC = timezone.utc
STATION_TIMEZONE = ZoneInfo("Asia/Almaty")
FIRST_ISSUE_LOCAL = datetime(2026, 1, 31, tzinfo=STATION_TIMEZONE)
LAST_ISSUE_LOCAL = datetime(2026, 2, 28, tzinfo=STATION_TIMEZONE)
FEBRUARY_START = datetime(2026, 2, 1, tzinfo=STATION_TIMEZONE).astimezone(UTC)
FEBRUARY_END = datetime(2026, 3, 1, tzinfo=STATION_TIMEZONE).astimezone(UTC)
HORIZON_HOURS = 48
MIN_EVALUATION_LEAD_HOURS = 24
CSV_COLUMNS = (
    "issue_time", "weather_run_time", "valid_time", "turbine_id", "lead_hour",
    "predicted_normalized_power", "model_version", "weather_input_hash",
    "weather_response_sha256", "forecast_input_hash", "model_available_at",
)


class BacktestIntegrityError(ValueError):
    """A run cannot be proven to have used only available historical inputs."""


def _utc(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BacktestIntegrityError(f"{name} is not a valid timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise BacktestIntegrityError(f"{name} must have a timezone")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def official_issues() -> list[datetime]:
    """Midnight station-local issues from Jan 31 through Feb 28 inclusive."""
    return [(FIRST_ISSUE_LOCAL + timedelta(days=day)).astimezone(UTC) for day in range(29)]


def _check_schedule(issues: Sequence[datetime]) -> list[datetime]:
    if not issues:
        raise ValueError("At least one issue time is required")
    checked = [_utc(issue, "issue_time") for issue in issues]
    allowed = set(official_issues())
    if any(issue not in allowed for issue in checked):
        raise ValueError("Issues must be local midnights from 2026-01-31 through 2026-02-28")
    if any(later - earlier != timedelta(hours=24) for earlier, later in zip(checked, checked[1:])):
        raise ValueError("Issues must be strictly ordered, consecutive local days")
    return checked


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BacktestIntegrityError(f"{name} must be a SHA-256 hex digest")
    return value


def _validated_rows(result: ForecastResult, issue: datetime) -> list[dict[str, Any]]:
    if result.status != "COMPLETE" or not result.output_path or not result.output_path.is_file():
        raise BacktestIntegrityError("run has no saved COMPLETE forecast artifact")
    try:
        saved = json.loads(result.output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BacktestIntegrityError("saved forecast artifact is unreadable") from exc
    if saved != result.to_json():
        raise BacktestIntegrityError("saved forecast artifact differs from the returned run")
    metadata = result.metadata
    if _utc(metadata.get("issue_time"), "metadata.issue_time") != issue:
        raise BacktestIntegrityError("forecast issue_time differs from scheduled issue")
    if metadata.get("horizon_hours") != HORIZON_HOURS or set(metadata.get("turbines", ())) != set(TURBINE_IDS):
        raise BacktestIntegrityError("forecast does not contain the two-turbine 48-hour horizon")
    weather = metadata.get("weather", {})
    models = metadata.get("models", {})
    details = metadata.get("model_details", {})
    if any(not isinstance(mapping, dict) or set(mapping) != set(TURBINE_IDS) for mapping in (weather, models, details)):
        raise BacktestIntegrityError("missing turbine weather or model provenance")
    forecast_hash = _sha256(metadata.get("input_hash"), "forecast_input_hash")
    provenance = {}
    for turbine_id in TURBINE_IDS:
        source = weather[turbine_id]
        if not isinstance(source, dict) or not isinstance(details[turbine_id], dict):
            raise BacktestIntegrityError(f"{turbine_id}: invalid weather or model provenance")
        if source.get("forecast_source") != FORECAST_SOURCE:
            raise BacktestIntegrityError(f"{turbine_id}: weather is not an archived forecast run")
        run_time = _utc(source.get("weather_run_time"), "weather_run_time")
        delay_seconds = source.get("publication_safety_delay_seconds")
        if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, (int, float)) or not math.isfinite(delay_seconds) or delay_seconds < 0:
            raise BacktestIntegrityError(f"{turbine_id}: missing weather publication delay")
        delay = max(DEFAULT_PUBLICATION_SAFETY_DELAY, timedelta(seconds=delay_seconds))
        if run_time.minute or run_time.second or run_time.microsecond or run_time.hour % 6 or run_time + delay > issue:
            raise BacktestIntegrityError(f"{turbine_id}: weather run was not safely available at issue_time")
        model = details[turbine_id]
        if model.get("model_source") != "archived_open_meteo_single_runs":
            raise BacktestIntegrityError(f"{turbine_id}: selected model is not archived-weather trained")
        available_at = _utc(model.get("available_at"), "model.available_at")
        if available_at > issue:
            raise BacktestIntegrityError(f"{turbine_id}: model selection uses future January targets")
        if not isinstance(models[turbine_id], str) or not models[turbine_id]:
            raise BacktestIntegrityError(f"{turbine_id}: missing model version")
        provenance[turbine_id] = {
            "weather_run_time": _iso(run_time),
            "model_available_at": _iso(available_at),
            "weather_input_hash": _sha256(source.get("input_hash"), "weather_input_hash"),
            "weather_response_sha256": _sha256(source.get("raw_response_sha256"), "weather_response_sha256"),
        }
    rows = []
    seen = set()
    for point in result.forecasts:
        if point.turbine_id not in TURBINE_IDS:
            raise BacktestIntegrityError("unknown turbine in forecast")
        valid_time = _utc(point.valid_time, "valid_time")
        seconds = (valid_time - issue).total_seconds()
        lead_hour = int(seconds // 3600)
        if seconds != lead_hour * 3600 or not 1 <= lead_hour <= HORIZON_HOURS:
            raise BacktestIntegrityError("valid_time has an invalid or fractional lead hour")
        if isinstance(point.prediction, bool) or not isinstance(point.prediction, (int, float)) or not math.isfinite(point.prediction) or not 0 <= point.prediction <= 1:
            raise BacktestIntegrityError("predicted power must be finite and in [0,1]")
        key = (point.turbine_id, lead_hour)
        if key in seen:
            raise BacktestIntegrityError("duplicate turbine and lead hour")
        seen.add(key)
        rows.append({
            "issue_time": _iso(issue),
            "weather_run_time": provenance[point.turbine_id]["weather_run_time"],
            "valid_time": _iso(valid_time),
            "turbine_id": point.turbine_id,
            "lead_hour": lead_hour,
            "predicted_normalized_power": float(point.prediction),
            "model_version": models[point.turbine_id],
            "weather_input_hash": provenance[point.turbine_id]["weather_input_hash"],
            "weather_response_sha256": provenance[point.turbine_id]["weather_response_sha256"],
            "forecast_input_hash": forecast_hash,
            "model_available_at": provenance[point.turbine_id]["model_available_at"],
        })
    if seen != {(turbine, hour) for turbine in TURBINE_IDS for hour in range(1, HORIZON_HOURS + 1)}:
        raise BacktestIntegrityError("forecast has missing turbine or valid-hour rows")
    return sorted(rows, key=lambda row: (row["valid_time"], row["turbine_id"]))


def _select_february(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    eligible = 0
    for row in rows:
        valid = _utc(row["valid_time"], "valid_time")
        if not FEBRUARY_START <= valid < FEBRUARY_END or not MIN_EVALUATION_LEAD_HOURS <= row["lead_hour"] <= HORIZON_HOURS:
            continue
        eligible += 1
        key = row["turbine_id"], row["valid_time"]
        previous = selected.get(key)
        if previous is None or row["issue_time"] > previous["issue_time"]:
            selected[key] = row
        elif row["issue_time"] == previous["issue_time"]:
            raise BacktestIntegrityError("duplicate eligible issue/turbine/valid_time")
    return [selected[key] for key in sorted(selected, key=lambda item: (item[1], item[0]))], eligible


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def simulate_rolling(agent: ForecastAgent, output_dir: Path, issues: Sequence[datetime] | None = None) -> dict[str, Any]:
    """Run sequential issues; retain all horizons and a deterministic February view."""
    scheduled = _check_schedule(official_issues() if issues is None else issues)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    rows: list[dict[str, Any]] = []
    for issue in scheduled:
        result = agent.run(issue, horizon_hours=HORIZON_HOURS)
        artifact = None
        if result.output_path:
            try:
                artifact = result.output_path.resolve().relative_to(output_dir.resolve()).as_posix()
            except ValueError:
                artifact = str(result.output_path)
        run = {"issue_time": _iso(issue), "status": result.status, "message": result.message,
               "artifact": artifact, "forecast_rows": 0}
        if result.status == "COMPLETE":
            try:
                checked = _validated_rows(result, issue)
            except (BacktestIntegrityError, KeyError, TypeError) as exc:
                run.update(status="FAILED", message=f"Leakage/integrity check failed: {exc}")
            else:
                rows.extend(checked)
                run["forecast_rows"] = len(checked)
        runs.append(run)
        print(f"{_iso(issue)} {run['status']} ({run['forecast_rows']} rows)", flush=True)
    evaluation_rows, eligible_count = _select_february(rows)
    expected_hours = int((FEBRUARY_END - FEBRUARY_START).total_seconds() / 3600)
    selected_hours = {turbine: sum(row["turbine_id"] == turbine for row in evaluation_rows) for turbine in TURBINE_IDS}
    official_schedule = scheduled == official_issues()
    report = {
        "status": "COMPLETE" if official_schedule and all(run["status"] == "COMPLETE" for run in runs) and all(value == expected_hours for value in selected_hours.values()) else "INCOMPLETE",
        "station_timezone": str(STATION_TIMEZONE),
        "first_issue_time": _iso(scheduled[0]), "last_issue_time": _iso(scheduled[-1]),
        "february_start": _iso(FEBRUARY_START), "february_end_exclusive": _iso(FEBRUARY_END),
        "issue_count": len(scheduled), "complete_runs": sum(run["status"] == "COMPLETE" for run in runs),
        "forecast_rows": len(rows), "february_eligible_rows": eligible_count,
        "evaluation_rows": len(evaluation_rows), "expected_hours_per_turbine": expected_hours,
        "selected_hours_per_turbine": selected_hours, "official_schedule": official_schedule,
        "horizon_hours": HORIZON_HOURS, "minimum_evaluation_lead_hours": MIN_EVALUATION_LEAD_HOURS,
        "overlap_selection": "For each turbine and February valid hour, use the latest issue_time with 24 <= lead_hour <= 48.",
        "leakage_checks": "Saved result matches artifact; archived forecast source and SHA-256; weather_run_time + publication delay (at least 6h) <= issue_time < valid_time; archived model available_at <= issue_time; exact 1-48h leads; complete turbines; finite clipped power.",
        "evaluation": "Prediction-only February view. February actual power is unavailable; no February accuracy metric is claimed.",
        "runs": runs,
    }
    _write_csv(output_dir / "rolling_forecasts.csv", rows)
    _write_csv(output_dir / "february_evaluation.csv", evaluation_rows)
    _write_json(output_dir / "rolling_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "forecasts")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "weather_cache")
    parser.add_argument("--cache-only", action="store_true", help="Fail explicitly when a historical run is not cached")
    args = parser.parse_args()
    agent = ForecastAgent(
        weather_client=OpenMeteoArchivedWeatherClient(cache_dir=args.cache_dir, cache_only=args.cache_only),
        model_loader=ArtifactModelLoader(), output_dir=args.output_dir / "rolling",
    )
    report = simulate_rolling(agent, args.output_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
