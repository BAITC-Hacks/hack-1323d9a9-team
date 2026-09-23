"""Reproduce daily historical forecasts and export the February test period."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ai.services.archive_download import issue_schedule, parse_issue
from ai.services.forecast_agent import ForecastAgent
from ai.services.weather_client import OpenMeteoArchivedWeatherClient, REPO_ROOT
from backend.app.services.model_loader import ArtifactModelLoader


UTC = timezone.utc
STATION_TIMEZONE = ZoneInfo("Asia/Almaty")
TEST_START = datetime(2026, 2, 1, tzinfo=STATION_TIMEZONE).astimezone(UTC)
TEST_END = datetime(2026, 3, 1, tzinfo=STATION_TIMEZONE).astimezone(UTC)
CSV_COLUMNS = (
    "turbine_id", "issue_time", "weather_run_time", "valid_time", "lead_hour",
    "predicted_normalized_power", "model_version", "warnings",
)


def replay(agent, issues: list[datetime], output_dir: Path, horizon_hours: int = 48,
           valid_start: datetime = TEST_START, valid_end: datetime = TEST_END) -> dict:
    """Keep each issue/horizon distinct; failures are never counted as forecasts."""
    for boundary in (valid_start, valid_end):
        if boundary.tzinfo is None or boundary.utcoffset() is None:
            raise ValueError("Test period boundaries must be timezone-aware exact hours")
        utc_boundary = boundary.astimezone(UTC)
        if utc_boundary.minute or utc_boundary.second or utc_boundary.microsecond:
            raise ValueError("Test period boundaries must be timezone-aware exact UTC hours")
    valid_start, valid_end = valid_start.astimezone(UTC), valid_end.astimezone(UTC)
    if valid_start >= valid_end:
        raise ValueError("Require valid_start < valid_end")
    if len(set(issues)) != len(issues):
        raise ValueError("Duplicate issue times are not allowed")
    for issue in issues:
        issue_schedule(issue, issue, 24)
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or not 24 <= horizon_hours <= 48:
        raise ValueError("horizon_hours must be an integer between 24 and 48")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, rows = [], []
    for issue in issues:
        result = agent.run(issue, horizon_hours=horizon_hours)
        runs.append({"issue_time": issue.isoformat(), "status": result.status,
                     "message": result.message, "artifact": str(result.output_path) if result.output_path else None})
        print(f"{issue.isoformat()} {result.status}", flush=True)
        if result.status != "COMPLETE":
            continue
        metadata = result.metadata
        for point in result.forecasts:
            if not valid_start <= point.valid_time < valid_end:
                continue
            rows.append({
                "turbine_id": point.turbine_id,
                "issue_time": issue.isoformat(),
                "weather_run_time": metadata["weather"][point.turbine_id]["weather_run_time"],
                "valid_time": point.valid_time.isoformat(),
                "lead_hour": int((point.valid_time - issue).total_seconds() / 3600),
                "predicted_normalized_power": point.prediction,
                "model_version": metadata["models"][point.turbine_id],
                "warnings": json.dumps(metadata.get("warnings", []), ensure_ascii=False),
            })
    with (output_dir / "february_forecasts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    expected_hours = int((valid_end - valid_start).total_seconds() / 3600)
    unique_hours = {turbine: len({row["valid_time"] for row in rows if row["turbine_id"] == turbine})
                    for turbine in ("turbine_1", "turbine_2")}
    complete_runs = sum(run["status"] == "COMPLETE" for run in runs)
    report = {
        "status": "COMPLETE" if issues and complete_runs == len(issues) and all(count == expected_hours for count in unique_hours.values()) else "INCOMPLETE",
        "calendar_timezone": str(STATION_TIMEZONE),
        "period_start": valid_start.isoformat(), "period_end_exclusive": valid_end.isoformat(),
        "horizon_hours": horizon_hours, "issues": len(issues),
        "complete": complete_runs,
        "forecast_rows": len(rows),
        "expected_hours_per_turbine": expected_hours,
        "unique_valid_hours": unique_hours,
        "evaluation": "February actual power is not supplied; no February accuracy metric is claimed.",
        "runs": runs,
    }
    (output_dir / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_issue, default=TEST_START - timedelta(hours=1))
    parser.add_argument("--end", type=parse_issue, default=TEST_END - timedelta(hours=25))
    parser.add_argument("--step-hours", type=int, default=24)
    parser.add_argument("--horizon-hours", type=int, choices=(24, 48), default=48)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "forecasts" / "february")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "weather_cache")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    agent = ForecastAgent(
        weather_client=OpenMeteoArchivedWeatherClient(cache_dir=args.cache_dir, cache_only=args.cache_only),
        model_loader=ArtifactModelLoader(), output_dir=args.output_dir,
    )
    report = replay(agent, issue_schedule(args.start, args.end, args.step_hours), args.output_dir, args.horizon_hours)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
