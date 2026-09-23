"""Download explicit historical model runs for later audited training/replay."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai.services.weather_client import OpenMeteoArchivedWeatherClient, REPO_ROOT, WeatherClientError


def parse_issue(value: str) -> datetime:
    issue = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if issue.tzinfo is None or issue.utcoffset() is None:
        raise ValueError("Issue time must include a timezone")
    issue = issue.astimezone(timezone.utc)
    if issue.minute or issue.second or issue.microsecond:
        raise ValueError("Issue time must be on an exact UTC hour")
    return issue


def issue_schedule(start: datetime, end: datetime, step_hours: int) -> list[datetime]:
    for issue in (start, end):
        if not isinstance(issue, datetime) or issue.tzinfo is None or issue.utcoffset() is None:
            raise ValueError("Issue times must be timezone-aware")
        utc_issue = issue.astimezone(timezone.utc)
        if utc_issue.minute or utc_issue.second or utc_issue.microsecond:
            raise ValueError("Issue times must be hourly")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    if isinstance(step_hours, bool) or not isinstance(step_hours, int) or start > end or step_hours < 1:
        raise ValueError("Require start <= end and positive step_hours")
    issues = []
    while start <= end:
        issues.append(start)
        start += timedelta(hours=step_hours)
    return issues


def download_archive(client, issues: list[datetime], workers: int = 2) -> dict:
    """Read cache first; record individual failures without inventing weather."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("workers must be an integer between 1 and 4")
    tasks = [(turbine, issue) for issue in issues for turbine in ("turbine_1", "turbine_2")]

    def fetch(task):
        turbine, issue = task
        row = {"turbine_id": turbine, "issue_time": issue.isoformat()}
        try:
            forecast = client.fetch_forecast(turbine, issue)
            row.update(status="COMPLETE", metadata=forecast.metadata.to_json(), hourly_rows=len(forecast.hourly))
        except (WeatherClientError, ValueError) as exc:
            row.update(status="FAILED", error=str(exc))
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fetch, tasks))
    return {"requests": len(rows), "complete": sum(r["status"] == "COMPLETE" for r in rows), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=parse_issue)
    parser.add_argument("--end", required=True, type=parse_issue)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "weather_cache")
    args = parser.parse_args()
    client = OpenMeteoArchivedWeatherClient(cache_dir=args.cache_dir)
    report = download_archive(client, issue_schedule(args.start, args.end, args.step_hours), args.workers)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "download_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Archived forecast requests complete: {report['complete']}/{report['requests']}")
    for row in report["results"]:
        if row["status"] == "FAILED":
            print(f"{row['turbine_id']} {row['issue_time']}: {row['error']}")
    if report["complete"] != report["requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
