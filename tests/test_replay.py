"""Offline end-to-end checks for historical scheduling and February exports."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai.services.archive_download import download_archive, issue_schedule, parse_issue
from ai.services.forecast_agent import ForecastAgent, ForecastPoint, ForecastResult
from ai.services.replay import CSV_COLUMNS, TEST_END, TEST_START, replay
from ai.services.weather_client import (
    ArchivedWeatherForecast,
    HourlyWeather,
    WeatherClientError,
    WeatherMetadata,
    select_safe_run,
)


UTC = timezone.utc
TURBINES = ("turbine_1", "turbine_2")


class ArchivedRunFixture:
    """Model a ten-day Single Runs response, including past hours and null tail."""

    publication_safety_delay = timedelta(hours=6)

    def __init__(self, failures=()):
        self.failures = set(failures)

    def fetch_forecast(self, turbine_id, issue_time):
        if (turbine_id, issue_time) in self.failures:
            raise WeatherClientError("Archived run is not available")
        run = select_safe_run(issue_time)
        rows = tuple(
            HourlyWeather(
                valid_time=run + timedelta(hours=hour),
                wind_speed_80m=5.5 if hour < 90 else None,
                wind_speed_100m=6.0 if hour < 90 else None,
                wind_speed_120m=6.5 if hour < 90 else None,
                wind_direction_100m=230.0 if hour < 90 else None,
                temperature_2m=-2.0 if hour < 90 else None,
                surface_pressure=940.0 if hour < 90 else None,
            )
            for hour in range(240)
        )
        metadata = WeatherMetadata(
            turbine_id, issue_time, run, "Open-Meteo", "ecmwf_ifs",
            datetime(2026, 9, 23, tzinfo=UTC), f"fixture-{turbine_id}-{issue_time.isoformat()}",
        )
        return ArchivedWeatherForecast(metadata, rows, {"fixture": True})


class ReplayPredictor:
    model_id = "fixture-archived-model-v1"
    model_source = "archived_open_meteo_single_runs"
    available_at = "2026-01-31T18:00:00+00:00"
    warnings = ("Fixture prediction; no measured weather is supplied.",)

    def predict(self, features):
        return [-0.2 if index % 3 == 0 else 1.2 if index % 3 == 1 else 0.5
                for index, _ in enumerate(features)]


class ScheduleTests(unittest.TestCase):
    def test_daily_schedule_includes_both_boundaries_without_march_issue(self):
        first = parse_issue("2026-01-31T18:00:00Z")
        last = parse_issue("2026-02-27T18:00:00Z")
        issues = issue_schedule(first, last, 24)
        self.assertEqual(len(issues), 28)
        self.assertEqual(issues[0], first)
        self.assertEqual(issues[-1], last)
        self.assertTrue(all(later - earlier == timedelta(days=1)
                            for earlier, later in zip(issues, issues[1:])))

    def test_schedule_does_not_round_up_past_exclusive_step(self):
        first = parse_issue("2026-01-31T23:00:00Z")
        self.assertEqual(issue_schedule(first, first, 24), [first])
        self.assertEqual(issue_schedule(first, first + timedelta(hours=25), 24),
                         [first, first + timedelta(hours=24)])
        for last, step in ((first - timedelta(hours=1), 24), (first, 0), (first, -1)):
            with self.subTest(last=last, step=step), self.assertRaises(ValueError):
                issue_schedule(first, last, step)

    def test_issue_parser_converts_local_offset_and_rejects_naive_partial_hour(self):
        self.assertEqual(parse_issue("2026-02-01T04:00:00+05:00"),
                         datetime(2026, 1, 31, 23, tzinfo=UTC))
        for text in ("2026-02-01T00:00:00", "2026-02-01T00:15:00Z", "not-a-date"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_issue(text)

    def test_schedule_rejects_naive_fractional_times_and_noninteger_steps(self):
        first = parse_issue("2026-01-31T18:00:00Z")
        for start, end, step in (
            (first.replace(tzinfo=None), first.replace(tzinfo=None), 24),
            (first + timedelta(minutes=15), first + timedelta(minutes=15), 24),
            (first.replace(tzinfo=timezone(timedelta(hours=5, minutes=30))),
             first.replace(tzinfo=timezone(timedelta(hours=5, minutes=30))), 24),
            (first, first, True), (first, first, 1.5), (first, first, "24"),
        ):
            with self.subTest(start=start, step=step), self.assertRaises(ValueError):
                issue_schedule(start, end, step)

    def test_archive_download_counts_failed_turbine_separately(self):
        issues = issue_schedule(parse_issue("2026-02-01T23:00Z"), parse_issue("2026-02-02T23:00Z"), 24)
        report = download_archive(ArchivedRunFixture({("turbine_2", issues[0])}), issues)
        self.assertEqual(report["requests"], 4)
        self.assertEqual(report["complete"], 3)
        failed = [row for row in report["results"] if row["status"] == "FAILED"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["turbine_id"], "turbine_2")
        self.assertIn("not available", failed[0]["error"])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name)

    def make_agent(self, failures=()):
        return ForecastAgent(
            weather_client=ArchivedRunFixture(failures),
            models={turbine: ReplayPredictor() for turbine in TURBINES},
            output_dir=self.output_dir,
        )

    def read_export(self):
        with (self.output_dir / "february_forecasts.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
            return list(reader)

    def test_real_agent_replays_all_672_local_february_hours_for_both_turbines(self):
        self.assertEqual(TEST_START, datetime(2026, 1, 31, 19, tzinfo=UTC))
        self.assertEqual(TEST_END, datetime(2026, 2, 28, 19, tzinfo=UTC))
        issues = issue_schedule(TEST_START - timedelta(hours=1), TEST_END - timedelta(hours=25), 24)
        with contextlib.redirect_stdout(io.StringIO()):
            report = replay(self.make_agent(), issues, self.output_dir)
        self.assertEqual(report["issues"], 28)
        self.assertEqual(report["complete"], 28, report["runs"])
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["expected_hours_per_turbine"], 672)
        self.assertEqual(report["unique_valid_hours"], {turbine: 672 for turbine in TURBINES})
        self.assertEqual(report["forecast_rows"], 2640)
        rows = self.read_export()
        self.assertEqual(len(rows), 2640)
        self.assertEqual(len({(r["turbine_id"], r["issue_time"], r["valid_time"]) for r in rows}), len(rows))
        expected = {TEST_START + timedelta(hours=hour) for hour in range(672)}
        for turbine in TURBINES:
            actual = {datetime.fromisoformat(row["valid_time"]) for row in rows if row["turbine_id"] == turbine}
            self.assertEqual(actual, expected)
        for row in rows:
            issue = datetime.fromisoformat(row["issue_time"])
            run = datetime.fromisoformat(row["weather_run_time"])
            valid = datetime.fromisoformat(row["valid_time"])
            self.assertGreaterEqual(valid, TEST_START)
            self.assertLess(valid, TEST_END)
            self.assertEqual(valid.astimezone(timezone(timedelta(hours=5))).month, 2)
            self.assertLessEqual(run + timedelta(hours=6), issue)
            self.assertGreater(valid, issue)
            self.assertEqual((valid - issue).total_seconds(), int(row["lead_hour"]) * 3600)
            self.assertTrue(1 <= int(row["lead_hour"]) <= 48)
            self.assertTrue(0 <= float(row["predicted_normalized_power"]) <= 1)
            self.assertTrue(row["model_version"])
            self.assertEqual(json.loads(row["warnings"]), list(ReplayPredictor.warnings))
        saved = json.loads((self.output_dir / "replay_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        self.assertIn("no February accuracy metric", report["evaluation"])

    def test_24_hour_horizon_produces_one_forecast_per_turbine_hour(self):
        issues = issue_schedule(TEST_START - timedelta(hours=1), TEST_END - timedelta(hours=25), 24)
        with contextlib.redirect_stdout(io.StringIO()):
            report = replay(self.make_agent(), issues, self.output_dir, horizon_hours=24)
        self.assertEqual(report["complete"], 28)
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["forecast_rows"], 1344)
        self.assertEqual(report["unique_valid_hours"], {turbine: 672 for turbine in TURBINES})

    def test_failed_run_is_reported_and_no_partial_turbine_forecast_is_exported(self):
        first = TEST_START - timedelta(hours=1)
        second = first + timedelta(days=1)
        with contextlib.redirect_stdout(io.StringIO()):
            report = replay(self.make_agent({("turbine_2", first)}), [first, second], self.output_dir)
        self.assertEqual(report["issues"], 2)
        self.assertEqual(report["complete"], 1)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["runs"][0]["status"], "FAILED")
        self.assertIsNone(report["runs"][0]["artifact"])
        rows = self.read_export()
        self.assertEqual(len(rows), 96)
        self.assertEqual({row["issue_time"] for row in rows}, {second.isoformat()})

    def test_failed_result_points_are_never_counted_as_forecasts(self):
        class FailedAgent:
            def run(self, issue, horizon_hours):
                return ForecastResult("FAILED", "partial response rejected",
                                      (ForecastPoint("turbine_1", TEST_START, 0.5),), {})

        with contextlib.redirect_stdout(io.StringIO()):
            report = replay(FailedAgent(), [TEST_START - timedelta(hours=1)], self.output_dir)
        self.assertEqual(report["complete"], 0)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["forecast_rows"], 0)
        self.assertEqual(report["unique_valid_hours"], {turbine: 0 for turbine in TURBINES})
        self.assertEqual(self.read_export(), [])

    def test_explicit_local_calendar_window_filters_at_utc_equivalent_boundaries(self):
        local = timezone(timedelta(hours=5))
        start = datetime(2026, 2, 1, tzinfo=local)
        end = start + timedelta(hours=24)
        # A deliberately earlier-selected fixture avoids using full January
        # validation to forecast the first local February hour.
        model = ReplayPredictor()
        model.available_at = "2025-12-31T19:00:00+00:00"
        agent = ForecastAgent(weather_client=ArchivedRunFixture(),
                              models={turbine: model for turbine in TURBINES}, output_dir=self.output_dir)
        issue = start.astimezone(UTC) - timedelta(hours=1)
        with contextlib.redirect_stdout(io.StringIO()):
            report = replay(agent, [issue], self.output_dir, valid_start=start, valid_end=end)
        self.assertEqual(report["forecast_rows"], 48)
        rows = self.read_export()
        actual = {datetime.fromisoformat(row["valid_time"]) for row in rows}
        self.assertEqual(actual, {start.astimezone(UTC) + timedelta(hours=hour) for hour in range(24)})

    def test_incomplete_period_coverage_fails_even_if_all_issues_complete(self):
        first = TEST_START - timedelta(hours=1)
        with contextlib.redirect_stdout(io.StringIO()):
            report = replay(self.make_agent(), [first], self.output_dir)
        self.assertEqual(report["complete"], report["issues"])
        self.assertEqual(report["unique_valid_hours"], {turbine: 48 for turbine in TURBINES})
        self.assertEqual(report["status"], "INCOMPLETE")

    def test_empty_or_reversed_validation_period_is_rejected(self):
        for end in (TEST_START, TEST_START - timedelta(hours=1)):
            with self.subTest(end=end), self.assertRaises(ValueError):
                replay(self.make_agent(), [TEST_START - timedelta(hours=1)], self.output_dir,
                       valid_start=TEST_START, valid_end=end)


if __name__ == "__main__":
    unittest.main()
