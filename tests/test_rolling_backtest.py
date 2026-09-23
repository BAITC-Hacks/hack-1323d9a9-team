"""Offline checks of the official February rolling issue-time simulation."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai.services.forecast_agent import ForecastAgent, ForecastResult
from ai.services.rolling_backtest import (
    BacktestIntegrityError,
    CSV_COLUMNS,
    FEBRUARY_END,
    FEBRUARY_START,
    _validated_rows,
    official_issues,
    simulate_rolling,
)
from ai.services.weather_client import (
    ArchivedWeatherForecast,
    HourlyWeather,
    WeatherClientError,
    WeatherMetadata,
    select_safe_run,
)


UTC = timezone.utc
TURBINES = ("turbine_1", "turbine_2")


class HistoricalWeatherFixture:
    publication_safety_delay = timedelta(hours=6)

    def __init__(self, failed_issues=()):
        self.failed_issues = set(failed_issues)

    def fetch_forecast(self, turbine_id, issue_time):
        if issue_time in self.failed_issues and turbine_id == "turbine_2":
            raise WeatherClientError("Historical forecast unavailable")
        run = select_safe_run(issue_time)
        records = tuple(HourlyWeather(
            valid_time=run + timedelta(hours=hour),
            wind_speed_80m=float(issue_time.day),
            wind_speed_100m=6.0,
            wind_speed_120m=7.0,
            wind_direction_100m=220.0,
            temperature_2m=-2.0,
            surface_pressure=940.0,
        ) for hour in range(240))
        request_hash = hashlib.sha256(f"{turbine_id}:{issue_time.isoformat()}:{run.isoformat()}".encode()).hexdigest()
        content_hash = hashlib.sha256(f"historical fixture:{turbine_id}:{run.isoformat()}".encode()).hexdigest()
        metadata = WeatherMetadata(turbine_id, issue_time, run, "Open-Meteo", "ecmwf_ifs",
                                   datetime(2026, 9, 23, tzinfo=UTC), request_hash,
                                   raw_response_sha256=content_hash)
        return ArchivedWeatherForecast(metadata, records, {"fixture": True})


class HistoricalPredictor:
    model_id = "historical-fixture-v1"
    model_source = "archived_open_meteo_single_runs"
    available_at = "2026-01-30T19:00:00Z"

    def predict(self, features):
        return [row["wind_speed_80m"] / 31 for row in features]


class RollingBacktestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def agent(self, failed_issues=()):
        return ForecastAgent(weather_client=HistoricalWeatherFixture(failed_issues),
                             models={turbine: HistoricalPredictor() for turbine in TURBINES},
                             output_dir=self.root / "rolling")

    def read_csv(self, name):
        with (self.root / name).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
            return list(reader)

    def test_official_schedule_and_full_rolling_evaluation(self):
        issues = official_issues()
        self.assertEqual(len(issues), 29)
        self.assertEqual(issues[0], datetime(2026, 1, 30, 19, tzinfo=UTC))
        self.assertEqual(issues[-1], datetime(2026, 2, 27, 19, tzinfo=UTC))
        report = simulate_rolling(self.agent(), self.root)
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["issue_count"], 29)
        self.assertEqual(report["complete_runs"], 29)
        self.assertEqual(report["forecast_rows"], 2784)
        self.assertEqual(report["february_eligible_rows"], 1398)
        self.assertEqual(report["evaluation_rows"], 1344)
        self.assertEqual(report["selected_hours_per_turbine"], {turbine: 672 for turbine in TURBINES})
        self.assertEqual(len(list((self.root / "rolling").glob("forecast_*.json"))), 29)
        self.assertTrue(all(run["artifact"].startswith("rolling/forecast_") for run in report["runs"]))
        self.assertEqual(json.loads((self.root / "rolling_report.json").read_text()), report)
        all_rows = self.read_csv("rolling_forecasts.csv")
        evaluation = self.read_csv("february_evaluation.csv")
        self.assertEqual(len(all_rows), 2784)
        self.assertEqual(len(evaluation), 1344)
        self.assertEqual(len({(row["turbine_id"], row["valid_time"]) for row in evaluation}), 1344)
        self.assertEqual({row["issue_time"] for row in all_rows}, {issue.isoformat().replace("+00:00", "Z") for issue in issues})
        for row in all_rows:
            issue = datetime.fromisoformat(row["issue_time"].replace("Z", "+00:00"))
            weather_run = datetime.fromisoformat(row["weather_run_time"].replace("Z", "+00:00"))
            valid = datetime.fromisoformat(row["valid_time"].replace("Z", "+00:00"))
            self.assertLessEqual(weather_run + timedelta(hours=6), issue)
            self.assertEqual(valid - issue, timedelta(hours=int(row["lead_hour"])))
            self.assertLessEqual(datetime.fromisoformat(row["model_available_at"].replace("Z", "+00:00")), issue)
            self.assertTrue(0 <= float(row["predicted_normalized_power"]) <= 1)
            self.assertEqual(len(row["weather_input_hash"]), 64)
        for turbine in TURBINES:
            actual = {datetime.fromisoformat(row["valid_time"].replace("Z", "+00:00")) for row in evaluation if row["turbine_id"] == turbine}
            self.assertEqual(actual, {FEBRUARY_START + timedelta(hours=hour) for hour in range(672)})
        self.assertTrue(all(FEBRUARY_START <= datetime.fromisoformat(row["valid_time"].replace("Z", "+00:00")) < FEBRUARY_END for row in evaluation))
        self.assertTrue(all(24 <= int(row["lead_hour"]) <= 48 for row in evaluation))
        february_2 = next(row for row in evaluation if row["turbine_id"] == "turbine_1" and row["valid_time"] == "2026-02-01T19:00:00Z")
        self.assertEqual(february_2["issue_time"], "2026-01-31T19:00:00Z")
        self.assertEqual(february_2["lead_hour"], "24")
        self.assertEqual(february_2["predicted_normalized_power"], str(31 / 31))

    def test_missing_weather_keeps_failed_run_out_of_exports(self):
        first, second = official_issues()[:2]
        report = simulate_rolling(self.agent({first}), self.root, [first, second])
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["complete_runs"], 1)
        self.assertEqual(report["forecast_rows"], 96)
        self.assertEqual(report["runs"][0]["status"], "FAILED")
        self.assertNotIn(first.isoformat().replace("+00:00", "Z"), {row["issue_time"] for row in self.read_csv("rolling_forecasts.csv")})

    def test_rejects_unsafe_weather_and_future_model_even_if_saved_artifact_claims_complete(self):
        issue = official_issues()[0]
        result = self.agent().run(issue)
        self.assertEqual(result.status, "COMPLETE")
        for problem in ("weather", "model", "source", "power"):
            with self.subTest(problem=problem):
                metadata = copy.deepcopy(result.metadata)
                forecasts = result.forecasts
                if problem == "weather":
                    metadata["weather"]["turbine_1"]["weather_run_time"] = issue.isoformat()
                elif problem == "model":
                    metadata["model_details"]["turbine_1"]["available_at"] = (issue + timedelta(hours=1)).isoformat()
                elif problem == "source":
                    metadata["weather"]["turbine_1"]["forecast_source"] = "measured_scada_weather"
                else:
                    forecasts = (replace(forecasts[0], prediction=1.2), *forecasts[1:])
                modified = ForecastResult("COMPLETE", result.message, forecasts, metadata,
                                          self.root / f"unsafe_{problem}.json", result.states)
                modified.output_path.write_text(json.dumps(modified.to_json()), encoding="utf-8")
                with self.assertRaises(BacktestIntegrityError):
                    _validated_rows(modified, issue)

    def test_schedule_rejects_out_of_order_or_non_midnight_issues(self):
        first, second = official_issues()[:2]
        for issues in ([second, first], [first, first], [first + timedelta(hours=1)]):
            with self.subTest(issues=issues), self.assertRaises(ValueError):
                simulate_rolling(self.agent(), self.root, issues)


if __name__ == "__main__":
    unittest.main()
