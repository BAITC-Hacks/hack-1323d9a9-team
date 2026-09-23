import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai.services.forecast_agent import ForecastAgent
from ai.services.weather_client import ArchivedWeatherForecast, HourlyWeather, WeatherMetadata


UTC = timezone.utc
ISSUE = datetime(2026, 2, 1, 13, 0, tzinfo=UTC)
RUN = datetime(2026, 2, 1, 6, 0, tzinfo=UTC)
FETCH = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)


class MockWeatherClient:
    publication_safety_delay = timedelta(hours=6)

    def __init__(self, *, unsafe=False, incomplete=False):
        self.calls = []
        self.unsafe = unsafe
        self.incomplete = incomplete
        self.wind_offset = 0.0

    def fetch_forecast(self, turbine_id, issue_time):
        self.calls.append((turbine_id, issue_time))
        run = ISSUE if self.unsafe else RUN
        rows = []
        for index in range(2):
            rows.append(HourlyWeather(
                valid_time=datetime(2026, 2, 2, index, tzinfo=UTC),
                wind_speed_80m=5.0 + self.wind_offset + index,
                wind_speed_100m=6.0 + self.wind_offset + index,
                wind_speed_120m=7.0 + self.wind_offset + index,
                wind_direction_100m=220.0,
                temperature_2m=-2.0,
                surface_pressure=914.0,
            ))
        if self.incomplete:
            rows[1] = HourlyWeather(
                valid_time=rows[1].valid_time,
                wind_speed_80m=None,
                wind_speed_100m=rows[1].wind_speed_100m,
                wind_speed_120m=rows[1].wind_speed_120m,
                wind_direction_100m=rows[1].wind_direction_100m,
                temperature_2m=rows[1].temperature_2m,
                surface_pressure=rows[1].surface_pressure,
            )
        metadata = WeatherMetadata(
            turbine_id=turbine_id,
            issue_time=issue_time,
            weather_run_time=run,
            provider="Open-Meteo",
            model="ecmwf_ifs",
            fetch_time=FETCH,
            input_hash=f"weather-{turbine_id}",
        )
        return ArchivedWeatherForecast(metadata, tuple(rows), {"fixture": True})


class MockPredictor:
    model_id = "mock-v1"

    def __init__(self, values=(1.2, -0.2)):
        self.values = values
        self.calls = []

    def predict(self, features):
        self.calls.append(tuple(features))
        return self.values


class ForecastAgentTests(unittest.TestCase):
    def test_run_executes_all_stages_for_both_turbines_and_saves_audit_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            weather = MockWeatherClient()
            models = {"turbine_1": MockPredictor(), "turbine_2": MockPredictor()}
            result = ForecastAgent(weather_client=weather, models=models, output_dir=temporary).run(ISSUE)

            self.assertEqual(result.status, "COMPLETE")
            self.assertEqual(result.states, (
                "FETCH_WEATHER", "VALIDATE_INPUT", "BUILD_FEATURES", "RUN_MODEL",
                "VALIDATE_OUTPUT", "SAVE_RESULT", "COMPLETE",
            ))
            self.assertEqual([call[0] for call in weather.calls], ["turbine_1", "turbine_2"])
            self.assertEqual(len(result.forecasts), 4)
            self.assertEqual({point.prediction for point in result.forecasts}, {0.0, 1.0})
            self.assertTrue(result.output_path and result.output_path.is_file())
            payload = json.loads(result.output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(payload["metadata"]["issue_time"], "2026-02-01T13:00:00Z")
            self.assertEqual(payload["metadata"]["weather"]["turbine_1"]["weather_run_time"], "2026-02-01T06:00:00Z")
            self.assertEqual(payload["metadata"]["state_history"][-1], "COMPLETE")
            self.assertEqual(len(payload["metadata"]["input_hash"]), 64)

    def test_changed_inputs_for_same_issue_time_create_new_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            weather = MockWeatherClient()
            models = {"turbine_1": MockPredictor(), "turbine_2": MockPredictor()}
            first = ForecastAgent(weather_client=weather, models=models, output_dir=temporary).run(ISSUE)
            weather.wind_offset = 0.5
            second = ForecastAgent(weather_client=weather, models=models, output_dir=temporary).run(ISSUE)

            self.assertEqual(first.status, "COMPLETE")
            self.assertEqual(second.status, "COMPLETE")
            self.assertNotEqual(first.metadata["input_hash"], second.metadata["input_hash"])
            self.assertEqual(first.metadata["forecast_version"], 1)
            self.assertEqual(second.metadata["forecast_version"], 2)
            self.assertNotEqual(first.output_path, second.output_path)

    def test_incomplete_weather_fails_explicitly_before_model_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            predictor = MockPredictor()
            result = ForecastAgent(
                weather_client=MockWeatherClient(incomplete=True),
                models={"turbine_1": predictor, "turbine_2": predictor},
                output_dir=temporary,
            ).run(ISSUE)

            self.assertEqual(result.status, "FAILED")
            self.assertIn("incomplete weather field wind_speed_80m", result.message)
            self.assertEqual(result.states[-2:], ("VALIDATE_INPUT", "FAILED"))
            self.assertEqual(predictor.calls, [])
            self.assertEqual(list(Path(temporary).glob("*.json")), [])

    def test_future_weather_run_is_rejected_for_no_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = ForecastAgent(
                weather_client=MockWeatherClient(unsafe=True),
                models={"turbine_1": MockPredictor(), "turbine_2": MockPredictor()},
                output_dir=temporary,
            ).run(ISSUE)

            self.assertEqual(result.status, "FAILED")
            self.assertIn("later than issue time", result.message)
            self.assertEqual(result.states[-2:], ("VALIDATE_INPUT", "FAILED"))

    def test_naive_issue_time_returns_failed_result(self):
        result = ForecastAgent(
            weather_client=MockWeatherClient(),
            models={"turbine_1": MockPredictor(), "turbine_2": MockPredictor()},
        ).run(datetime(2026, 2, 1, 13))
        self.assertEqual(result.status, "FAILED")
        self.assertIn("timezone-aware", result.message)
        self.assertEqual(result.states, ("FAILED",))


if __name__ == "__main__":
    unittest.main()
