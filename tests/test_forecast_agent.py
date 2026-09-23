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

    def __init__(self, *, unsafe=False, incomplete=False, missing_hour=False, extra_hours=False, duplicate=False, fractional=False):
        self.calls = []
        self.unsafe = unsafe
        self.incomplete = incomplete
        self.wind_offset = 0.0
        self.missing_hour = missing_hour
        self.extra_hours = extra_hours
        self.duplicate = duplicate
        self.fractional = fractional
        self.fetch_time = FETCH

    def fetch_forecast(self, turbine_id, issue_time):
        self.calls.append((turbine_id, issue_time))
        run = ISSUE if self.unsafe else RUN
        rows = []
        hours = range(-2, 60) if self.extra_hours else range(1, 49)
        for index in hours:
            if self.missing_hour and index == 10:
                continue
            rows.append(HourlyWeather(
                valid_time=issue_time + timedelta(hours=index, minutes=30 if self.fractional else 0),
                wind_speed_80m=5.0 + self.wind_offset + index,
                wind_speed_100m=6.0 + self.wind_offset + index,
                wind_speed_120m=7.0 + self.wind_offset + index,
                wind_direction_100m=220.0,
                temperature_2m=-2.0,
                surface_pressure=914.0,
            ))
        if self.duplicate:
            rows.append(rows[5])
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
            fetch_time=self.fetch_time,
            input_hash=f"weather-{turbine_id}",
        )
        return ArchivedWeatherForecast(metadata, tuple(rows), {"fixture": True})


class MockPredictor:
    model_id = "mock-v1"

    def __init__(self, values=None):
        self.values = values
        self.calls = []

    def predict(self, features):
        self.calls.append(tuple(features))
        return self.values if self.values is not None else [1.2 if index % 2 else -0.2 for index in range(len(features))]


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
            self.assertEqual(len(result.forecasts), 96)
            self.assertEqual(result.metadata["horizon_hours"], 48)
            for turbine in ("turbine_1", "turbine_2"):
                self.assertEqual(
                    [point.valid_time for point in result.forecasts if point.turbine_id == turbine],
                    [ISSUE + timedelta(hours=hour) for hour in range(1, 49)],
                )
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

    def test_exact_requested_horizon_excludes_past_and_extra_future_hours(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = {"turbine_1": MockPredictor(), "turbine_2": MockPredictor()}
            agent = ForecastAgent(weather_client=MockWeatherClient(extra_hours=True), models=models, output_dir=temporary)
            short = agent.run(ISSUE, horizon_hours=24)
            long = agent.run(ISSUE, horizon_hours=48)
            self.assertEqual(short.status, "COMPLETE", short.message)
            self.assertEqual(long.status, "COMPLETE", long.message)
            self.assertEqual(len(short.forecasts), 48)
            self.assertEqual(len(long.forecasts), 96)
            self.assertNotEqual(short.metadata["input_hash"], long.metadata["input_hash"])
            self.assertTrue(all(ISSUE < point.valid_time <= ISSUE + timedelta(hours=24) for point in short.forecasts))

    def test_missing_duplicate_or_fractional_hour_fails_before_prediction(self):
        for kwargs in ({"missing_hour": True}, {"duplicate": True}, {"fractional": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                predictor = MockPredictor()
                result = ForecastAgent(weather_client=MockWeatherClient(**kwargs), models={"turbine_1": predictor, "turbine_2": predictor}, output_dir=temporary).run(ISSUE)
                self.assertEqual(result.status, "FAILED")
                self.assertEqual(result.states[-2:], ("VALIDATE_INPUT", "FAILED"))
                self.assertEqual(predictor.calls, [])

    def test_invalid_horizon_and_fractional_issue_fail_without_fetch(self):
        weather = MockWeatherClient()
        agent = ForecastAgent(weather_client=weather)
        for horizon in (0, 23, 49, 24.5, True, "48"):
            with self.subTest(horizon=horizon):
                self.assertEqual(agent.run(ISSUE, horizon_hours=horizon).status, "FAILED")
        self.assertEqual(agent.run(ISSUE + timedelta(minutes=15)).status, "FAILED")
        self.assertEqual(weather.calls, [])

    def test_repeated_identical_inputs_reuse_artifact_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = ForecastAgent(weather_client=MockWeatherClient(), models={"turbine_1": MockPredictor(), "turbine_2": MockPredictor()}, output_dir=temporary)
            first = agent.run(ISSUE)
            second = agent.run(ISSUE)
            self.assertEqual(first.to_json(), second.to_json())
            self.assertEqual(first.output_path, second.output_path)
            self.assertEqual(len(list(Path(temporary).glob("forecast_*.json"))), 1)

    def test_retrieval_time_change_reuses_original_artifact_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            weather = MockWeatherClient()
            agent = ForecastAgent(weather_client=weather, models={"turbine_1": MockPredictor(), "turbine_2": MockPredictor()}, output_dir=temporary)
            first = agent.run(ISSUE)
            original = first.output_path.read_bytes()
            weather.fetch_time += timedelta(hours=1)
            second = agent.run(ISSUE)
            self.assertEqual(second.status, "COMPLETE", second.message)
            self.assertEqual(second.to_json(), first.to_json())
            self.assertEqual(second.output_path.read_bytes(), original)

    def test_existing_forecast_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = ForecastAgent(weather_client=MockWeatherClient(), models={"turbine_1": MockPredictor(), "turbine_2": MockPredictor()}, output_dir=temporary)
            first = agent.run(ISSUE)
            path = first.output_path
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["forecast"][0]["prediction"] = 0.12345
            path.write_text(json.dumps(saved), encoding="utf-8")
            second = agent.run(ISSUE)
            self.assertEqual(second.status, "FAILED")
            self.assertIn("Refusing to overwrite", second.message)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)

    def test_nonfinite_or_wrong_length_predictions_fail_without_saved_artifact(self):
        for values in ([float("nan")] * 48, [float("inf")] * 48, [True] * 48, [0.5]):
            with self.subTest(values=values[:1]), tempfile.TemporaryDirectory() as temporary:
                predictor = MockPredictor(values)
                result = ForecastAgent(weather_client=MockWeatherClient(), models={"turbine_1": predictor, "turbine_2": predictor}, output_dir=temporary).run(ISSUE)
                self.assertEqual(result.status, "FAILED")
                self.assertEqual(list(Path(temporary).glob("*.json")), [])

    def test_model_selection_cannot_use_future_validation_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            predictor = MockPredictor()
            predictor.available_at = ISSUE + timedelta(hours=1)
            result = ForecastAgent(weather_client=MockWeatherClient(), models={"turbine_1": predictor, "turbine_2": predictor}, output_dir=temporary).run(ISSUE)
            self.assertEqual(result.status, "FAILED")
            self.assertIn("model selection uses data unavailable", result.message)
            self.assertEqual(predictor.calls, [])

    def test_model_warnings_are_persisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            predictor = MockPredictor()
            predictor.warnings = ("SCADA fallback, accuracy on forecast weather not validated",)
            predictor.model_source = "scada_weather_baseline_fallback"
            result = ForecastAgent(weather_client=MockWeatherClient(), models={"turbine_1": predictor, "turbine_2": predictor}, output_dir=temporary).run(ISSUE)
            self.assertEqual(result.status, "COMPLETE", result.message)
            self.assertEqual(result.metadata["warnings"], list(predictor.warnings))
            self.assertEqual(result.metadata["model_details"]["turbine_1"]["model_source"], predictor.model_source)


if __name__ == "__main__":
    unittest.main()
