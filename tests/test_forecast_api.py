import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai.services.forecast_agent import ForecastAgent
from ai.services.weather_client import ArchivedWeatherForecast, HourlyWeather, WeatherMetadata

try:
    from backend.app.main import create_app
    from backend.app.services.forecast_service import ForecastService
    _API_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - dependencies are installed in CI
    create_app = None
    ForecastService = None
    _API_IMPORT_ERROR = exc


UTC = timezone.utc
ISSUE = datetime(2026, 1, 31, 6, 0, tzinfo=UTC)
RUN = datetime(2026, 1, 31, 0, 0, tzinfo=UTC)


class MockWeather:
    publication_safety_delay = timedelta(hours=6)

    def fetch_forecast(self, turbine_id, issue_time):
        rows = tuple(
            HourlyWeather(
                valid_time=issue_time + timedelta(hours=hour),
                wind_speed_80m=5 + hour,
                wind_speed_100m=6 + hour,
                wind_speed_120m=7 + hour,
                wind_direction_100m=200,
                temperature_2m=-2,
                surface_pressure=915,
            )
            for hour in range(-5, 61)
        )
        metadata = WeatherMetadata(
            turbine_id=turbine_id,
            issue_time=issue_time,
            weather_run_time=RUN,
            provider="Open-Meteo",
            model="ecmwf_ifs",
            fetch_time=issue_time,
            input_hash=f"{turbine_id}-input",
        )
        return ArchivedWeatherForecast(metadata, rows, {"fixture": True})


class MockPredictor:
    model_id = "mock-v1"

    def predict(self, features):
        return [0.25 for _ in features]


def build_test_app(output_dir):
    agent = ForecastAgent(
        weather_client=MockWeather(),
        models={"turbine_1": MockPredictor(), "turbine_2": MockPredictor()},
        output_dir=output_dir,
    )
    metrics = Path(output_dir) / "metrics.json"
    metrics.write_text(json.dumps({"selected_models": {}}), encoding="utf-8")
    return create_app(forecast_service=ForecastService(agent=agent, output_dir=output_dir, metrics_path=metrics))


class ForecastApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _API_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"FastAPI dependencies unavailable: {_API_IMPORT_ERROR}")
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # pragma: no cover - dependency is installed in CI
            raise unittest.SkipTest(f"FastAPI test dependencies unavailable: {exc}")
        cls.TestClient = TestClient

    def test_http_request_runs_forecast_and_latest_returns_saved_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.TestClient(build_test_app(Path(temporary)))
            response = client.post("/api/forecast/run", json={"issue_time": "2026-01-31T06:00:00Z", "horizon_hours": 48})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "COMPLETE")
            self.assertEqual(len(body["forecasts"]), 96)
            row = body["forecasts"][0]
            self.assertEqual(
                set(("issue_time", "weather_run_time", "model_version", "turbine", "valid_time", "lead_hour", "predicted_normalized_power", "agent_status", "warnings")),
                set(row),
            )
            self.assertEqual(row["model_version"], "mock-v1")
            self.assertEqual(row["lead_hour"], 1)

            latest = client.get("/api/forecast/latest")
            self.assertEqual(latest.status_code, 200)
            self.assertEqual(latest.json()["status"], "COMPLETE")
            self.assertEqual(latest.json(), body)

    def test_24_hour_run_and_latest_have_identical_complete_horizon(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.TestClient(build_test_app(Path(temporary)))
            response = client.post("/api/forecast/run", json={"issue_time": "2026-01-31T06:00:00Z", "horizon_hours": 24})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["horizon_hours"], 24)
            self.assertEqual(len(body["forecasts"]), 48)
            self.assertEqual({row["lead_hour"] for row in body["forecasts"]}, set(range(1, 25)))
            self.assertEqual(client.get("/api/forecast/latest").json(), body)

    def test_fallback_warnings_reach_run_and_latest_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = build_test_app(Path(temporary))
            predictor = MockPredictor()
            predictor.warnings = ("Measured SCADA fallback, not validated on forecast weather",)
            app.state.forecast_service.agent.model_loader = lambda turbine: predictor
            client = self.TestClient(app)
            response = client.post("/api/forecast/run", json={"issue_time": "2026-01-31T06:00:00Z"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["warnings"], list(predictor.warnings))
            self.assertTrue(all(row["warnings"] == body["warnings"] for row in body["forecasts"]))
            self.assertEqual(client.get("/api/forecast/latest").json(), body)

    def test_health_and_invalid_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.TestClient(build_test_app(Path(temporary)))
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            invalid = client.post("/api/forecast/run", json={"issue_time": "2026-01-31T06:00:00", "horizon_hours": 48})
            self.assertEqual(invalid.status_code, 422)
            for issue, horizon in (("2026-01-31T06:30:00Z", 48), ("2026-01-31T06:00:00Z", 23), ("2026-01-31T06:00:00Z", 49), ("2026-01-31T06:00:00Z", True)):
                with self.subTest(issue=issue, horizon=horizon):
                    invalid = client.post("/api/forecast/run", json={"issue_time": issue, "horizon_hours": horizon})
                    self.assertEqual(invalid.status_code, 422)

    def test_missing_latest_and_unavailable_weather_have_clear_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = build_test_app(Path(temporary))
            client = self.TestClient(app)
            self.assertEqual(client.get("/api/forecast/latest").status_code, 404)
            def fail_weather(*args):
                raise RuntimeError("Archived weather unavailable")
            app.state.forecast_service.agent.weather_client.fetch_forecast = fail_weather
            response = client.post("/api/forecast/run", json={"issue_time": "2026-01-31T06:00:00Z"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["code"], "weather_unavailable")

    def test_latest_rejects_corrupt_out_of_range_or_missing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = self.TestClient(build_test_app(root))
            response = client.post("/api/forecast/run", json={"issue_time": "2026-01-31T06:00:00Z"})
            self.assertEqual(response.status_code, 200)
            path = next(root.glob("forecast_*.json"))
            original = path.read_text(encoding="utf-8")
            for corruption in ("range", "missing", "duplicate", "fractional", "future_run", "unpublished_run", "future_model"):
                with self.subTest(corruption=corruption):
                    payload = json.loads(original)
                    if corruption == "range":
                        payload["forecast"][0]["prediction"] = 1.1
                    elif corruption == "missing":
                        payload["forecast"].pop()
                    elif corruption == "duplicate":
                        payload["forecast"].append(payload["forecast"][0])
                    elif corruption == "fractional":
                        payload["forecast"][0]["valid_time"] = "2026-01-31T07:30:00Z"
                    elif corruption == "future_run":
                        payload["metadata"]["weather"]["turbine_1"]["weather_run_time"] = "2026-02-01T00:00:00Z"
                    elif corruption == "unpublished_run":
                        payload["metadata"]["weather"]["turbine_1"]["weather_run_time"] = "2026-01-31T03:00:00Z"
                    else:
                        payload["metadata"]["model_details"]["turbine_1"]["available_at"] = "2026-02-01T00:00:00Z"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    latest = client.get("/api/forecast/latest")
                    self.assertEqual(latest.status_code, 500)
                    self.assertEqual(latest.json()["detail"]["code"], "agent_failure")

    def test_metrics_endpoint_returns_json_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = build_test_app(root)
            client = self.TestClient(app)
            response = client.get("/api/metrics")
            self.assertEqual(response.status_code, 200)
            self.assertIn("selected_models", response.json())


if __name__ == "__main__":
    unittest.main()
