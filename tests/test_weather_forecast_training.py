import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor

from unittest.mock import patch
from zoneinfo import ZoneInfo

from ai.models.train import (
    WEATHER_FEATURE_COLUMNS, ForecastTrainingRow, WeatherFeatureRegressor,
    _split_rows, _weather_candidates, load_archived_weather_rows,
    train_all, train_archived_turbine,
)
from ai.services.weather_client import HOURLY_VARIABLES, MODEL, PROVIDER, OpenMeteoArchivedWeatherClient, WeatherMetadata


UTC = timezone.utc


def write_processed(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "normalized_active_power"])
        writer.writeheader()
        writer.writerows([
            {"timestamp": "2025-12-31 05:00:00", "normalized_active_power": "0.4"},
            {"timestamp": "2026-01-01 05:00:00", "normalized_active_power": "0.6"},
        ])


def write_cache(cache_dir: Path, stem: str, issue: str, run: str) -> Path:
    cache_dir.mkdir(exist_ok=True)
    valid_times = [(datetime(2025, 12, 31, tzinfo=UTC) + timedelta(hours=hour)).timestamp() for hour in range(25)]
    hourly = {"time": valid_times}
    for index, name in enumerate(HOURLY_VARIABLES):
        hourly[name] = [float(index + 1)] * len(valid_times)
    body = json.dumps({
        "timezone": "GMT", "utc_offset_seconds": 0,
        "hourly_units": {
            "time": "unixtime", "wind_speed_80m": "m/s", "wind_speed_100m": "m/s", "wind_speed_120m": "m/s",
            "wind_direction_100m": "\u00b0", "temperature_2m": "\u00b0C", "surface_pressure": "hPa",
        },
        "hourly": hourly,
    }).encode("utf-8")
    client = OpenMeteoArchivedWeatherClient(cache_dir=cache_dir)
    issue_time, run_time = datetime.fromisoformat(issue.replace("Z", "+00:00")), datetime.fromisoformat(run.replace("Z", "+00:00"))
    latitude, longitude = client._coordinates("turbine_1")
    key = client._input_hash("turbine_1", issue_time, client._params(latitude, longitude, run_time))
    metadata = WeatherMetadata("turbine_1", issue_time, run_time, PROVIDER, MODEL, datetime(2026, 9, 1, tzinfo=UTC), key, raw_response_sha256=hashlib.sha256(body).hexdigest())
    client._write_cache(key, body, metadata)
    return cache_dir / f"{key}.json"


class WeatherForecastTrainingTests(unittest.TestCase):
    def test_rows_join_only_to_safe_previous_run_weather_at_24_or_48_hours(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed, cache = root / "processed.csv", root / "cache"
            write_processed(processed)
            write_cache(cache, "safe", "2025-12-30T00:00:00Z", "2025-12-29T18:00:00Z")
            # This entry has a run later than its issue time and must be ignored.
            write_cache(cache, "unsafe", "2025-12-30T00:00:00Z", "2025-12-30T06:00:00Z")

            rows, stats = load_archived_weather_rows(processed, cache, "turbine_1")

            self.assertEqual(stats["cache_entries"], 1)
            self.assertEqual([row.target for row in rows[24]], [0.4])
            self.assertEqual([row.target for row in rows[48]], [0.6])
            self.assertTrue(all(row.weather_run_time <= row.issue_time < row.valid_time for values in rows.values() for row in values))

    def test_named_weather_feature_model_clips_predictions(self) -> None:
        columns = ("wind_speed_80m",)
        model = WeatherFeatureRegressor(ExtraTreesRegressor(n_estimators=2, random_state=42), columns)
        model.fit([{"wind_speed_80m": 0.0}, {"wind_speed_80m": 1.0}], [0.0, 1.0])
        prediction = model.predict([{"wind_speed_80m": 1_000_000.0}])
        self.assertEqual(len(prediction), 1)
        self.assertTrue(0.0 <= prediction[0] <= 1.0)

    def test_tampered_weather_cache_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed, cache = root / "processed.csv", root / "cache"
            write_processed(processed)
            raw_path = write_cache(cache, "safe", "2025-12-30T00:00:00Z", "2025-12-29T18:00:00Z")
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            payload["hourly"]["wind_speed_100m"][0] = 999
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            rows, stats = load_archived_weather_rows(processed, cache, "turbine_1")
            self.assertFalse(rows[24])
            self.assertFalse(rows[48])
            self.assertEqual(stats["rejected_cache_entries"], 1)

    def test_local_january_boundary_is_validation_even_in_december_utc(self) -> None:
        local = datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Almaty"))
        valid = local.astimezone(UTC)
        row = ForecastTrainingRow(valid, valid - timedelta(hours=24), valid - timedelta(hours=30), 24, [1.0] * 10, 0.5, local)
        training, validation = _split_rows([row])
        self.assertEqual(valid, datetime(2025, 12, 31, 19, tzinfo=UTC))
        self.assertEqual(training, [])
        self.assertEqual(validation, [row])

    def test_weather_candidate_has_no_random_early_stopping_holdout(self) -> None:
        self.assertFalse(_weather_candidates()["hist_gradient_boosting"].estimator.early_stopping)

    def test_both_leads_train_one_loadable_artifact_on_identical_holdout(self) -> None:
        rows = {24: [], 48: []}
        zone = ZoneInfo("Asia/Almaty")
        for lead in rows:
            for index in range(6):
                local = datetime(2025, 12, 31, 21, tzinfo=zone) + timedelta(hours=index)
                valid = local.astimezone(UTC)
                rows[lead].append(ForecastTrainingRow(valid, valid - timedelta(hours=lead), valid - timedelta(hours=lead + 6), lead, [float(index)] * len(WEATHER_FEATURE_COLUMNS), index / 6, local))
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "turbine_1_archived.joblib"
            result = train_archived_turbine(rows, artifact)
            self.assertEqual(result["lead_time_hours"], [24, 48])
            self.assertEqual(result["training_date_range"]["rows"], 6)
            self.assertEqual(result["validation_period"]["rows"], 6)
            self.assertEqual(result["available_at"], "2026-01-31T18:00:00+00:00")
            audit_path = artifact.parent / result["training_rows_audit"]["file"]
            self.assertEqual(result["training_rows_audit"]["sha256"], hashlib.sha256(audit_path.read_bytes()).hexdigest())
            with audit_path.open(encoding="utf-8", newline="") as stream:
                audit_rows = list(csv.DictReader(stream))
            self.assertEqual(len(audit_rows), 12)
            self.assertTrue(all(row["scada_time"].startswith("2025-") for row in audit_rows if row["phase"] == "training"))
            self.assertTrue(all(datetime.fromisoformat(row["weather_run_time"]) < datetime.fromisoformat(row["issue_time"]) < datetime.fromisoformat(row["valid_time"]) for row in audit_rows))
            self.assertEqual(result["random_seed"], 42)
            self.assertIn("scikit-learn", result["software_versions"])
            self.assertFalse(result["candidate_parameters"]["hist_gradient_boosting"]["early_stopping"])
            for candidate in result["candidate_metrics"].values():
                self.assertEqual(candidate["validation_by_lead"]["24"]["rows"], 3)
                self.assertEqual(candidate["validation_by_lead"]["48"]["rows"], 3)
            output = subprocess.check_output([sys.executable, "-c", "import joblib,sys; print(joblib.load(sys.argv[1]).predict([[0.0]*10])[0])", str(artifact)], text=True)
            self.assertTrue(0 <= float(output.strip()) <= 1)

    def test_unavailable_archive_does_not_overwrite_existing_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            turbines = {}
            for number in (1, 2):
                turbine = f"turbine_{number}"
                write_processed(root / f"{turbine}_hourly.csv")
                (artifacts / f"{turbine}.joblib").write_bytes(b"baseline sentinel")
                turbines[turbine] = {"selected_model": "extra_trees", "metrics": {"extra_trees": {"mae": 0.1, "rmse": 0.1, "r2": 0.5}}}
            (artifacts / "validation_metrics.json").write_text(json.dumps({"turbines": turbines}), encoding="utf-8")
            report = train_all(root, root / "missing_cache", artifacts, artifacts / "metrics.json")
            for turbine in turbines:
                self.assertEqual((artifacts / f"{turbine}.joblib").read_bytes(), b"baseline sentinel")
                self.assertEqual(report["selected_models"][turbine]["model_source"], "scada_weather_baseline_fallback")

    def test_final_january_hour_is_reported_but_cannot_change_model_selection(self) -> None:
        zone = ZoneInfo("Asia/Almaty")
        rows = {24: [], 48: []}
        local_times = [datetime(2025, 12, 30, hour, tzinfo=zone) for hour in (0, 1)] + [datetime(2026, 1, 30, tzinfo=zone), datetime(2026, 1, 31, 23, tzinfo=zone)]
        for lead in rows:
            for index, local in enumerate(local_times):
                valid = local.astimezone(UTC)
                rows[lead].append(ForecastTrainingRow(valid, valid - timedelta(hours=lead), valid - timedelta(hours=lead + 6), lead, [float(index)] * 10, float(index == 3), local))
        candidates = {
            "a_high": WeatherFeatureRegressor(DummyRegressor(strategy="constant", constant=1.0)),
            "z_low": WeatherFeatureRegressor(DummyRegressor(strategy="constant", constant=0.0)),
        }
        with tempfile.TemporaryDirectory() as temporary, patch("ai.models.train._weather_candidates", return_value=candidates):
            result = train_archived_turbine(rows, Path(temporary) / "archived.joblib")
        self.assertEqual(result["selected_model"], "z_low")
        self.assertEqual(result["metrics"]["mae"], 0.5)
        self.assertEqual(result["selection_metrics"]["mae"], 0.0)
        self.assertEqual(result["selection_validation_range"]["rows"], 2)
        self.assertEqual(result["validation_period"]["rows"], 4)


if __name__ == "__main__":
    unittest.main()
