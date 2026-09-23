import unittest
import json
import tempfile
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib

from backend.app.services.model_loader import ArtifactModelLoader, ArtifactPredictor, ModelArtifactError


class RecordingModel:
    def __init__(self):
        self.rows = None

    def predict(self, rows):
        self.rows = rows
        return [0.4 for _ in rows]


class ArtifactPredictorTests(unittest.TestCase):
    def test_legacy_baseline_features_are_adapted_from_forecast_features(self):
        model = RecordingModel()
        predictor = ArtifactPredictor(model, ("mean_wind_speed", "mean_ambient_temperature", "hour_sin"), "mock")
        output = predictor.predict([{
            "wind_speed_80m": 4.0,
            "wind_speed_100m": 5.0,
            "wind_speed_120m": 6.0,
            "temperature_2m": -3.0,
            "hour_sin": 0.5,
        }])
        self.assertEqual(output, [0.4])
        self.assertEqual(model.rows, [[5.0, -3.0, 0.5]])

    def test_nonfinite_features_are_rejected(self):
        predictor = ArtifactPredictor(RecordingModel(), ("wind_speed_100m",), "mock")
        with self.assertRaises(ModelArtifactError):
            predictor.predict([{"wind_speed_100m": float("nan")}])

    def test_scada_calendar_features_use_station_local_time(self):
        model = RecordingModel()
        predictor = ArtifactPredictor(model, ("hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"), "mock", feature_timezone="Asia/Almaty")
        predictor.predict([{"valid_time": datetime(2026, 1, 31, 19, tzinfo=timezone.utc), "hour_sin": -1.0, "hour_cos": -1.0, "day_of_year_sin": -1.0, "day_of_year_cos": -1.0}])
        self.assertAlmostEqual(model.rows[0][0], 0.0)
        self.assertAlmostEqual(model.rows[0][1], 1.0)
        self.assertAlmostEqual(model.rows[0][2], math.sin(2 * math.pi * 32 / 365.25))
        self.assertAlmostEqual(model.rows[0][3], math.cos(2 * math.pi * 32 / 365.25))


class ArtifactModelLoaderTests(unittest.TestCase):
    @staticmethod
    def write_metrics(root, *, artifact="turbine_1_archived.joblib", source="archived_weather_forecast", features=None):
        selected = {
            "artifact": artifact,
            "features": features or ["wind_speed_100m"],
            "selected_model": "extra_trees",
            "model_source": source,
            "available_at": "2026-01-31T19:00:00Z",
        }
        (root / "metrics.json").write_text(json.dumps({"selected_models": {"turbine_1": selected}}), encoding="utf-8")

    def test_selected_archived_artifact_honors_portable_windows_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joblib.dump(RecordingModel(), root / "turbine_1_archived.joblib")
            self.write_metrics(root, artifact=r"C:\different-machine\artifacts\turbine_1_archived.joblib")
            loaded = ArtifactModelLoader(artifacts_dir=root).load("turbine_1")
            self.assertEqual(loaded.predict([{"wind_speed_100m": 5}]), [0.4])
            self.assertEqual(loaded.model_source, "archived_weather_forecast")
            self.assertEqual(loaded.available_at, "2026-01-31T19:00:00+00:00")
            self.assertEqual(loaded.warnings, ())

    def test_baseline_fallback_has_explicit_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joblib.dump(RecordingModel(), root / "turbine_1.joblib")
            self.write_metrics(root, artifact="turbine_1.joblib", source="scada_weather_baseline_fallback", features=["mean_wind_speed"])
            loaded = ArtifactModelLoader(artifacts_dir=root).load("turbine_1")
            self.assertTrue(loaded.warnings)
            self.assertIn("not been validated on forecast-weather", loaded.warnings[0])

    def test_content_change_updates_version_even_when_algorithm_name_is_same(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_metrics(root)
            artifact = root / "turbine_1_archived.joblib"
            joblib.dump(RecordingModel(), artifact)
            loader = ArtifactModelLoader(artifacts_dir=root)
            first = loader.load("turbine_1")
            updated = RecordingModel()
            updated.rows = [[42.0]]
            joblib.dump(updated, artifact)
            second = loader.load("turbine_1")
            self.assertNotEqual(first.model_id, second.model_id)

    def test_metrics_changes_are_not_hidden_by_loader_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joblib.dump(RecordingModel(), root / "turbine_1_archived.joblib")
            self.write_metrics(root)
            loader = ArtifactModelLoader(artifacts_dir=root)
            first = loader.load("turbine_1")
            self.write_metrics(root, features=["temperature_2m"])
            second = loader.load("turbine_1")
            self.assertNotEqual(first.model_id, second.model_id)
            self.assertEqual(second.feature_columns, ("temperature_2m",))

    def test_invalid_artifact_path_and_cross_turbine_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("../turbine_1.joblib", "turbine_2.joblib", "turbine_1.pkl"):
                with self.subTest(name=name):
                    self.write_metrics(root, artifact=name)
                    with self.assertRaises(ModelArtifactError):
                        ArtifactModelLoader(artifacts_dir=root).load("turbine_1")

    def test_saved_feature_contract_must_match_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = RecordingModel()
            model.feature_columns = ("temperature_2m",)
            joblib.dump(model, root / "turbine_1_archived.joblib")
            self.write_metrics(root)
            with self.assertRaisesRegex(ModelArtifactError, "feature columns"):
                ArtifactModelLoader(artifacts_dir=root).load("turbine_1")


if __name__ == "__main__":
    unittest.main()
