import unittest

from backend.app.services.model_loader import ArtifactPredictor


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


if __name__ == "__main__":
    unittest.main()
