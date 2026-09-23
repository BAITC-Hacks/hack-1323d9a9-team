import csv
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from ai.models.baseline_forecasting import (
    FEATURE_COLUMNS,
    clip_predictions,
    load_chronological_split,
    train_all,
)


def write_processed(path: Path) -> None:
    fields = ["timestamp", *FEATURE_COLUMNS, "normalized_active_power"]
    start = datetime(2025, 12, 30)
    rows = []
    for offset in range(72):
        timestamp = start + timedelta(hours=offset)
        wind_speed = float(offset % 16)
        rows.append({
            "timestamp": timestamp.isoformat(sep=" "),
            "mean_wind_speed": wind_speed,
            "mean_ambient_temperature": 2.0 + offset % 5,
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_of_year_sin": 0.0,
            "day_of_year_cos": 1.0,
            "normalized_active_power": min(1.0, wind_speed / 15.0),
        })
    rows.append({
        "timestamp": "2026-02-01 00:00:00",
        "mean_wind_speed": 5.0,
        "mean_ambient_temperature": 5.0,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "day_of_year_sin": 0.0,
        "day_of_year_cos": 1.0,
        "normalized_active_power": 0.5,
    })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class BaselineForecastingTests(unittest.TestCase):
    def test_chronological_split_never_trains_on_january_or_february(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            dataset = Path(temporary) / "turbine.csv"
            write_processed(dataset)
            split = load_chronological_split(dataset)
            self.assertEqual(len(split.train_targets), 48)
            self.assertEqual(len(split.validation_targets), 24)

    def test_train_all_saves_loadable_clipped_models_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            processed, artifacts = root / "processed", root / "artifacts"
            processed.mkdir()
            write_processed(processed / "turbine_1_hourly.csv")
            write_processed(processed / "turbine_2_hourly.csv")
            metrics_path = artifacts / "validation_metrics.json"
            report = train_all(processed, artifacts, metrics_path)
            self.assertTrue(metrics_path.exists())
            self.assertEqual(json.loads(metrics_path.read_text(encoding="utf-8")), report)
            for number in (1, 2):
                artifact = artifacts / f"turbine_{number}.joblib"
                self.assertTrue(artifact.exists())
                output = subprocess.check_output(
                    [sys.executable, "-c", "import joblib; print(joblib.load(r'%s').predict([[2, 3, 0, 1, 0, 1]])[0])" % artifact],
                    text=True,
                )
                self.assertTrue(0.0 <= float(output.strip()) <= 1.0)
                selected = report["turbines"][f"turbine_{number}"]["selected_model"]
                self.assertIn("mae", report["turbines"][f"turbine_{number}"]["metrics"][selected])

    def test_clipping_always_enforces_normalized_power_bounds(self) -> None:
        self.assertEqual(clip_predictions([-3.0, 0.25, 4.0]), [0.0, 0.25, 1.0])


if __name__ == "__main__":
    unittest.main()
