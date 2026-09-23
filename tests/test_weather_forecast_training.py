import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor

from ai.models.train import WeatherFeatureRegressor, load_archived_weather_rows
from ai.services.weather_client import HOURLY_VARIABLES, MODEL, PROVIDER


UTC = timezone.utc


def write_processed(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp", "normalized_active_power"])
        writer.writeheader()
        writer.writerows([
            {"timestamp": "2025-12-31 00:00:00", "normalized_active_power": "0.4"},
            {"timestamp": "2026-01-01 00:00:00", "normalized_active_power": "0.6"},
        ])


def write_cache(cache_dir: Path, stem: str, issue: str, run: str) -> None:
    cache_dir.mkdir(exist_ok=True)
    valid_times = [
        datetime(2025, 12, 31, tzinfo=UTC).timestamp(),
        datetime(2026, 1, 1, tzinfo=UTC).timestamp(),
    ]
    hourly = {"time": valid_times}
    for index, name in enumerate(HOURLY_VARIABLES):
        hourly[name] = [float(index + 1), float(index + 2)]
    (cache_dir / f"{stem}.json").write_text(json.dumps({"hourly": hourly}), encoding="utf-8")
    (cache_dir / f"{stem}.meta.json").write_text(json.dumps({
        "turbine_id": "turbine_1",
        "issue_time": issue,
        "weather_run_time": run,
        "provider": PROVIDER,
        "model": MODEL,
    }), encoding="utf-8")


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


if __name__ == "__main__":
    unittest.main()
