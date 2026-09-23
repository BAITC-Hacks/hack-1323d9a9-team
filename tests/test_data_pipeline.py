import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from ai.models.scada_preprocessing import process_turbine, run_pipeline


HEADERS = ["Статистическое время", "Скорость ветра", "Температура окружающей среды", "Нормализованная активная мощность"]


def write_raw(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(HEADERS)
        writer.writerows(rows)


class ScadaPipelineTests(unittest.TestCase):
    def test_single_digit_hours_in_raw_export_are_not_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw.csv", root / "hourly.csv"
            write_raw(raw, [[f"2025-12-31 {hour}:{minute:02d}:00", "3", "2", "0.3"] for hour in range(24) for minute in (0, 10, 20)])
            stats = process_turbine(raw, output)
            self.assertEqual(stats.parsed_rows, 72)
            self.assertEqual(stats.output_rows, 24)

    def test_duplicate_and_nonfinite_samples_do_not_inflate_hourly_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw.csv", root / "hourly.csv"
            write_raw(raw, [
                ["2025-12-31 0:00:00", "3", "2", "0.3"],
                ["2025-12-31 0:00:00", "3", "2", "0.3"],
                ["2025-12-31 0:10:00", "3", "2", "0.3"],
                ["2025-12-31 0:20:00", "nan", "2", "0.3"],
            ])
            stats = process_turbine(raw, output)
            self.assertEqual(stats.duplicate_timestamps, 1)
            self.assertEqual(stats.output_rows, 0)

    def test_february_targets_are_excluded_from_historical_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw.csv", root / "hourly.csv"
            write_raw(raw, [[f"2026-02-01 0:{minute:02d}:00", "3", "2", "0.3"] for minute in (0, 10, 20)])
            self.assertEqual(process_turbine(raw, output).output_rows, 0)

    def test_hourly_aggregation_filters_sparse_hours_and_does_not_change_raw(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            raw = root / "turbine_1.csv"
            output = root / "processed" / "turbine_1_hourly.csv"
            write_raw(raw, [
                ["31.01.2026 00:20", "6,0", "-2,0", "0,9"],
                ["31.01.2026 00:00", "4,0", "-4,0", "0,2"],
                ["31.01.2026 00:10", "5,0", "-3,0", "0,4"],
                ["31.01.2026 01:00", "9,0", "1,0", "1,2"],
                ["31.01.2026 01:10", "9,0", "1,0", "1,2"],
                ["31.01.2026 02:00", "1,0", "0,0", "0,1"],
            ])
            raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
            stats = process_turbine(raw, output)
            self.assertEqual(raw_hash, hashlib.sha256(raw.read_bytes()).hexdigest())
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["timestamp"], "2026-01-31 00:00:00")
            self.assertEqual(rows[0]["sample_count"], "3")
            self.assertAlmostEqual(float(rows[0]["mean_wind_speed"]), 5.0)
            self.assertAlmostEqual(float(rows[0]["normalized_active_power"]), 0.5)
            self.assertEqual(stats.gaps_over_10_minutes, 2)

    def test_both_turbines_are_processed_by_one_command_function(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            raw_dir, processed_dir, report = root / "raw", root / "processed", root / "data-analysis.md"
            raw_dir.mkdir()
            rows = [[f"01.02.2026 00:{minute:02d}", "3", "2", "0.3"] for minute in (0, 10, 20)]
            write_raw(raw_dir / "turbine_1.csv", rows)
            write_raw(raw_dir / "turbine_2.csv", rows)
            run_pipeline(raw_dir, processed_dir, report)
            self.assertTrue((processed_dir / "turbine_1_hourly.csv").exists())
            self.assertTrue((processed_dir / "turbine_2_hourly.csv").exists())
            self.assertIn("Row counts", report.read_text(encoding="utf-8"))

    def test_existing_tribune_filename_spelling_is_supported(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            rows = [[f"01.02.2026 00:{minute:02d}", "3", "2", "0.3"] for minute in (0, 10, 20)]
            write_raw(raw_dir / "tribune_1.csv", rows)
            write_raw(raw_dir / "tribune_2.csv", rows)
            run_pipeline(raw_dir, root / "processed", root / "report.md")
            self.assertTrue((root / "processed" / "turbine_1_hourly.csv").exists())

    def test_hourly_timestamps_are_unique_chronological_and_power_is_clipped(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            raw, output = root / "raw.csv", root / "out.csv"
            rows = []
            for hour, power in ((1, "-0.2"), (0, "1.4")):
                rows.extend([[f"01.01.2026 {hour:02d}:{minute:02d}", "1", "2", power] for minute in (0, 10, 20)])
            write_raw(raw, rows)
            process_turbine(raw, output)
            with output.open(newline="", encoding="utf-8") as stream:
                result = list(csv.DictReader(stream))
            timestamps = [row["timestamp"] for row in result]
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertEqual(len(timestamps), len(set(timestamps)))
            self.assertTrue(all(0.0 <= float(row["normalized_active_power"]) <= 1.0 for row in result))


if __name__ == "__main__":
    unittest.main()
