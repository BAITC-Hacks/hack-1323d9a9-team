"""Leakage-free hourly preprocessing for raw turbine SCADA exports.

The module deliberately reads raw files without modifying them.  It neither
interpolates nor forward-fills SCADA values: an output hour is emitted only
when that hour contains at least ``min_samples`` source records.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Iterable


TIMESTAMP_COLUMN = "\u0421\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u0432\u0440\u0435\u043c\u044f"
OUTPUT_COLUMNS = (
    "timestamp",
    "mean_wind_speed",
    "mean_ambient_temperature",
    "normalized_active_power",
    "sample_count",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)


def _normalise_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold().replace("\u0451", "\u0435"))


def _find_column(headers: Iterable[str], exact: tuple[str, ...], required_words: tuple[str, ...]) -> str:
    normalised = {_normalise_header(header): header for header in headers}
    for name in exact:
        if name in normalised:
            return normalised[name]
    for normalised_name, original in normalised.items():
        if required_words and all(word in normalised_name for word in required_words):
            return original
    raise ValueError(
        "Could not identify a required SCADA column. Available columns: "
        + ", ".join(headers)
    )


def _resolve_columns(headers: list[str]) -> dict[str, str]:
    timestamp = _find_column(headers, (_normalise_header(TIMESTAMP_COLUMN), "statistical time", "timestamp"), ())
    wind = _find_column(headers, ("wind speed", "\u0441\u043a\u043e\u0440\u043e\u0441\u0442\u044c \u0432\u0435\u0442\u0440\u0430"), ("\u0441\u043a\u043e\u0440\u043e\u0441\u0442\u044c", "\u0432\u0435\u0442\u0440\u0430"))
    temperature = _find_column(
        headers,
        ("ambient temperature", "\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430 \u043e\u043a\u0440\u0443\u0436\u0430\u044e\u0449\u0435\u0439 \u0441\u0440\u0435\u0434\u044b"),
        ("\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430",),
    )
    power = _find_column(
        headers,
        ("normalized active power", "active power", "\u043d\u043e\u0440\u043c\u0430\u043b\u0438\u0437\u043e\u0432\u0430\u043d\u043d\u0430\u044f \u0430\u043a\u0442\u0438\u0432\u043d\u0430\u044f \u043c\u043e\u0449\u043d\u043e\u0441\u0442\u044c"),
        ("\u0430\u043a\u0442\u0438\u0432", "\u043c\u043e\u0449\u043d\u043e\u0441\u0442"),
    )
    return {"timestamp": timestamp, "wind": wind, "temperature": temperature, "power": power}


def _parse_datetime(value: str) -> datetime:
    cleaned = value.strip().replace("T", " ")
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp {value!r}")


def _parse_number(value: str) -> float:
    cleaned = value.strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
    numeric = float(cleaned)
    if not math.isfinite(numeric):
        raise ValueError("SCADA measurements must be finite")
    return numeric


def _hour_start(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _cyclical_features(timestamp: datetime) -> tuple[float, float, float, float]:
    hour_angle = 2 * math.pi * timestamp.hour / 24
    # 365.25 keeps the encoding continuous across leap years.
    day_angle = 2 * math.pi * timestamp.timetuple().tm_yday / 365.25
    return math.sin(hour_angle), math.cos(hour_angle), math.sin(day_angle), math.cos(day_angle)


@dataclass(frozen=True)
class PipelineStats:
    raw_rows: int
    parsed_rows: int
    output_rows: int
    start: datetime | None
    end: datetime | None
    duplicate_timestamps: int
    gaps_over_10_minutes: int
    estimated_missing_10_minute_timestamps: int
    target_min: float | None
    target_max: float | None
    correlations: dict[str, float | None]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return numerator / denominator if denominator else None


def process_turbine(input_path: Path, output_path: Path, min_samples: int = 3) -> PipelineStats:
    """Aggregate one raw SCADA CSV to valid hourly targets and write CSV."""
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"{input_path} has no header row")
        columns = _resolve_columns(reader.fieldnames)
        raw_rows, parsed_rows = 0, 0
        records: list[tuple[datetime, float, float, float]] = []
        for row in reader:
            raw_rows += 1
            try:
                record = (
                    _parse_datetime(row[columns["timestamp"]]),
                    _parse_number(row[columns["wind"]]),
                    _parse_number(row[columns["temperature"]]),
                    _parse_number(row[columns["power"]]),
                )
                if record[0].replace(tzinfo=None) >= datetime(2026, 2, 1):
                    continue
                records.append(record)
                parsed_rows += 1
            except (KeyError, TypeError, ValueError):
                # Invalid rows cannot make a reliable hourly target; they are not imputed.
                continue

    if len({record[0].tzinfo is None for record in records}) > 1:
        raise ValueError("Mixed timezone-aware and naive SCADA timestamps are ambiguous")
    records.sort(key=lambda record: record[0])
    timestamps = [record[0] for record in records]
    duplicate_timestamps = sum(current == previous for previous, current in zip(timestamps, timestamps[1:]))
    gaps = [current - previous for previous, current in zip(timestamps, timestamps[1:]) if current > previous]
    long_gaps = [gap for gap in gaps if gap > timedelta(minutes=10)]
    missing = sum(max(0, round(gap.total_seconds() / 600) - 1) for gap in long_gaps)

    buckets: dict[datetime, list[tuple[float, float, float]]] = defaultdict(list)
    unique_samples: dict[datetime, set[tuple[float, float, float]]] = defaultdict(set)
    for timestamp, wind, temperature, power in records:
        unique_samples[timestamp].add((wind, temperature, power))
    for timestamp, samples in unique_samples.items():
        # A duplicated record is one measurement. Conflicting duplicates cannot
        # supply a reliable target and must not inflate hourly coverage.
        if len(samples) == 1:
            buckets[_hour_start(timestamp)].append(next(iter(samples)))

    output_rows: list[dict[str, float | int | str]] = []
    for timestamp in sorted(buckets):
        values = buckets[timestamp]
        if len(values) < min_samples:
            continue
        wind, temperature, power = zip(*values)
        hour_sin, hour_cos, day_sin, day_cos = _cyclical_features(timestamp)
        output_rows.append({
            "timestamp": timestamp.isoformat(sep=" "),
            "mean_wind_speed": fmean(wind),
            "mean_ambient_temperature": fmean(temperature),
            "normalized_active_power": min(1.0, max(0.0, fmean(power))),
            "sample_count": len(values),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_year_sin": day_sin,
            "day_of_year_cos": day_cos,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    powers = [float(row["normalized_active_power"]) for row in output_rows]
    winds = [float(row["mean_wind_speed"]) for row in output_rows]
    temperatures = [float(row["mean_ambient_temperature"]) for row in output_rows]
    return PipelineStats(raw_rows, parsed_rows, len(output_rows), timestamps[0] if timestamps else None,
                         timestamps[-1] if timestamps else None, duplicate_timestamps, len(long_gaps), missing,
                         min(powers) if powers else None, max(powers) if powers else None,
                         {"wind_speed_vs_power": _pearson(winds, powers),
                          "ambient_temperature_vs_power": _pearson(temperatures, powers)})


def _format_value(value: object) -> str:
    return "n/a" if value is None else str(value)


def write_analysis(report_path: Path, statistics: dict[str, PipelineStats]) -> None:
    lines = ["# SCADA data analysis", "", "Generated by `python -m ai.models.scada_preprocessing`. Raw files are read only; no gap interpolation or forward fill is performed.", ""]
    for turbine, stats in statistics.items():
        lines.extend([
            f"## {turbine}", "",
            f"- Row counts: {stats.raw_rows} raw, {stats.parsed_rows} valid source, {stats.output_rows} hourly targets.",
            f"- Date range (valid source timestamps): {_format_value(stats.start)} to {_format_value(stats.end)}.",
            f"- Missing timestamp statistics: {stats.duplicate_timestamps} duplicate timestamps; {stats.gaps_over_10_minutes} gaps over 10 minutes; approximately {stats.estimated_missing_10_minute_timestamps} missing 10-minute timestamps.",
            f"- Target range: {_format_value(stats.target_min)} to {_format_value(stats.target_max)}.",
            f"- Basic correlations (hourly Pearson): wind speed vs power = {_format_value(stats.correlations['wind_speed_vs_power'])}; ambient temperature vs power = {_format_value(stats.correlations['ambient_temperature_vs_power'])}.", "",
        ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _raw_turbine_path(raw_dir: Path, number: int) -> Path:
    """Accept the documented name and the existing `tribune` source spelling."""
    documented = raw_dir / f"turbine_{number}.csv"
    existing_spelling = raw_dir / f"tribune_{number}.csv"
    if documented.exists():
        return documented
    if existing_spelling.exists():
        return existing_spelling
    raise FileNotFoundError(
        f"Expected {documented.name} or {existing_spelling.name} in {raw_dir}"
    )


def run_pipeline(raw_dir: Path, processed_dir: Path, report_path: Path, min_samples: int = 3) -> dict[str, PipelineStats]:
    statistics = {
        f"Turbine {number}": process_turbine(
            _raw_turbine_path(raw_dir, number),
            processed_dir / f"turbine_{number}_hourly.csv",
            min_samples,
        )
        for number in (1, 2)
    }
    write_analysis(report_path, statistics)
    return statistics


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create hourly SCADA datasets for both turbines.")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--report", type=Path, default=root / "docs" / "data-analysis.md")
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.processed_dir, args.report, args.min_samples)


if __name__ == "__main__":
    main()
