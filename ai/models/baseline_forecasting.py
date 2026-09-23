"""Train deterministic SCADA-weather baseline power forecasts.

This module is deliberately a *historical measured-weather* baseline.  It
uses the hourly wind speed and temperature measured by SCADA, not weather
forecasts.  Production inference must instead provide archived forecast
weather that was available at the forecast issue time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if __name__ == "__main__":
    # Keep joblib's canonical import path and the module-run class identical.
    sys.modules["ai.models.baseline_forecasting"] = sys.modules[__name__]


FEATURE_COLUMNS = (
    "mean_wind_speed",
    "mean_ambient_temperature",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
TARGET_COLUMN = "normalized_active_power"
TRAIN_END_EXCLUSIVE = datetime(2026, 1, 1)
VALIDATION_START = datetime(2026, 1, 1)
VALIDATION_END_EXCLUSIVE = datetime(2026, 2, 1)
RANDOM_SEED = 42
BASELINE_DESCRIPTION = (
    "Historical measured-SCADA-weather baseline only; production inference "
    "must use archived weather forecasts available at issue time."
)


@dataclass(frozen=True)
class DatasetSplit:
    """Chronological, usable feature/target rows for one turbine."""

    train_features: list[list[float]]
    train_targets: list[float]
    validation_features: list[list[float]]
    validation_targets: list[float]
    usable_rows: int
    dropped_rows: int
    train_timestamps: list[datetime]
    validation_timestamps: list[datetime]


class BinnedWindPowerCurve:
    """Simple deterministic wind-speed-to-mean-power comparison baseline."""

    def __init__(self, bin_width: float = 0.5) -> None:
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")
        self.bin_width = bin_width

    def fit(self, features: list[list[float]], targets: list[float]) -> "BinnedWindPowerCurve":
        if not features or len(features) != len(targets):
            raise ValueError("features and targets must be non-empty and equally sized")
        bins: dict[int, list[float]] = {}
        for row, target in zip(features, targets):
            key = self._bin(row[0])
            bins.setdefault(key, []).append(target)
        self.bin_means_ = {key: sum(values) / len(values) for key, values in bins.items()}
        self.global_mean_ = sum(targets) / len(targets)
        return self

    def predict(self, features: Iterable[list[float]]) -> list[float]:
        if not hasattr(self, "bin_means_"):
            raise ValueError("The baseline must be fitted before prediction")
        return clip_predictions(self._prediction_for_bin(self._bin(row[0])) for row in features)

    def _bin(self, wind_speed: float) -> int:
        return math.floor(wind_speed / self.bin_width)

    def _prediction_for_bin(self, key: int) -> float:
        if key in self.bin_means_:
            return self.bin_means_[key]
        nearest = min(self.bin_means_, key=lambda existing: (abs(existing - key), existing))
        return self.bin_means_.get(nearest, self.global_mean_)


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Make the normalized-power output constraint part of the saved model."""

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator

    def fit(self, features: list[list[float]], targets: list[float]) -> "ClippedRegressor":
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(features, targets)
        return self

    def predict(self, features: Iterable[list[float]]) -> list[float]:
        return clip_predictions(self.estimator_.predict(features))


# `python -m ai.models.baseline_forecasting` executes this file as __main__.
# A stable module path keeps its joblib artifacts loadable in a fresh process.
ClippedRegressor.__module__ = "ai.models.baseline_forecasting"


def _parse_usable_row(row: dict[str, str]) -> tuple[datetime, list[float], float] | None:
    try:
        timestamp = datetime.fromisoformat(row["timestamp"])
        features = [float(row[column]) for column in FEATURE_COLUMNS]
        target = float(row[TARGET_COLUMN])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*features, target)):
        return None
    if not 0 <= target <= 1:
        return None
    return timestamp, features, target


def load_chronological_split(path: Path) -> DatasetSplit:
    """Load finite rows and partition them without random sampling or leakage."""
    train_features: list[list[float]] = []
    train_targets: list[float] = []
    validation_features: list[list[float]] = []
    validation_targets: list[float] = []
    usable_rows = dropped_rows = 0
    train_timestamps: list[datetime] = []
    validation_timestamps: list[datetime] = []
    seen: dict[datetime, tuple[list[float], float]] = {}

    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parsed = _parse_usable_row(row)
            if parsed is None:
                dropped_rows += 1
                continue
            timestamp, features, target = parsed
            # Calendar boundaries belong to the source SCADA clock. Do not
            # silently reinterpret a timezone-naive export as UTC.
            calendar_time = timestamp.replace(tzinfo=None)
            if calendar_time in seen:
                if seen[calendar_time] != (features, target):
                    raise ValueError(f"Conflicting duplicate SCADA timestamp: {timestamp}")
                dropped_rows += 1
                continue
            seen[calendar_time] = (features, target)
            if calendar_time >= VALIDATION_END_EXCLUSIVE:
                dropped_rows += 1
                continue
            usable_rows += 1
            if calendar_time < TRAIN_END_EXCLUSIVE:
                train_features.append(features)
                train_targets.append(target)
                train_timestamps.append(timestamp)
            elif VALIDATION_START <= calendar_time < VALIDATION_END_EXCLUSIVE:
                validation_features.append(features)
                validation_targets.append(target)
                validation_timestamps.append(timestamp)

    if not train_features:
        raise ValueError(f"No usable training rows before 2026-01-01 in {path}")
    if not validation_features:
        raise ValueError(f"No usable January 2026 validation rows in {path}")
    return DatasetSplit(
        train_features, train_targets, validation_features, validation_targets, usable_rows, dropped_rows,
        train_timestamps, validation_timestamps,
    )


def clip_predictions(predictions: Iterable[float]) -> list[float]:
    """Enforce the normalized power output range at every evaluation/inference use."""
    result = []
    for value in predictions:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Power predictions must be finite before clipping")
        result.append(min(1.0, max(0.0, numeric)))
    return result


def validation_metrics(targets: list[float], predictions: Iterable[float]) -> dict[str, float]:
    clipped = clip_predictions(predictions)
    return {
        "mae": float(mean_absolute_error(targets, clipped)),
        "rmse": float(mean_squared_error(targets, clipped) ** 0.5),
        "r2": float(r2_score(targets, clipped)),
    }


def _candidate_models() -> dict[str, object]:
    return {
        "extra_trees": ClippedRegressor(
            ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
        ),
        "hist_gradient_boosting": ClippedRegressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                l2_regularization=0.1,
                random_state=RANDOM_SEED,
                # sklearn's automatic early stopping creates a random holdout.
                early_stopping=False,
            )
        ),
    }


def train_turbine(input_path: Path, artifact_path: Path) -> dict[str, object]:
    """Train both candidate regressors, select by validation MAE, and save it."""
    split = load_chronological_split(input_path)
    models = _candidate_models()
    model_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(split.train_features, split.train_targets)
        model_metrics[name] = validation_metrics(
            split.validation_targets, model.predict(split.validation_features)
        )

    wind_curve = BinnedWindPowerCurve().fit(split.train_features, split.train_targets)
    model_metrics["binned_wind_power_curve"] = validation_metrics(
        split.validation_targets, wind_curve.predict(split.validation_features)
    )
    selected_name = min(("extra_trees", "hist_gradient_boosting"), key=lambda name: model_metrics[name]["mae"])
    selected_model = models[selected_name]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    # Compression keeps versioned baseline artifacts below Git hosting size limits.
    joblib.dump(selected_model, artifact_path, compress=3)

    return {
        "input": str(input_path),
        "artifact": str(artifact_path),
        "data_source": "measured_historical_scada_weather",
        "important_limitation": BASELINE_DESCRIPTION,
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "split": {
            "training_start": min(split.train_timestamps).isoformat(sep=" "),
            "training_end": max(split.train_timestamps).isoformat(sep=" "),
            "training_end_exclusive": TRAIN_END_EXCLUSIVE.isoformat(sep=" "),
            "validation_start": VALIDATION_START.isoformat(sep=" "),
            "validation_end_exclusive": VALIDATION_END_EXCLUSIVE.isoformat(sep=" "),
            "training_rows": len(split.train_targets),
            "validation_rows": len(split.validation_targets),
            "usable_rows": split.usable_rows,
            "dropped_rows": split.dropped_rows,
        },
        "random_seed": RANDOM_SEED,
        "metrics": model_metrics,
        "selected_model": selected_name,
        "selection_basis": "lowest clipped January 2026 validation MAE among sklearn regressors",
        "selection_requires_targets_before": VALIDATION_END_EXCLUSIVE.isoformat(sep=" "),
        "timestamp_convention": "Source SCADA calendar; timezone is not assumed to be UTC.",
    }


def train_all(processed_dir: Path, artifacts_dir: Path, metrics_path: Path) -> dict[str, object]:
    """Train and evaluate the strictly chronological baseline for both turbines."""
    results = {
        f"turbine_{number}": train_turbine(
            processed_dir / f"turbine_{number}_hourly.csv",
            artifacts_dir / f"turbine_{number}.joblib",
        )
        for number in (1, 2)
    }
    report = {
        "baseline_type": "historical_measured_scada_weather",
        "important_limitation": BASELINE_DESCRIPTION,
        "turbines": results,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train historical measured-SCADA-weather baseline models.")
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "ai" / "models" / "artifacts")
    parser.add_argument("--metrics", type=Path, default=root / "ai" / "models" / "artifacts" / "validation_metrics.json")
    args = parser.parse_args()
    report = train_all(args.processed_dir, args.artifacts_dir, args.metrics)
    print(BASELINE_DESCRIPTION)
    for turbine, result in report["turbines"].items():
        selected = result["selected_model"]
        metrics = result["metrics"][selected]
        print(f"{turbine}: {selected}; MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
