"""Load repository model artifacts behind the ForecastAgent predictor API."""

from __future__ import annotations

import json
import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "ai" / "models" / "artifacts"


class ModelArtifactError(RuntimeError):
    """A configured turbine model cannot be loaded or used."""


class ArtifactPredictor:
    """Adapt a saved named-feature model to ForecastAgent's feature mappings."""

    def __init__(self, model: Any, feature_columns: Sequence[str], model_version: str, *, model_source: str = "unspecified", warnings: Sequence[str] = (), available_at: str | None = None, feature_timezone: str = "UTC") -> None:
        self.model = model
        self.feature_columns = tuple(feature_columns)
        self.model_id = model_version
        self.model_source = model_source
        self.warnings = tuple(warnings)
        self.available_at = available_at
        self.feature_timezone = feature_timezone

    def predict(self, features: Sequence[Mapping[str, Any]]) -> Sequence[float]:
        rows: list[list[float]] = []
        for index, feature in enumerate(features):
            try:
                values = dict(feature)
                if self.feature_timezone != "UTC":
                    timestamp = feature["valid_time"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
                        raise ValueError("Timezone-aware valid_time required for local SCADA calendar features")
                    local_time = timestamp.astimezone(ZoneInfo(self.feature_timezone))
                    hour = 2 * math.pi * local_time.hour / 24
                    day = 2 * math.pi * local_time.timetuple().tm_yday / 365.25
                    values.update(hour_sin=math.sin(hour), hour_cos=math.cos(hour), day_of_year_sin=math.sin(day), day_of_year_cos=math.cos(day))
                row = [self._value(values, column) for column in self.feature_columns]
                if not all(math.isfinite(value) for value in row):
                    raise ValueError("Non-finite model feature")
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelArtifactError(
                    f"feature row {index} does not contain model columns {self.feature_columns}"
                ) from exc
            rows.append(row)
        try:
            return self.model.predict(rows)
        except Exception as exc:
            raise ModelArtifactError(f"saved model prediction failed: {exc}") from exc

    @staticmethod
    def _value(feature: Mapping[str, Any], column: str) -> float:
        if column in feature:
            return float(feature[column])
        # The repository's fallback artifacts use the legacy SCADA feature names.
        if column == "mean_wind_speed":
            return sum(float(feature[name]) for name in ("wind_speed_80m", "wind_speed_100m", "wind_speed_120m")) / 3.0
        if column == "mean_ambient_temperature":
            return float(feature["temperature_2m"])
        raise KeyError(column)


class ArtifactModelLoader:
    """Resolve each turbine's artifact and feature contract from metrics.json."""

    def __init__(
        self,
        *,
        artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
        metrics_path: Path | str | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.metrics_path = Path(metrics_path) if metrics_path else self.artifacts_dir / "metrics.json"

    def _load_metrics(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"Cannot read model metrics {self.metrics_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ModelArtifactError(f"Model metrics {self.metrics_path} must contain an object")
        return payload

    def load(self, turbine_id: str) -> ArtifactPredictor:
        if turbine_id not in ("turbine_1", "turbine_2"):
            raise ModelArtifactError(f"Unknown turbine {turbine_id!r}")
        selected_models = self._load_metrics().get("selected_models", {})
        selected = selected_models.get(turbine_id) if isinstance(selected_models, dict) else None
        if not isinstance(selected, dict):
            raise ModelArtifactError(f"No model metadata found for {turbine_id}")
        feature_columns = selected.get("features")
        if not isinstance(feature_columns, list) or not feature_columns or not all(isinstance(column, str) and column for column in feature_columns) or len(set(feature_columns)) != len(feature_columns):
            raise ModelArtifactError(f"No feature contract found for {turbine_id}")
        # Historical metrics contain absolute Windows paths. Resolve only the
        # leaf name inside the configured directory, never arbitrary paths.
        recorded_path = selected.get("artifact", f"{turbine_id}.joblib")
        if not isinstance(recorded_path, str) or ".." in PureWindowsPath(recorded_path).parts:
            raise ModelArtifactError(f"Invalid model artifact path for {turbine_id}")
        basename = PureWindowsPath(recorded_path).name
        if not re.fullmatch(rf"{re.escape(turbine_id)}(?:_[A-Za-z0-9_-]+)?\.joblib", basename):
            raise ModelArtifactError(f"Invalid model artifact basename for {turbine_id}: {basename}")
        artifact = self.artifacts_dir / basename
        if artifact.resolve().parent != self.artifacts_dir.resolve():
            raise ModelArtifactError(f"Model artifact escapes the configured directory: {artifact}")
        if not artifact.is_file():
            raise ModelArtifactError(f"Model artifact is missing for {turbine_id}: {artifact}")
        try:
            import joblib

            with artifact.open("rb") as stream:
                artifact_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                stream.seek(0)
                model = joblib.load(stream)
        except Exception as exc:
            raise ModelArtifactError(f"Cannot load model artifact for {turbine_id}: {exc}") from exc
        saved_columns = getattr(model, "feature_columns", None)
        if saved_columns is not None and tuple(saved_columns) != tuple(feature_columns):
            raise ModelArtifactError(f"Artifact feature columns do not match metrics for {turbine_id}")
        expected_width = getattr(getattr(model, "estimator_", model), "n_features_in_", len(feature_columns))
        if expected_width != len(feature_columns):
            raise ModelArtifactError(f"Artifact feature count does not match metrics for {turbine_id}")
        source = str(selected.get("model_source") or "unspecified")
        warnings = []
        is_baseline = "scada" in source.lower() or any(column.startswith("mean_") for column in feature_columns)
        if is_baseline:
            warnings.append(
                f"{turbine_id}: measured-SCADA-weather baseline fallback; weather forecast wind heights "
                "are averaged as a proxy for SCADA wind and temperature_2m is used for SCADA temperature. "
                "This model has not been validated on forecast-weather inputs; SCADA validation metrics "
                "do not measure production forecast accuracy."
            )
        if selected.get("limitation"):
            warnings.append(f"{turbine_id}: {selected['limitation']}")
        feature_timezone = selected.get("feature_timezone", "Asia/Almaty" if is_baseline else "UTC")
        try:
            ZoneInfo(feature_timezone)
        except (ValueError, TypeError, KeyError) as exc:
            raise ModelArtifactError(f"Invalid feature timezone for {turbine_id}: {feature_timezone}") from exc
        cutoffs = [selected.get("available_at"), getattr(model, "available_at_", None)]
        parsed_cutoffs = []
        for cutoff in cutoffs:
            if cutoff is None:
                continue
            try:
                parsed = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError("available_at must include timezone")
                parsed_cutoffs.append(parsed.astimezone(timezone.utc))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ModelArtifactError(f"Invalid model availability for {turbine_id}: {cutoff}") from exc
        available_at = max(parsed_cutoffs).isoformat() if parsed_cutoffs else None
        contract = json.dumps({"features": feature_columns, "model_source": source, "feature_timezone": feature_timezone, "available_at": available_at}, sort_keys=True).encode()
        contract_hash = hashlib.sha256(contract).hexdigest()[:12]
        version = f"{selected.get('selected_model') or artifact.stem}:{artifact_hash[:16]}:{contract_hash}"
        return ArtifactPredictor(model, feature_columns, version, model_source=source, warnings=warnings, available_at=available_at, feature_timezone=feature_timezone)

    __call__ = load
