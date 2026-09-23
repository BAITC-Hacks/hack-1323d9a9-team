"""Load repository model artifacts behind the ForecastAgent predictor API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "ai" / "models" / "artifacts"


class ModelArtifactError(RuntimeError):
    """A configured turbine model cannot be loaded or used."""


class ArtifactPredictor:
    """Adapt a saved named-feature model to ForecastAgent's feature mappings."""

    def __init__(self, model: Any, feature_columns: Sequence[str], model_version: str) -> None:
        self.model = model
        self.feature_columns = tuple(feature_columns)
        self.model_id = model_version

    def predict(self, features: Sequence[Mapping[str, Any]]) -> Sequence[float]:
        rows: list[list[float]] = []
        for index, feature in enumerate(features):
            try:
                row = [self._value(feature, column) for column in self.feature_columns]
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
        self._metrics: dict[str, Any] | None = None

    def _load_metrics(self) -> dict[str, Any]:
        if self._metrics is None:
            try:
                payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ModelArtifactError(f"Cannot read model metrics {self.metrics_path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ModelArtifactError(f"Model metrics {self.metrics_path} must contain an object")
            self._metrics = payload
        return self._metrics

    def load(self, turbine_id: str) -> ArtifactPredictor:
        selected = self._load_metrics().get("selected_models", {}).get(turbine_id)
        if not isinstance(selected, dict):
            raise ModelArtifactError(f"No model metadata found for {turbine_id}")
        feature_columns = selected.get("features")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ModelArtifactError(f"No feature contract found for {turbine_id}")
        # Metrics may have been generated on another machine; artifact names are stable.
        artifact = self.artifacts_dir / f"{turbine_id}.joblib"
        if not artifact.is_file():
            raise ModelArtifactError(f"Model artifact is missing for {turbine_id}: {artifact}")
        try:
            import joblib

            model = joblib.load(artifact)
        except Exception as exc:
            raise ModelArtifactError(f"Cannot load model artifact for {turbine_id}: {exc}") from exc
        version = str(selected.get("selected_model") or selected.get("model_source") or artifact.name)
        return ArtifactPredictor(model, feature_columns, version)

    __call__ = load
