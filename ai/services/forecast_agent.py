"""Autonomous, auditable orchestration for hourly turbine forecasts.

The agent deliberately knows nothing about a particular ML library.  A model
loader returns an object with ``predict(features)`` (or a callable), so model
implementations can be replaced without changing the workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .weather_client import (
    DEFAULT_PUBLICATION_SAFETY_DELAY,
    OpenMeteoArchivedWeatherClient,
    assert_no_leakage,
    select_safe_run,
)


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_DIR = REPO_ROOT / "data" / "forecasts"
TURBINE_IDS = ("turbine_1", "turbine_2")
REQUIRED_WEATHER_FIELDS = (
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
)
_SAVE_LOCK = threading.Lock()


class ForecastAgentError(RuntimeError):
    """A workflow stage could not produce a valid result."""


class Predictor(Protocol):
    """Replaceable model interface used by :class:`ForecastAgent`."""

    def predict(self, features: Sequence[Mapping[str, Any]]) -> Sequence[float]:
        ...


class ModelLoader(Protocol):
    def __call__(self, turbine_id: str) -> Any:
        ...


def _as_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ForecastAgentError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return _as_utc(value, "datetime").isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> Any:
    """Convert workflow inputs to deterministic JSON-compatible values."""
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ForecastAgentError("Inputs must contain only finite numbers")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _get(value: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _parse_time(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value, name)
    if isinstance(value, str):
        try:
            return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")), name)
        except ValueError as exc:
            raise ForecastAgentError(f"{name} is not a valid timezone-aware timestamp") from exc
    raise ForecastAgentError(f"{name} is not a timezone-aware timestamp")


def _model_key(model: Any, turbine_id: str) -> str:
    for attr in ("model_id", "model_version", "name"):
        value = getattr(model, attr, None)
        if value is not None:
            return f"{turbine_id}:{value}"
    cls = type(model)
    return f"{turbine_id}:{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class ForecastPoint:
    turbine_id: str
    valid_time: datetime
    prediction: float

    def to_json(self) -> dict[str, Any]:
        return {
            "turbine_id": self.turbine_id,
            "valid_time": _iso_utc(self.valid_time),
            "prediction": self.prediction,
        }


@dataclass(frozen=True)
class ForecastResult:
    status: str
    message: str
    forecasts: tuple[ForecastPoint, ...]
    metadata: dict[str, Any]
    output_path: Path | None = None
    states: tuple[str, ...] = ()

    @property
    def forecast(self) -> tuple[ForecastPoint, ...]:
        """Convenient singular alias for consumers expecting ``result.forecast``."""
        return self.forecasts

    @property
    def state(self) -> str:
        return self.status

    @property
    def audit_metadata(self) -> dict[str, Any]:
        return self.metadata

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "forecast": [point.to_json() for point in self.forecasts],
            "metadata": self.metadata,
            "states": list(self.states),
        }


class ForecastAgent:
    """Run the complete weather-to-forecast workflow for both turbines."""

    STATES = (
        "FETCH_WEATHER",
        "VALIDATE_INPUT",
        "BUILD_FEATURES",
        "RUN_MODEL",
        "VALIDATE_OUTPUT",
        "SAVE_RESULT",
        "COMPLETE",
        "FAILED",
    )

    def __init__(
        self,
        *,
        weather_client: Any | None = None,
        model_loader: ModelLoader | None = None,
        models: Mapping[str, Any] | None = None,
        output_dir: Path | str = DEFAULT_FORECAST_DIR,
        turbine_ids: Sequence[str] = TURBINE_IDS,
    ) -> None:
        self.weather_client = weather_client if weather_client is not None else OpenMeteoArchivedWeatherClient()
        if model_loader is not None and models is not None:
            raise ValueError("Pass model_loader or models, not both")
        if models is not None:
            frozen_models = dict(models)
            model_loader = lambda turbine_id: frozen_models[turbine_id]
        self.model_loader = model_loader
        self.output_dir = Path(output_dir)
        self.turbine_ids = tuple(turbine_ids)
        if not self.turbine_ids:
            raise ValueError("At least one turbine_id is required")
        if len(set(self.turbine_ids)) != len(self.turbine_ids):
            raise ValueError("turbine_ids must be unique")

    def run(self, issue_time: datetime, horizon_hours: int = 48) -> ForecastResult:
        """Execute every workflow stage once and return an auditable result.

        Stage failures are returned as ``status='FAILED'`` with the failing
        state and a human-readable message; callers do not need exception
        handling to inspect a failed run.
        """
        states: list[str] = []
        try:
            issue = _as_utc(issue_time, "issue_time")
            if issue.minute or issue.second or issue.microsecond:
                raise ForecastAgentError("issue_time must be aligned to a UTC hour")
            if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or not 24 <= horizon_hours <= 48:
                raise ForecastAgentError("horizon_hours must be an integer between 24 and 48")
        except Exception as exc:
            states.append("FAILED")
            return self._failed(str(exc), states)

        try:
            self._enter(states, "FETCH_WEATHER")
            weather = {
                turbine_id: self.weather_client.fetch_forecast(turbine_id, issue)
                for turbine_id in self.turbine_ids
            }

            self._enter(states, "VALIDATE_INPUT")
            validated = self._validate_weather(weather, issue, horizon_hours)

            self._enter(states, "BUILD_FEATURES")
            features = {
                turbine_id: self._build_features(turbine_id, records)
                for turbine_id, records in validated.items()
            }

            self._enter(states, "RUN_MODEL")
            models = {turbine_id: self._load_model(turbine_id) for turbine_id in self.turbine_ids}
            for turbine_id, model in models.items():
                available_at = getattr(model, "available_at", None)
                if available_at is not None and _parse_time(available_at, "model.available_at") > issue:
                    raise ForecastAgentError(f"{turbine_id}: model selection uses data unavailable at issue_time")
            input_hash = self._input_hash(issue, weather, features, models)
            raw_predictions = {
                turbine_id: self._predict(models[turbine_id], features[turbine_id], turbine_id)
                for turbine_id in self.turbine_ids
            }

            self._enter(states, "VALIDATE_OUTPUT")
            forecasts = self._validate_and_clip_predictions(raw_predictions, features)

            self._enter(states, "SAVE_RESULT")
            return self._save_result(issue, input_hash, forecasts, weather, models, states, horizon_hours)
        except Exception as exc:
            if not states or states[-1] != "FAILED":
                states.append("FAILED")
            return self._failed(f"{states[-2] if len(states) > 1 else 'workflow'}: {exc}", states)

    @staticmethod
    def _enter(states: list[str], state: str) -> None:
        if state not in ForecastAgent.STATES:
            raise ForecastAgentError(f"Unknown workflow state {state}")
        states.append(state)

    @staticmethod
    def _failed(message: str, states: Sequence[str]) -> ForecastResult:
        return ForecastResult("FAILED", message, (), {"error": message}, None, tuple(states))

    def _validate_weather(self, weather: Mapping[str, Any], issue: datetime, horizon_hours: int) -> dict[str, tuple[dict[str, Any], ...]]:
        if set(weather) != set(self.turbine_ids):
            raise ForecastAgentError("Weather was not obtained for exactly the configured turbines")
        validated: dict[str, tuple[dict[str, Any], ...]] = {}
        expected = tuple(issue + timedelta(hours=lead) for lead in range(1, horizon_hours + 1))
        delay = getattr(self.weather_client, "publication_safety_delay", DEFAULT_PUBLICATION_SAFETY_DELAY)
        safe_run = select_safe_run(issue, delay)
        for turbine_id, result in weather.items():
            metadata = _get(result, "metadata")
            if metadata is None:
                raise ForecastAgentError(f"{turbine_id}: weather metadata is missing")
            metadata_turbine = _get(metadata, "turbine_id")
            if metadata_turbine is not None and metadata_turbine != turbine_id:
                raise ForecastAgentError(
                    f"{turbine_id}: weather metadata belongs to {metadata_turbine!r}"
                )
            metadata_issue = _get(metadata, "issue_time")
            if metadata_issue is not None and _parse_time(metadata_issue, f"{turbine_id}.issue_time") != issue:
                raise ForecastAgentError(f"{turbine_id}: weather issue_time does not match the run issue_time")
            run_time = _parse_time(_get(metadata, "weather_run_time"), f"{turbine_id}.weather_run_time")
            assert_no_leakage(run_time, issue, delay)
            if run_time > safe_run:
                raise ForecastAgentError(
                    f"{turbine_id}: weather run {_iso_utc(run_time)} is newer than the safe run "
                    f"{_iso_utc(safe_run)}"
                )
            hourly = _get(result, "hourly")
            if not isinstance(hourly, Sequence) or isinstance(hourly, (str, bytes)) or not hourly:
                raise ForecastAgentError(f"{turbine_id}: weather hourly data is empty")
            rows: list[dict[str, Any]] = []
            for index, record in enumerate(hourly):
                valid_time = _parse_time(_get(record, "valid_time"), f"{turbine_id}.valid_time[{index}]")
                # A run may contain historical hours and a ten-day forecast. Only
                # the requested future window belongs in this forecast artifact.
                if valid_time <= issue or valid_time > expected[-1]:
                    continue
                row: dict[str, Any] = {"valid_time": valid_time}
                for field in REQUIRED_WEATHER_FIELDS:
                    value = _get(record, field)
                    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise ForecastAgentError(f"{turbine_id}: incomplete weather field {field} at {_iso_utc(valid_time)}")
                    row[field] = float(value)
                rows.append(row)
            rows.sort(key=lambda row: row["valid_time"])
            times = tuple(row["valid_time"] for row in rows)
            if len(set(times)) != len(times):
                raise ForecastAgentError(f"{turbine_id}: duplicate weather valid_time values")
            if times != expected:
                raise ForecastAgentError(
                    f"{turbine_id}: weather must cover every hour from {_iso_utc(expected[0])} "
                    f"through {_iso_utc(expected[-1])}; got {len(times)} rows for {horizon_hours} hours"
                )
            validated[turbine_id] = tuple(rows)
        return validated

    @staticmethod
    def _build_features(turbine_id: str, records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        features: list[dict[str, Any]] = []
        for row in records:
            timestamp = _parse_time(row["valid_time"], "valid_time")
            hour_angle = 2 * math.pi * timestamp.hour / 24
            day_angle = 2 * math.pi * timestamp.timetuple().tm_yday / 365.25
            features.append({
                "turbine_id": turbine_id,
                "valid_time": timestamp,
                "wind_speed_80m": row["wind_speed_80m"],
                "wind_speed_100m": row["wind_speed_100m"],
                "wind_speed_120m": row["wind_speed_120m"],
                "wind_direction_100m": row["wind_direction_100m"],
                "temperature_2m": row["temperature_2m"],
                "surface_pressure": row["surface_pressure"],
                "hour_sin": math.sin(hour_angle),
                "hour_cos": math.cos(hour_angle),
                "day_of_year_sin": math.sin(day_angle),
                "day_of_year_cos": math.cos(day_angle),
            })
        return tuple(features)

    def _load_model(self, turbine_id: str) -> Any:
        if self.model_loader is None:
            raise ForecastAgentError(f"No model loader configured for {turbine_id}")
        try:
            if callable(self.model_loader):
                model = self.model_loader(turbine_id)
            elif callable(getattr(self.model_loader, "load", None)):
                model = self.model_loader.load(turbine_id)
            else:
                raise TypeError("model_loader must be callable or provide load(turbine_id)")
        except Exception as exc:
            raise ForecastAgentError(f"Could not load model for {turbine_id}: {exc}") from exc
        if model is None or not callable(getattr(model, "predict", None)) and not callable(model):
            raise ForecastAgentError(f"Loaded model for {turbine_id} has no predict(features) interface")
        return model

    @staticmethod
    def _predict(model: Any, features: Sequence[Mapping[str, Any]], turbine_id: str) -> tuple[float, ...]:
        try:
            output = model.predict(features) if callable(getattr(model, "predict", None)) else model(features)
            if hasattr(output, "tolist"):
                output = output.tolist()
            values = tuple(output)
        except Exception as exc:
            raise ForecastAgentError(f"Model prediction failed for {turbine_id}: {exc}") from exc
        if len(values) != len(features):
            raise ForecastAgentError(f"Model for {turbine_id} returned {len(values)} predictions for {len(features)} rows")
        return values

    @staticmethod
    def _validate_and_clip_predictions(
        predictions: Mapping[str, Sequence[float]], features: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> tuple[ForecastPoint, ...]:
        points: list[ForecastPoint] = []
        for turbine_id, values in predictions.items():
            for feature, value in zip(features[turbine_id], values):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ForecastAgentError(f"Invalid non-finite prediction for {turbine_id}")
                clipped = min(1.0, max(0.0, float(value)))
                points.append(ForecastPoint(turbine_id, _parse_time(feature["valid_time"], "valid_time"), clipped))
        if not points:
            raise ForecastAgentError("Model produced an empty forecast")
        if len({(point.turbine_id, point.valid_time) for point in points}) != len(points):
            raise ForecastAgentError("Forecast output contains duplicate turbine/valid_time rows")
        return tuple(sorted(points, key=lambda point: (point.valid_time, point.turbine_id)))

    @staticmethod
    def _input_hash(issue: datetime, weather: Mapping[str, Any], features: Mapping[str, Any], models: Mapping[str, Any]) -> str:
        payload = {
            "issue_time": _iso_utc(issue),
            "weather": {
                turbine_id: {
                    "metadata": {
                        "turbine_id": _get(_get(result, "metadata"), "turbine_id"),
                        "weather_run_time": _canonical(_get(_get(result, "metadata"), "weather_run_time")),
                        "provider": _get(_get(result, "metadata"), "provider"),
                        "model": _get(_get(result, "metadata"), "model"),
                        "forecast_source": _get(_get(result, "metadata"), "forecast_source"),
                        "input_hash": _get(_get(result, "metadata"), "input_hash"),
                    },
                    "hourly": [
                        {
                            "valid_time": _canonical(_get(row, "valid_time")),
                            **{field: _get(row, field) for field in REQUIRED_WEATHER_FIELDS},
                        }
                        for row in _get(result, "hourly")
                    ],
                }
                for turbine_id, result in sorted(weather.items())
            },
            "features": _canonical(features),
            "models": {turbine_id: _model_key(models[turbine_id], turbine_id) for turbine_id in sorted(models)},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _save_result(self, issue: datetime, input_hash: str, forecasts: tuple[ForecastPoint, ...], weather: Mapping[str, Any], models: Mapping[str, Any], states: Sequence[str], horizon_hours: int) -> ForecastResult:
        # Requests handled concurrently in this process must allocate versions
        # consistently. Publish only the final COMPLETE JSON, in one replace.
        with _SAVE_LOCK:
            version = self._next_version(issue, input_hash)
            path = self.output_dir / f"forecast_{self._issue_token(issue)}_v{version}_{input_hash[:12]}.json"
            complete_states = (*states, "COMPLETE")
            metadata = self._metadata(issue, input_hash, weather, models, version, path, complete_states, horizon_hours)
            result = ForecastResult("COMPLETE", "Forecast completed", forecasts, metadata, path, complete_states)
            self._write_artifact(path, result)
            return result

    def _metadata(self, issue: datetime, input_hash: str, weather: Mapping[str, Any], models: Mapping[str, Any], version: int, path: Path, states: Sequence[str], horizon_hours: int) -> dict[str, Any]:
        return {
            "issue_time": _iso_utc(issue),
            "horizon_hours": horizon_hours,
            "input_hash": input_hash,
            "forecast_version": version,
            "provider": "Open-Meteo",
            "model": "ecmwf_ifs",
            "turbines": list(self.turbine_ids),
            "weather": {
                turbine_id: {
                    "weather_run_time": _iso_utc(_parse_time(_get(_get(result, "metadata"), "weather_run_time"), "weather_run_time")),
                    "provider": _get(_get(result, "metadata"), "provider"),
                    "model": _get(_get(result, "metadata"), "model"),
                    "forecast_source": _get(_get(result, "metadata"), "forecast_source"),
                    "publication_safety_delay_seconds": getattr(self.weather_client, "publication_safety_delay", DEFAULT_PUBLICATION_SAFETY_DELAY).total_seconds(),
                    "raw_response_sha256": _get(_get(result, "metadata"), "raw_response_sha256"),
                    "fetch_time": _iso_utc(_parse_time(_get(_get(result, "metadata"), "fetch_time"), "fetch_time")),
                    "input_hash": _get(_get(result, "metadata"), "input_hash"),
                }
                for turbine_id, result in weather.items()
            },
            "models": {turbine_id: _model_key(models[turbine_id], turbine_id) for turbine_id in self.turbine_ids},
            "model_details": {
                turbine_id: {
                    "model_source": getattr(models[turbine_id], "model_source", "unspecified"),
                    "available_at": _canonical(getattr(models[turbine_id], "available_at", None)),
                    "feature_timezone": getattr(models[turbine_id], "feature_timezone", "UTC"),
                    "warnings": list(getattr(models[turbine_id], "warnings", ())),
                }
                for turbine_id in self.turbine_ids
            },
            "warnings": list(dict.fromkeys(
                warning for turbine_id in self.turbine_ids
                for warning in getattr(models[turbine_id], "warnings", ())
            )),
            "state_history": list(states),
            "output_path": str(path),
        }

    def _write_artifact(self, path: Path, result: ForecastResult) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _next_version(self, issue: datetime, input_hash: str) -> int:
        token = self._issue_token(issue)
        highest = 0
        for path in self.output_dir.glob(f"forecast_{token}_v*.json"):
            match = re.match(rf"forecast_{re.escape(token)}_v(\d+)_([0-9a-f]+)\.json$", path.name)
            if match:
                highest = max(highest, int(match.group(1)))
                if match.group(2) == input_hash[:12]:
                    return int(match.group(1))
        return highest + 1

    @staticmethod
    def _issue_token(issue: datetime) -> str:
        return _as_utc(issue, "issue_time").strftime("%Y%m%dT%H%M%SZ")
