"""Small application service that exposes ForecastAgent to HTTP handlers."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai.services.forecast_agent import ForecastAgent, ForecastResult

from ..schemas import ForecastItem, ForecastRunResponse
from .model_loader import ArtifactModelLoader


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FORECAST_DIR = REPO_ROOT / "data" / "forecasts"
DEFAULT_METRICS_PATH = REPO_ROOT / "ai" / "models" / "artifacts" / "metrics.json"


class ForecastServiceError(RuntimeError):
    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ForecastServiceError("Forecast artifact contains an invalid timestamp", code="agent_failure", status_code=500)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForecastServiceError("Forecast artifact contains an invalid timestamp", code="agent_failure", status_code=500) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForecastServiceError("Forecast artifact timestamp is not timezone-aware", code="agent_failure", status_code=500)
    return parsed.astimezone(UTC)


class ForecastService:
    def __init__(
        self,
        *,
        agent: Any | None = None,
        output_dir: Path | str = DEFAULT_FORECAST_DIR,
        metrics_path: Path | str = DEFAULT_METRICS_PATH,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metrics_path = Path(metrics_path)
        self.agent = agent if agent is not None else ForecastAgent(
            model_loader=ArtifactModelLoader(artifacts_dir=self.metrics_path.parent, metrics_path=self.metrics_path),
            output_dir=self.output_dir,
        )

    def run(self, issue_time: datetime, horizon_hours: int) -> ForecastRunResponse:
        result: ForecastResult = self.agent.run(issue_time, horizon_hours=horizon_hours)
        if result.status != "COMPLETE":
            raise self._agent_error(result)
        return self._response_from_result(result, horizon_hours)

    def latest(self) -> ForecastRunResponse:
        candidates = sorted(self.output_dir.glob("forecast_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise ForecastServiceError("No forecast has been generated yet", code="not_found", status_code=404)
        try:
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            return self._response_from_payload(payload)
        except ForecastServiceError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ForecastServiceError(f"Latest forecast artifact is invalid: {exc}", code="agent_failure", status_code=500) from exc

    def metrics(self) -> dict[str, Any]:
        path = self.metrics_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ForecastServiceError(f"Cannot read model metrics: {exc}", code="not_found", status_code=404) from exc
        if not isinstance(payload, dict):
            raise ForecastServiceError("Model metrics must contain an object", code="agent_failure", status_code=500)
        return payload

    @staticmethod
    def _agent_error(result: ForecastResult) -> ForecastServiceError:
        failed_state = result.states[-2] if len(result.states) >= 2 and result.states[-1] == "FAILED" else "FAILED"
        if failed_state == "FETCH_WEATHER":
            return ForecastServiceError(result.message, code="weather_unavailable", status_code=503)
        if failed_state == "RUN_MODEL":
            return ForecastServiceError(result.message, code="model_unavailable", status_code=503)
        return ForecastServiceError(result.message, code="agent_failure", status_code=500)

    @classmethod
    def _response_from_result(cls, result: ForecastResult, horizon_hours: int) -> ForecastRunResponse:
        if result.metadata.get("horizon_hours") != horizon_hours:
            raise ForecastServiceError("Agent returned a different forecast horizon", code="agent_failure", status_code=500)
        return cls._response_from_payload(result.to_json())

    @classmethod
    def _response_from_payload(cls, payload: dict[str, Any]) -> ForecastRunResponse:
        try:
            return cls._validated_response(payload)
        except ForecastServiceError:
            raise
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ForecastServiceError(f"Invalid forecast artifact: {exc}", code="agent_failure", status_code=500) from exc

    @classmethod
    def _validated_response(cls, payload: dict[str, Any]) -> ForecastRunResponse:
        if payload.get("status") != "COMPLETE":
            raise ForecastServiceError("Latest artifact is not complete", code="agent_failure", status_code=500)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ForecastServiceError("Latest artifact has no metadata", code="agent_failure", status_code=500)
        issue_time = _parse_utc(metadata.get("issue_time"))
        if issue_time.minute or issue_time.second or issue_time.microsecond:
            raise ValueError("issue_time must be aligned to a UTC hour")
        horizon = metadata.get("horizon_hours")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 24 <= horizon <= 48:
            raise ValueError("missing or invalid horizon_hours")
        turbines = metadata.get("turbines")
        if not isinstance(turbines, list) or not turbines or len(set(turbines)) != len(turbines):
            raise ValueError("missing or invalid turbines")
        warnings = metadata.get("warnings", [])
        if not isinstance(warnings, list) or not all(isinstance(warning, str) for warning in warnings):
            raise ValueError("invalid forecast warnings")
        weather_meta = metadata.get("weather", {})
        model_meta = metadata.get("models", {})
        model_details = metadata.get("model_details", {})
        rows = []
        for raw in payload.get("forecast", []):
            turbine = raw.get("turbine_id")
            valid_time = _parse_utc(raw.get("valid_time"))
            lead_hours = (valid_time - issue_time).total_seconds() / 3600
            if not lead_hours.is_integer() or not 1 <= lead_hours <= horizon:
                raise ValueError("forecast row lies outside the exact hourly horizon")
            turbine_weather = weather_meta.get(turbine, {})
            weather_run_time = _parse_utc(turbine_weather.get("weather_run_time"))
            if weather_run_time > issue_time:
                raise ValueError("forecast weather run is newer than issue_time")
            delay = turbine_weather.get("publication_safety_delay_seconds", 0)
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
                raise ValueError("invalid weather publication safety delay")
            if weather_run_time + timedelta(seconds=delay) > issue_time:
                raise ValueError("forecast weather run was not available at issue_time")
            available_at = model_details.get(turbine, {}).get("available_at")
            if available_at is not None and _parse_utc(available_at) > issue_time:
                raise ValueError("model selection uses data unavailable at issue_time")
            model_version = str(model_meta.get(turbine, "unknown"))
            if ":" in model_version:
                model_version = model_version.split(":", 1)[1]
            rows.append(ForecastItem(
                issue_time=issue_time,
                weather_run_time=weather_run_time,
                model_version=model_version,
                turbine=turbine,
                valid_time=valid_time,
                lead_hour=int(lead_hours),
                predicted_normalized_power=float(raw.get("prediction")),
                agent_status="COMPLETE",
                warnings=warnings,
            ))
        expected = {(turbine, hour) for turbine in turbines for hour in range(1, horizon + 1)}
        actual = {(row.turbine, row.lead_hour) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError("forecast must contain exactly one row per turbine and horizon hour")
        return ForecastRunResponse(
            status="COMPLETE",
            issue_time=issue_time,
            horizon_hours=horizon,
            forecasts=rows,
            warnings=warnings,
            metadata=metadata,
        )
