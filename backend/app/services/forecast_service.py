"""Small application service that exposes ForecastAgent to HTTP handlers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
            model_loader=ArtifactModelLoader(metrics_path=self.metrics_path),
            output_dir=self.output_dir,
        )

    def run(self, issue_time: datetime, horizon_hours: int) -> ForecastRunResponse:
        result: ForecastResult = self.agent.run(issue_time)
        if result.status != "COMPLETE":
            raise self._agent_error(result)
        response = self._response_from_result(result, horizon_hours)
        if not response.forecasts:
            raise ForecastServiceError(
                f"Agent completed without forecast rows in the requested {horizon_hours}-hour horizon",
                code="agent_failure",
                status_code=500,
            )
        return response

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
        issue_time = _parse_utc(result.metadata.get("issue_time"))
        weather_meta = result.metadata.get("weather", {})
        model_meta = result.metadata.get("models", {})
        rows: list[ForecastItem] = []
        warnings: list[str] = []
        for point in result.forecasts:
            valid_time = _parse_utc(point.to_json()["valid_time"])
            lead_seconds = (valid_time - issue_time).total_seconds()
            lead_hour = int(round(lead_seconds / 3600))
            if lead_hour < 1 or lead_hour > horizon_hours:
                continue
            turbine_weather = weather_meta.get(point.turbine_id, {})
            model_version = str(model_meta.get(point.turbine_id, "unknown"))
            if ":" in model_version:
                model_version = model_version.split(":", 1)[1]
            rows.append(ForecastItem(
                issue_time=issue_time,
                weather_run_time=_parse_utc(turbine_weather.get("weather_run_time")),
                model_version=model_version,
                turbine=point.turbine_id,
                valid_time=valid_time,
                lead_hour=lead_hour,
                predicted_normalized_power=float(point.prediction),
                agent_status=result.status,
                warnings=list(warnings),
            ))
        return ForecastRunResponse(
            status=result.status,
            issue_time=issue_time,
            horizon_hours=horizon_hours,
            forecasts=rows,
            warnings=warnings,
            metadata=result.metadata,
        )

    @classmethod
    def _response_from_payload(cls, payload: dict[str, Any]) -> ForecastRunResponse:
        if payload.get("status") != "COMPLETE":
            raise ForecastServiceError("Latest artifact is not complete", code="agent_failure", status_code=500)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ForecastServiceError("Latest artifact has no metadata", code="agent_failure", status_code=500)
        issue_time = _parse_utc(metadata.get("issue_time"))
        weather_meta = metadata.get("weather", {})
        model_meta = metadata.get("models", {})
        rows = []
        for raw in payload.get("forecast", []):
            turbine = raw.get("turbine_id")
            model_version = str(model_meta.get(turbine, "unknown"))
            if ":" in model_version:
                model_version = model_version.split(":", 1)[1]
            rows.append(ForecastItem(
                issue_time=issue_time,
                weather_run_time=_parse_utc(weather_meta.get(turbine, {}).get("weather_run_time")),
                model_version=model_version,
                turbine=turbine,
                valid_time=_parse_utc(raw.get("valid_time")),
                lead_hour=int(round((_parse_utc(raw.get("valid_time")) - issue_time).total_seconds() / 3600)),
                predicted_normalized_power=float(raw.get("prediction")),
                agent_status="COMPLETE",
                warnings=[],
            ))
        return ForecastRunResponse(
            status="COMPLETE",
            issue_time=issue_time,
            horizon_hours=max((row.lead_hour for row in rows), default=0),
            forecasts=rows,
            warnings=[],
            metadata=metadata,
        )
