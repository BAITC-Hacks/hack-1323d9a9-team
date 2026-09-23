"""Minimal HTTP routes delegating all forecasting work to ForecastService."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import ForecastRunRequest, ForecastRunResponse, HealthResponse
from ..services.forecast_service import ForecastServiceError


router = APIRouter()


def _service(request: Request):
    return request.app.state.forecast_service


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/api/forecast/run", response_model=ForecastRunResponse)
def run_forecast(payload: ForecastRunRequest, request: Request) -> ForecastRunResponse:
    try:
        return _service(request).run(payload.issue_time, payload.horizon_hours)
    except ForecastServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc


@router.get("/api/forecast/latest", response_model=ForecastRunResponse)
def latest_forecast(request: Request) -> ForecastRunResponse:
    try:
        return _service(request).latest()
    except ForecastServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc


@router.get("/api/metrics")
def metrics(request: Request):
    try:
        return _service(request).metrics()
    except ForecastServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc

