"""FastAPI entry point for the wind forecasting hackathon service."""

from __future__ import annotations

from fastapi import FastAPI

from .api.routes import router
from .services.forecast_service import ForecastService


def create_app(*, forecast_service: ForecastService | None = None, agent: object | None = None) -> FastAPI:
    if forecast_service is not None and agent is not None:
        raise ValueError("Pass forecast_service or agent, not both")
    app = FastAPI(title="Wind Forecasting API", version="1.0.0")
    app.state.forecast_service = forecast_service or ForecastService(agent=agent)
    app.include_router(router)
    return app


app = create_app()
