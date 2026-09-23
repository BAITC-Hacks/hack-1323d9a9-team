"""Pydantic request and response schemas for the forecast API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class ForecastRunRequest(BaseModel):
    issue_time: datetime
    horizon_hours: int = Field(default=48, ge=1, le=48)

    @validator("issue_time")
    def issue_time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issue_time must include a timezone, for example Z")
        return value


class ForecastItem(BaseModel):
    issue_time: datetime
    weather_run_time: datetime
    model_version: str
    turbine: str
    valid_time: datetime
    lead_hour: int
    predicted_normalized_power: float
    agent_status: str
    warnings: list[str] = Field(default_factory=list)


class ForecastRunResponse(BaseModel):
    status: str
    issue_time: datetime
    horizon_hours: int
    forecasts: list[ForecastItem]
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str

