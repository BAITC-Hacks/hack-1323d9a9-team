"""Pydantic request and response schemas for the forecast API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, validator


class ForecastRunRequest(BaseModel):
    issue_time: datetime
    horizon_hours: int = Field(default=48, ge=24, le=48, strict=True)

    @validator("issue_time")
    def issue_time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issue_time must include a timezone, for example Z")
        utc = value.astimezone(timezone.utc)
        if utc.minute or utc.second or utc.microsecond:
            raise ValueError("issue_time must be aligned to a UTC hour")
        return value


class ForecastItem(BaseModel):
    issue_time: datetime
    weather_run_time: datetime
    model_version: str
    turbine: str
    valid_time: datetime
    lead_hour: int = Field(ge=1, le=48)
    predicted_normalized_power: float = Field(ge=0, le=1, allow_inf_nan=False)
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

