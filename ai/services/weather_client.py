"""Leakage-safe access to archived Open-Meteo ECMWF IFS HRES runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


UTC = timezone.utc
RUN_CYCLE = timedelta(hours=6)
DEFAULT_PUBLICATION_SAFETY_DELAY = timedelta(hours=6)
API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
FORECAST_SOURCE = "open_meteo_single_runs"
ARCHIVE_START = datetime(2024, 3, 14, tzinfo=UTC)
# Open-Meteo describes the early archive as IFS 49r1 hindcasts. That cycle
# became operational on 2024-11-12; do not replay its earlier hindcasts as
# forecasts that were publicly available at the historical issue time.
OPERATIONAL_ARCHIVE_START = datetime(2024, 11, 12, tzinfo=UTC)
REQUESTED_HORIZON = timedelta(days=10)
PROVIDER = "Open-Meteo"
MODEL = "ecmwf_ifs"  # ECMWF IFS HRES 9 km in the Open-Meteo API.
HOURLY_VARIABLES = (
    "wind_speed_80m",
    "wind_speed_100m",
    "wind_speed_120m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


class WeatherClientError(RuntimeError):
    """An archived weather request, response, or cache entry is unusable."""


class RunUnavailableError(WeatherClientError):
    """Open-Meteo does not have the requested model cycle."""


def _as_utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _checked_delay(delay: timedelta) -> timedelta:
    if not isinstance(delay, timedelta) or delay < timedelta(0):
        raise ValueError("publication_safety_delay must be a non-negative timedelta")
    return delay


def _iso_utc(value: datetime) -> str:
    return _as_utc(value, "datetime").isoformat().replace("+00:00", "Z")


def select_safe_run(
    issue_time: datetime,
    publication_safety_delay: timedelta = DEFAULT_PUBLICATION_SAFETY_DELAY,
) -> datetime:
    """Select the latest 00/06/12/18 UTC run safely published by issue time."""
    issue = _as_utc(issue_time, "issue_time")
    delay = _checked_delay(publication_safety_delay)
    cutoff = issue - delay
    run = cutoff.replace(hour=(cutoff.hour // 6) * 6, minute=0, second=0, microsecond=0)
    assert_no_leakage(run, issue, delay)
    return run


def assert_no_leakage(
    weather_run_time: datetime,
    issue_time: datetime,
    publication_safety_delay: timedelta = DEFAULT_PUBLICATION_SAFETY_DELAY,
) -> None:
    """Enforce the publication cutoff even when Python assertions are disabled."""
    run = _as_utc(weather_run_time, "weather_run_time")
    issue = _as_utc(issue_time, "issue_time")
    delay = _checked_delay(publication_safety_delay)
    if run + delay > issue:
        raise WeatherClientError(
            f"Weather run {_iso_utc(run)} with safety delay {delay} is later than "
            f"issue time {_iso_utc(issue)}"
        )
    assert run + delay <= issue  # Explicit no-leakage invariant.


@dataclass(frozen=True)
class WeatherMetadata:
    turbine_id: str
    issue_time: datetime
    weather_run_time: datetime
    provider: str
    model: str
    fetch_time: datetime
    input_hash: str
    forecast_source: str = FORECAST_SOURCE
    api_url: str = API_URL
    publication_safety_delay_seconds: float = DEFAULT_PUBLICATION_SAFETY_DELAY.total_seconds()
    raw_response_sha256: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "turbine_id": self.turbine_id,
            "issue_time": _iso_utc(self.issue_time),
            "weather_run_time": _iso_utc(self.weather_run_time),
            "provider": self.provider,
            "model": self.model,
            "fetch_time": _iso_utc(self.fetch_time),
            "input_hash": self.input_hash,
            "forecast_source": self.forecast_source,
            "api_url": self.api_url,
            "publication_safety_delay_seconds": self.publication_safety_delay_seconds,
            "raw_response_sha256": self.raw_response_sha256,
        }


@dataclass(frozen=True)
class HourlyWeather:
    valid_time: datetime
    wind_speed_80m: float | None
    wind_speed_100m: float | None
    wind_speed_120m: float | None
    wind_direction_100m: float | None
    temperature_2m: float | None
    surface_pressure: float | None


@dataclass(frozen=True)
class ArchivedWeatherForecast:
    metadata: WeatherMetadata
    hourly: tuple[HourlyWeather, ...]
    raw_response: dict[str, Any]


class OpenMeteoArchivedWeatherClient:
    """Fetch one historical run, with a single older-cycle availability fallback."""

    def __init__(
        self,
        *,
        turbines_path: Path | str = REPO_ROOT / "data" / "turbines.json",
        cache_dir: Path | str = REPO_ROOT / "data" / "weather_cache",
        publication_safety_delay: timedelta = DEFAULT_PUBLICATION_SAFETY_DELAY,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        opener: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        cache_only: bool = False,
    ) -> None:
        self.turbines_path = Path(turbines_path)
        self.cache_dir = Path(cache_dir)
        self.publication_safety_delay = _checked_delay(publication_safety_delay)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be a non-negative finite number")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.opener = opener
        self.sleep = sleep
        self.clock = clock
        if not isinstance(cache_only, bool):
            raise ValueError("cache_only must be a boolean")
        self.cache_only = cache_only

    def fetch_forecast(self, turbine_id: str, issue_time: datetime) -> ArchivedWeatherForecast:
        """Get the safe run's full hourly forecast for a configured turbine."""
        issue = _as_utc(issue_time, "issue_time")
        latitude, longitude = self._coordinates(turbine_id)
        first_run = select_safe_run(issue, self.publication_safety_delay)
        if first_run < OPERATIONAL_ARCHIVE_START:
            raise RunUnavailableError(
                f"Operational ECMWF Single Runs replay starts at {_iso_utc(OPERATIONAL_ARCHIVE_START)}; "
                f"earlier IFS 49r1 hindcasts cannot be used for issue {_iso_utc(issue)}"
            )
        unavailable: list[str] = []
        for run in (first_run, first_run - RUN_CYCLE):
            if run < OPERATIONAL_ARCHIVE_START:
                continue
            assert_no_leakage(run, issue, self.publication_safety_delay)
            params = self._params(latitude, longitude, run)
            input_hash = self._input_hash(turbine_id, issue, params)
            cached = self._load_cache(turbine_id, issue, run, input_hash)
            if cached is not None:
                return cached
            if self.cache_only:
                unavailable.append(f"run {_iso_utc(run)} is not cached (cache_only=True)")
                continue
            try:
                raw_bytes = self._request(params, run)
                raw_response, hourly = self._parse_response(raw_bytes, run)
            except RunUnavailableError as exc:
                unavailable.append(str(exc))
                continue
            metadata = WeatherMetadata(
                turbine_id=turbine_id,
                issue_time=issue,
                weather_run_time=run,
                provider=PROVIDER,
                model=MODEL,
                fetch_time=_as_utc(self.clock(), "fetch_time"),
                input_hash=input_hash,
                publication_safety_delay_seconds=self.publication_safety_delay.total_seconds(),
                raw_response_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            )
            assert_no_leakage(metadata.weather_run_time, metadata.issue_time, self.publication_safety_delay)
            self._write_cache(input_hash, raw_bytes, metadata)
            return ArchivedWeatherForecast(metadata, hourly, raw_response)
        raise WeatherClientError(
            "Neither the selected run nor the one-cycle fallback is available: "
            + "; ".join(unavailable)
        )

    def _coordinates(self, turbine_id: str) -> tuple[float, float]:
        if not isinstance(turbine_id, str) or not turbine_id:
            raise ValueError("turbine_id must be a non-empty string")
        try:
            turbines = json.loads(self.turbines_path.read_text(encoding="utf-8"))
            location = turbines[turbine_id]
            latitude = float(location["latitude"])
            longitude = float(location["longitude"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WeatherClientError(
                f"Cannot load coordinates for turbine {turbine_id!r} from {self.turbines_path}: {exc}"
            ) from exc
        if not (math.isfinite(latitude) and -90 <= latitude <= 90):
            raise WeatherClientError(f"Invalid latitude for turbine {turbine_id!r}: {latitude}")
        if not (math.isfinite(longitude) and -180 <= longitude <= 180):
            raise WeatherClientError(f"Invalid longitude for turbine {turbine_id!r}: {longitude}")
        return latitude, longitude

    @staticmethod
    def _params(latitude: float, longitude: float, run: datetime) -> dict[str, str]:
        return {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "hourly": ",".join(HOURLY_VARIABLES),
            "models": MODEL,
            "run": run.strftime("%Y-%m-%dT%H:%M"),  # API requires UTC without an offset.
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "timezone": "GMT",
            "timeformat": "unixtime",
            "forecast_days": "10",
        }

    def _input_hash(self, turbine_id: str, issue: datetime, params: dict[str, str]) -> str:
        inputs = {
            "turbine_id": turbine_id,
            "issue_time": _iso_utc(issue),
            "publication_safety_delay_seconds": self.publication_safety_delay.total_seconds(),
            "url": API_URL,
            "params": params,
        }
        encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _request(self, params: dict[str, str], run: datetime) -> bytes:
        request = Request(
            f"{API_URL}?{urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": "wind-forecast-weather-client/1"},
        )
        for attempt in range(self.max_retries + 1):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    return response.read()
            except HTTPError as exc:
                try:
                    detail = self._error_detail(exc.read())
                finally:
                    exc.close()
                if exc.code == 404 or (
                    exc.code in (400, 422) and self._is_run_unavailable(detail)
                ):
                    raise RunUnavailableError(f"run {_iso_utc(run)} unavailable (HTTP {exc.code}: {detail})") from exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise WeatherClientError(
                        f"Open-Meteo request for run {_iso_utc(run)} failed after {attempt + 1} "
                        f"attempt(s) (HTTP {exc.code}: {detail})"
                    ) from exc
            except (URLError, OSError) as exc:
                if attempt == self.max_retries:
                    raise WeatherClientError(
                        f"Open-Meteo request for run {_iso_utc(run)} failed after {attempt + 1} "
                        f"attempt(s): {exc}"
                    ) from exc
            self.sleep(self.retry_backoff_seconds * (2**attempt))
        raise AssertionError("retry loop exited unexpectedly")

    @staticmethod
    def _error_detail(body: bytes) -> str:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict) and isinstance(payload.get("reason"), str):
                return payload["reason"]
        except (ValueError, UnicodeDecodeError):
            pass
        return body.decode("utf-8", errors="replace")[:300] or "no error details"

    @staticmethod
    def _is_run_unavailable(reason: str) -> bool:
        text = reason.lower()
        return "run" in text and any(
            marker in text for marker in ("not available", "unavailable", "not found", "no data")
        )

    def _parse_response(
        self, raw_bytes: bytes, run: datetime
    ) -> tuple[dict[str, Any], tuple[HourlyWeather, ...]]:
        try:
            raw = json.loads(raw_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} returned invalid JSON") from exc
        if not isinstance(raw, dict):
            raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} returned a non-object response")
        if raw.get("error"):
            reason = str(raw.get("reason", "unspecified API error"))
            if self._is_run_unavailable(reason):
                raise RunUnavailableError(f"run {_iso_utc(run)} unavailable: {reason}")
            raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} returned an error: {reason}")
        if raw.get("utc_offset_seconds") != 0 or raw.get("timezone") not in ("GMT", "UTC"):
            raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} did not return UTC times")
        hourly = raw.get("hourly")
        units = raw.get("hourly_units")
        if not isinstance(hourly, dict) or not isinstance(units, dict):
            raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} is missing hourly data or units")
        expected_units = {
            "time": "unixtime",
            "wind_speed_80m": "m/s",
            "wind_speed_100m": "m/s",
            "wind_speed_120m": "m/s",
            "wind_direction_100m": "\u00b0",
            "temperature_2m": "\u00b0C",
            "surface_pressure": "hPa",
        }
        for variable, unit in expected_units.items():
            if units.get(variable) != unit:
                raise WeatherClientError(
                    f"Open-Meteo run {_iso_utc(run)} did not return {variable} in {unit}"
                )
        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} has no hourly valid times")
        for variable in HOURLY_VARIABLES:
            if not isinstance(hourly.get(variable), list) or len(hourly[variable]) != len(times):
                raise WeatherClientError(
                    f"Open-Meteo run {_iso_utc(run)} has missing or misaligned {variable} values"
                )
        records = []
        for index, epoch in enumerate(times):
            if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
                raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} has an invalid valid time")
            try:
                valid_time = datetime.fromtimestamp(epoch, UTC)
            except (OverflowError, OSError, ValueError) as exc:
                raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} has an invalid valid time") from exc
            if (
                valid_time.minute or valid_time.second or valid_time.microsecond
                or not run <= valid_time < run + REQUESTED_HORIZON
                or (records and valid_time != records[-1].valid_time + timedelta(hours=1))
            ):
                raise WeatherClientError(
                    f"Open-Meteo run {_iso_utc(run)} has non-hourly, unordered, "
                    "or out-of-run valid times"
                )
            values = {}
            for variable in HOURLY_VARIABLES:
                value = hourly[variable][index]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise WeatherClientError(f"Open-Meteo run {_iso_utc(run)} has invalid {variable} data")
                try:
                    value = float(value) if value is not None else None
                    if value is not None and not math.isfinite(value):
                        raise ValueError("non-finite value")
                except (OverflowError, ValueError) as exc:
                    raise WeatherClientError(
                        f"Open-Meteo run {_iso_utc(run)} has non-finite {variable} data"
                    ) from exc
                values[variable] = value
            records.append(HourlyWeather(valid_time=valid_time, **values))
        if not any(row.wind_speed_100m is not None and row.temperature_2m is not None for row in records):
            raise RunUnavailableError(f"run {_iso_utc(run)} has no usable wind and temperature forecasts")
        return raw, tuple(records)

    def _cache_paths(self, input_hash: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{input_hash}.json", self.cache_dir / f"{input_hash}.meta.json"

    def _load_cache(
        self, turbine_id: str, issue: datetime, run: datetime, input_hash: str
    ) -> ArchivedWeatherForecast | None:
        raw_path, meta_path = self._cache_paths(input_hash)
        if not raw_path.is_file() or not meta_path.is_file():
            return None
        forecast = self.load_cached_forecast(meta_path)
        if (
            forecast.metadata.turbine_id != turbine_id
            or forecast.metadata.issue_time != issue
            or forecast.metadata.weather_run_time != run
            or forecast.metadata.input_hash != input_hash
        ):
            raise WeatherClientError(f"Invalid weather cache entry {raw_path}: metadata does not match request")
        return forecast

    def load_cached_forecast(self, meta_path: Path | str) -> ArchivedWeatherForecast:
        """Read a verified Single Runs cache entry for inference or training.

        Reconstructing the request hash prevents a renamed observation file or
        mismatched coordinates/run from being accepted as an archived forecast.
        Older cache entries may omit provenance fields added in schema version 2;
        their endpoint is still bound by the original request hash.
        """
        meta_path = Path(meta_path)
        raw_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".json")
        try:
            if not meta_path.name.endswith(".meta.json"):
                raise ValueError("expected a .meta.json cache metadata file")
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("metadata is not an object")
            turbine_id = saved["turbine_id"]
            issue = _as_utc(datetime.fromisoformat(saved["issue_time"]), "issue_time")
            run = _as_utc(datetime.fromisoformat(saved["weather_run_time"]), "weather_run_time")
            if run < OPERATIONAL_ARCHIVE_START or run.hour % 6 or run.minute or run.second or run.microsecond:
                raise ValueError("weather_run_time predates operational IFS 49r1 or is not an ECMWF run cycle")
            latitude, longitude = self._coordinates(turbine_id)
            params = self._params(latitude, longitude, run)
            input_hash = self._input_hash(turbine_id, issue, params)
            # The original client relied on the API's Celsius default.
            legacy_params = {key: value for key, value in params.items() if key != "temperature_unit"}
            legacy_hash = self._input_hash(turbine_id, issue, legacy_params)
            if saved.get("input_hash") == legacy_hash:
                input_hash = legacy_hash
            expected = {
                "turbine_id": turbine_id,
                "issue_time": _iso_utc(issue),
                "weather_run_time": _iso_utc(run),
                "provider": PROVIDER,
                "model": MODEL,
                "input_hash": input_hash,
            }
            if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError("metadata does not match request")
            if meta_path.name != f"{input_hash}.meta.json":
                raise ValueError("cache filename does not match request hash")
            if saved.get("forecast_source", FORECAST_SOURCE) != FORECAST_SOURCE or saved.get("api_url", API_URL) != API_URL:
                raise ValueError("metadata source is not Open-Meteo Single Runs")
            if saved.get("publication_safety_delay_seconds", self.publication_safety_delay.total_seconds()) != self.publication_safety_delay.total_seconds():
                raise ValueError("metadata publication delay does not match client policy")
            fetch_time = _as_utc(datetime.fromisoformat(saved["fetch_time"]), "fetch_time")
            raw_bytes = raw_path.read_bytes()
            raw_hash = saved.get("raw_response_sha256")
            if raw_hash is not None and raw_hash != hashlib.sha256(raw_bytes).hexdigest():
                raise ValueError("raw response checksum does not match metadata")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise WeatherClientError(f"Invalid weather cache entry {raw_path}: {exc}") from exc
        assert_no_leakage(run, issue, self.publication_safety_delay)
        raw_response, hourly = self._parse_response(raw_bytes, run)
        metadata = WeatherMetadata(
            turbine_id, issue, run, PROVIDER, MODEL, fetch_time, input_hash,
            publication_safety_delay_seconds=self.publication_safety_delay.total_seconds(),
            raw_response_sha256=raw_hash,
        )
        return ArchivedWeatherForecast(metadata, hourly, raw_response)

    def _write_cache(self, input_hash: str, raw_bytes: bytes, metadata: WeatherMetadata) -> None:
        raw_path, meta_path = self._cache_paths(input_hash)
        raw_tmp = self.cache_dir / f".{input_hash}.{uuid.uuid4().hex}.raw.tmp"
        meta_tmp = self.cache_dir / f".{input_hash}.{uuid.uuid4().hex}.meta.tmp"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            raw_tmp.write_bytes(raw_bytes)
            meta_tmp.write_text(json.dumps(metadata.to_json(), indent=2) + "\n", encoding="utf-8")
            os.replace(raw_tmp, raw_path)
            os.replace(meta_tmp, meta_path)
        except OSError as exc:
            raise WeatherClientError(f"Cannot write weather cache in {self.cache_dir}: {exc}") from exc
        finally:
            raw_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
