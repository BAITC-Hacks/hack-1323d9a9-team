"""Offline tests for archived weather run selection and transport."""

import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from ai.services.weather_client import (
    API_URL,
    HOURLY_VARIABLES,
    OpenMeteoArchivedWeatherClient,
    WeatherClientError,
    assert_no_leakage,
    select_safe_run,
)


UTC = timezone.utc
ISSUE = datetime(2026, 2, 1, 13, 0, tzinfo=UTC)
FETCH = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)
HOURLY_RESPONSE = {
    "timezone": "GMT",
    "utc_offset_seconds": 0,
    "hourly_units": {
        "time": "unixtime",
        "wind_speed_80m": "m/s",
        "wind_speed_100m": "m/s",
        "wind_speed_120m": "m/s",
        "wind_direction_100m": "°",
        "temperature_2m": "°C",
        "surface_pressure": "hPa",
    },
    "hourly": {
        "time": [1769990400, 1769994000],  # 2026-02-02 00:00/01:00 UTC
        "wind_speed_80m": [6.1, 6.2],
        "wind_speed_100m": [7.1, 7.2],
        "wind_speed_120m": [8.1, 8.2],
        "wind_direction_100m": [245, 246],
        "temperature_2m": [-2.0, -1.8],
        "surface_pressure": [914.0, 914.1],
    },
}
BODY = json.dumps(HOURLY_RESPONSE, separators=(",", ":")).encode("utf-8")


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body


class FakeOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request.full_url, timeout))
        if not self.outcomes:
            raise AssertionError("Unexpected network request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


def unavailable_http_error():
    body = b'{"error":true,"reason":"Run is not available"}'
    return HTTPError(API_URL, 400, "Bad Request", {}, io.BytesIO(body))


class SafeRunTests(unittest.TestCase):
    def test_default_delay_and_cycle_boundaries(self):
        cases = {
            datetime(2026, 2, 1, 5, 59, tzinfo=UTC): datetime(2026, 1, 31, 18, tzinfo=UTC),
            datetime(2026, 2, 1, 6, 0, tzinfo=UTC): datetime(2026, 2, 1, 0, tzinfo=UTC),
            datetime(2026, 2, 1, 11, 59, tzinfo=UTC): datetime(2026, 2, 1, 0, tzinfo=UTC),
            datetime(2026, 2, 1, 12, 0, tzinfo=UTC): datetime(2026, 2, 1, 6, tzinfo=UTC),
        }
        for issue, expected in cases.items():
            with self.subTest(issue=issue):
                self.assertEqual(select_safe_run(issue), expected)
                self.assertLessEqual(expected + timedelta(hours=6), issue)

    def test_converts_aware_offset_to_utc_and_supports_custom_delay(self):
        local_issue = datetime(2026, 2, 1, 18, 30, tzinfo=timezone(timedelta(hours=5)))
        run = select_safe_run(local_issue, timedelta(hours=7))
        self.assertEqual(run, datetime(2026, 2, 1, 6, tzinfo=UTC))
        self.assertIs(run.tzinfo, UTC)

    def test_every_quarter_hour_obeys_explicit_no_leakage_assertion(self):
        start = datetime(2026, 2, 1, tzinfo=UTC)
        delay = timedelta(hours=6)
        for quarter in range(48 * 4):
            issue = start + timedelta(minutes=15 * quarter)
            run = select_safe_run(issue, delay)
            assert_no_leakage(run, issue, delay)
            self.assertLessEqual(run + delay, issue)

    def test_rejects_naive_time_and_unsafe_run(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            select_safe_run(datetime(2026, 2, 1, 12))
        with self.assertRaisesRegex(WeatherClientError, "later than issue time"):
            assert_no_leakage(datetime(2026, 2, 1, 12, tzinfo=UTC), ISSUE)


class WeatherClientTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache_dir = Path(self.temp_dir.name) / "cache"

    def make_client(self, opener, **kwargs):
        return OpenMeteoArchivedWeatherClient(
            cache_dir=self.cache_dir,
            opener=opener,
            clock=lambda: FETCH,
            retry_backoff_seconds=0,
            sleep=lambda seconds: None,
            **kwargs,
        )

    def test_fetch_uses_single_run_ecmwf_parameters_and_caches_raw_bytes(self):
        opener = FakeOpener([BODY])
        client = self.make_client(opener)
        forecast = client.fetch_forecast("turbine_1", ISSUE)

        self.assertEqual(len(opener.calls), 1)
        url, timeout = opener.calls[0]
        self.assertEqual(urlparse(url).scheme + "://" + urlparse(url).netloc + urlparse(url).path, API_URL)
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["run"], ["2026-02-01T06:00"])
        self.assertEqual(params["models"], ["ecmwf_ifs"])
        self.assertEqual(params["hourly"], [",".join(HOURLY_VARIABLES)])
        self.assertEqual(params["wind_speed_unit"], ["ms"])
        self.assertEqual(params["timezone"], ["GMT"])
        self.assertEqual(params["timeformat"], ["unixtime"])
        self.assertEqual(params["latitude"], ["43.645138889"])
        self.assertEqual(params["longitude"], ["78.535611111"])
        self.assertEqual(timeout, 15.0)

        metadata = forecast.metadata
        self.assertEqual(metadata.turbine_id, "turbine_1")
        self.assertEqual(metadata.issue_time, ISSUE)
        self.assertEqual(metadata.weather_run_time, datetime(2026, 2, 1, 6, tzinfo=UTC))
        self.assertEqual(metadata.provider, "Open-Meteo")
        self.assertEqual(metadata.model, "ecmwf_ifs")
        self.assertEqual(metadata.fetch_time, FETCH)
        self.assertEqual(len(metadata.input_hash), 64)
        self.assertLessEqual(metadata.weather_run_time + client.publication_safety_delay, ISSUE)
        self.assertEqual(forecast.hourly[0].valid_time.tzinfo, UTC)
        self.assertEqual(forecast.hourly[0].valid_time, datetime.fromtimestamp(1769990400, UTC))
        self.assertEqual(forecast.hourly[0].wind_speed_100m, 7.1)

        raw_path = self.cache_dir / f"{metadata.input_hash}.json"
        meta_path = self.cache_dir / f"{metadata.input_hash}.meta.json"
        self.assertEqual(raw_path.read_bytes(), BODY)
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, metadata.to_json())

        cached = client.fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(cached, forecast)

    def test_unavailable_selected_run_falls_back_exactly_one_cycle(self):
        opener = FakeOpener([unavailable_http_error(), BODY])
        client = self.make_client(opener)
        forecast = client.fetch_forecast("turbine_2", ISSUE)
        runs = [parse_qs(urlparse(url).query)["run"][0] for url, _ in opener.calls]
        self.assertEqual(runs, ["2026-02-01T06:00", "2026-02-01T00:00"])
        self.assertEqual(forecast.metadata.weather_run_time, datetime(2026, 2, 1, 0, tzinfo=UTC))
        self.assertLessEqual(
            forecast.metadata.weather_run_time + client.publication_safety_delay,
            forecast.metadata.issue_time,
        )

    def test_api_error_body_also_triggers_one_cycle_fallback(self):
        opener = FakeOpener([b'{"error":true,"reason":"Run is not available"}', BODY])
        forecast = self.make_client(opener).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(forecast.metadata.weather_run_time, datetime(2026, 2, 1, 0, tzinfo=UTC))

    def test_does_not_try_a_third_cycle(self):
        opener = FakeOpener([unavailable_http_error(), unavailable_http_error()])
        with self.assertRaisesRegex(WeatherClientError, "one-cycle fallback"):
            self.make_client(opener).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(len(opener.calls), 2)

    def test_retries_transient_transport_error_then_succeeds(self):
        opener = FakeOpener([URLError("temporary DNS failure"), BODY])
        forecast = self.make_client(opener, max_retries=1).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(len(forecast.hourly), 2)

    def test_exhausted_retry_has_useful_error_and_does_not_fall_back(self):
        opener = FakeOpener([URLError("timeout"), URLError("timeout")])
        with self.assertRaisesRegex(WeatherClientError, "2026-02-01T06:00:00Z.*2 attempt"):
            self.make_client(opener, max_retries=1).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(len(opener.calls), 2)

    def test_rejects_non_utc_or_wrong_wind_units(self):
        for change in ({"utc_offset_seconds": 3600}, {"hourly_units": {**HOURLY_RESPONSE["hourly_units"], "wind_speed_80m": "km/h"}}):
            with self.subTest(change=change):
                bad = {**HOURLY_RESPONSE, **change}
                opener = FakeOpener([json.dumps(bad).encode("utf-8")])
                with self.assertRaises(WeatherClientError):
                    self.make_client(opener).fetch_forecast("turbine_1", ISSUE)
                self.assertFalse(list(self.cache_dir.glob("*.json")))


if __name__ == "__main__":
    unittest.main()
