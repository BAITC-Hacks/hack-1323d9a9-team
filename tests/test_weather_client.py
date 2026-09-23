"""Offline tests for archived weather run selection and transport."""

import copy
import hashlib
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
    FORECAST_SOURCE,
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
        self.assertEqual(params["temperature_unit"], ["celsius"])
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
        self.assertEqual(metadata.forecast_source, FORECAST_SOURCE)
        self.assertEqual(metadata.api_url, API_URL)
        self.assertEqual(metadata.publication_safety_delay_seconds, 21600)
        self.assertEqual(metadata.raw_response_sha256, hashlib.sha256(BODY).hexdigest())
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
        error = unavailable_http_error()
        opener = FakeOpener([error, BODY])
        client = self.make_client(opener)
        forecast = client.fetch_forecast("turbine_2", ISSUE)
        runs = [parse_qs(urlparse(url).query)["run"][0] for url, _ in opener.calls]
        self.assertEqual(runs, ["2026-02-01T06:00", "2026-02-01T00:00"])
        self.assertTrue(error.closed)
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

    def test_fatal_http_error_closes_response_and_preserves_details_and_cause(self):
        error = HTTPError(API_URL, 403, "Forbidden", {}, io.BytesIO(b'{"reason":"Access denied"}'))
        with self.assertRaisesRegex(WeatherClientError, "HTTP 403: Access denied") as raised:
            self.make_client(FakeOpener([error])).fetch_forecast("turbine_1", ISSUE)
        self.assertTrue(error.closed)
        self.assertIs(raised.exception.__cause__, error)

    def test_rejects_non_utc_or_wrong_wind_units(self):
        for change in ({"utc_offset_seconds": 3600}, {"hourly_units": {**HOURLY_RESPONSE["hourly_units"], "wind_speed_80m": "km/h"}}):
            with self.subTest(change=change):
                bad = {**HOURLY_RESPONSE, **change}
                opener = FakeOpener([json.dumps(bad).encode("utf-8")])
                with self.assertRaises(WeatherClientError):
                    self.make_client(opener).fetch_forecast("turbine_1", ISSUE)
                self.assertFalse(list(self.cache_dir.glob("*.json")))

    def test_cache_only_never_opens_network_and_can_use_cached_older_run(self):
        missing = FakeOpener([])
        with self.assertRaisesRegex(WeatherClientError, "cache_only=True"):
            self.make_client(missing, cache_only=True).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(missing.calls, [])
        original = self.make_client(FakeOpener([unavailable_http_error(), BODY])).fetch_forecast("turbine_1", ISSUE)
        cached = self.make_client(missing, cache_only=True).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(cached, original)
        self.assertEqual(missing.calls, [])

    def test_rejects_preoperational_hindcasts_before_network_access(self):
        opener = FakeOpener([])
        with self.assertRaisesRegex(WeatherClientError, "hindcasts"):
            self.make_client(opener).fetch_forecast("turbine_1", datetime(2024, 6, 1, tzinfo=UTC))
        self.assertEqual(opener.calls, [])

    def test_load_cached_forecast_rejects_modified_response_checksum(self):
        client = self.make_client(FakeOpener([BODY]))
        forecast = client.fetch_forecast("turbine_1", ISSUE)
        stem = forecast.metadata.input_hash
        raw = copy.deepcopy(HOURLY_RESPONSE)
        raw["hourly"]["wind_speed_100m"][0] = 40.0
        (self.cache_dir / f"{stem}.json").write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(WeatherClientError, "checksum"):
            client.load_cached_forecast(self.cache_dir / f"{stem}.meta.json")

    def test_load_cached_forecast_rejects_changed_run_source_and_coordinates(self):
        client = self.make_client(FakeOpener([BODY]))
        forecast = client.fetch_forecast("turbine_1", ISSUE)
        path = self.cache_dir / f"{forecast.metadata.input_hash}.meta.json"
        original = forecast.metadata.to_json()
        for changes in (
            {"weather_run_time": "2026-02-01T12:00:00Z"},
            {"turbine_id": "turbine_2"},
            {"api_url": "https://archive-api.open-meteo.com/v1/archive"},
            {"forecast_source": "weather_observation"},
            {"publication_safety_delay_seconds": 0},
        ):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**original, **changes}), encoding="utf-8")
                with self.assertRaises(WeatherClientError):
                    client.load_cached_forecast(path)

    def test_verified_legacy_cache_can_still_be_loaded(self):
        client = self.make_client(FakeOpener([BODY]))
        forecast = client.fetch_forecast("turbine_1", ISSUE)
        params = client._params(*client._coordinates("turbine_1"), forecast.metadata.weather_run_time)
        params.pop("temperature_unit")
        legacy_hash = client._input_hash("turbine_1", ISSUE, params)
        legacy = {key: value for key, value in forecast.metadata.to_json().items() if key in (
            "turbine_id", "issue_time", "weather_run_time", "provider", "model", "fetch_time", "input_hash"
        )}
        legacy["input_hash"] = legacy_hash
        (self.cache_dir / f"{legacy_hash}.json").write_bytes(BODY)
        path = self.cache_dir / f"{legacy_hash}.meta.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = client.load_cached_forecast(path)
        self.assertEqual(loaded.hourly, forecast.hourly)
        self.assertEqual(loaded.metadata.forecast_source, FORECAST_SOURCE)

    def test_rejects_nonfinite_weather_values(self):
        client = self.make_client(FakeOpener([]))
        for value in (float("nan"), float("inf"), -float("inf"), 10**400):
            with self.subTest(value=str(value)[:30]):
                raw = copy.deepcopy(HOURLY_RESPONSE)
                raw["hourly"]["temperature_2m"][0] = value
                with self.assertRaisesRegex(WeatherClientError, "non-finite"):
                    client._parse_response(json.dumps(raw).encode("utf-8"), select_safe_run(ISSUE))

    def test_rejects_duplicate_unsorted_nonhourly_and_wrong_run_times(self):
        client = self.make_client(FakeOpener([]))
        first, second = HOURLY_RESPONSE["hourly"]["time"]
        for times in (
            [first, first], [second, first], [first, second + 3600],
            [first + 1, second + 1], [first + 0.5, second + 0.5],
            [first - 86400 * 10, second - 86400 * 10],
            [first + 86400 * 10, second + 86400 * 10],
        ):
            with self.subTest(times=times):
                raw = copy.deepcopy(HOURLY_RESPONSE)
                raw["hourly"]["time"] = times
                with self.assertRaisesRegex(WeatherClientError, "valid times"):
                    client._parse_response(json.dumps(raw).encode("utf-8"), select_safe_run(ISSUE))

    def test_rejects_wrong_temperature_direction_pressure_and_time_units(self):
        client = self.make_client(FakeOpener([]))
        for variable, unit in (("temperature_2m", "F"), ("wind_direction_100m", "radian"),
                               ("surface_pressure", "Pa"), ("time", "iso8601")):
            with self.subTest(variable=variable):
                raw = copy.deepcopy(HOURLY_RESPONSE)
                raw["hourly_units"][variable] = unit
                with self.assertRaisesRegex(WeatherClientError, "did not return"):
                    client._parse_response(json.dumps(raw).encode("utf-8"), select_safe_run(ISSUE))

    def test_null_tail_is_preserved_without_imputation(self):
        raw = copy.deepcopy(HOURLY_RESPONSE)
        for variable in HOURLY_VARIABLES:
            raw["hourly"][variable][1] = None
        client = self.make_client(FakeOpener([]))
        _, rows = client._parse_response(json.dumps(raw).encode("utf-8"), select_safe_run(ISSUE))
        self.assertEqual(rows[0].wind_speed_100m, 7.1)
        self.assertIsNone(rows[1].wind_speed_100m)
        self.assertIsNone(rows[1].temperature_2m)

    def test_all_null_response_uses_one_cycle_fallback(self):
        raw = copy.deepcopy(HOURLY_RESPONSE)
        for variable in HOURLY_VARIABLES:
            raw["hourly"][variable] = [None, None]
        opener = FakeOpener([json.dumps(raw).encode("utf-8"), BODY])
        forecast = self.make_client(opener).fetch_forecast("turbine_1", ISSUE)
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(forecast.metadata.weather_run_time, select_safe_run(ISSUE) - timedelta(hours=6))


if __name__ == "__main__":
    unittest.main()
