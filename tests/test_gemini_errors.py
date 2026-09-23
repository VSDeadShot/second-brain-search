"""Reading a Gemini API error: which quota it names, and how long to wait.

Embedding and generation hit the same quota machinery, so these helpers must not
be embed-specific. The tests use an embedding quota id and a generation one and
expect identical handling.
"""

from __future__ import annotations

import pytest

from second_brain.gemini_errors import (
    RATE_LIMIT_MARGIN_SECONDS,
    RATE_WINDOW_SECONDS,
    daily_quota_limit,
    error_details,
    has_detail,
    is_unexplained_rate_limit,
    rate_limit_delay,
)

from fakes import (
    PER_DAY_QUOTA_ID,
    PER_MINUTE_QUOTA_ID,
    bare_rate_limit_error,
    daily_quota_error,
    rate_limit_error,
)

QUOTA_FAILURE = "type.googleapis.com/google.rpc.QuotaFailure"
RETRY_INFO = "type.googleapis.com/google.rpc.RetryInfo"
# Shaped like the embedding quota ids. Matching is on "PerDay", never on the
# method or model, so a generation quota has to read the same way.
GENERATE_PER_DAY_QUOTA_ID = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def client_error(code: int, body: dict) -> Exception:
    from google.genai import errors

    return errors.ClientError(code, body)


def test_error_details_returns_the_google_rpc_entries() -> None:
    kinds = [str(d["@type"]).rsplit(".", 1)[-1] for d in error_details(rate_limit_error("45s"))]

    assert kinds == ["QuotaFailure", "RetryInfo"]


def test_error_details_reads_a_body_that_is_not_wrapped_in_error() -> None:
    exc = client_error(429, {"code": 429, "details": [{"@type": RETRY_INFO}]})

    assert error_details(exc) == [{"@type": RETRY_INFO}]


def test_error_details_is_empty_for_an_exception_carrying_nothing() -> None:
    assert error_details(RuntimeError("boom")) == []


def test_error_details_skips_entries_that_are_not_objects() -> None:
    exc = client_error(429, {"error": {"details": ["not an object"]}})

    assert error_details(exc) == []


def test_has_detail_matches_on_the_type_suffix() -> None:
    exc = rate_limit_error("45s")

    assert has_detail(exc, "QuotaFailure")
    assert has_detail(exc, "RetryInfo")
    assert not has_detail(exc, "Help")


def test_has_detail_is_false_when_the_error_carries_no_details() -> None:
    assert not has_detail(RuntimeError("boom"), "QuotaFailure")


def test_daily_quota_limit_reports_the_limit_a_per_day_429_names() -> None:
    assert daily_quota_limit(daily_quota_error()) == "1000"


def test_a_generation_per_day_quota_reads_exactly_the_same() -> None:
    exc = rate_limit_error("58s", quota_id=GENERATE_PER_DAY_QUOTA_ID, quota_value="500")

    assert daily_quota_limit(exc) == "500"


def test_a_per_minute_429_is_not_a_daily_one_despite_its_retry_delay() -> None:
    exc = rate_limit_error("45s", quota_id=PER_MINUTE_QUOTA_ID)

    assert daily_quota_limit(exc) is None


def test_daily_quota_limit_falls_back_when_no_value_is_given() -> None:
    exc = client_error(
        429,
        {"error": {"details": [{"@type": QUOTA_FAILURE, "violations": [{"quotaId": PER_DAY_QUOTA_ID}]}]}},
    )

    assert daily_quota_limit(exc) == "the daily"


def test_daily_quota_limit_is_none_for_anything_but_a_429() -> None:
    exc = client_error(400, {"error": {"details": [{"@type": QUOTA_FAILURE, "violations": [{"quotaId": PER_DAY_QUOTA_ID}]}]}})

    assert daily_quota_limit(exc) is None


def test_the_bare_429_is_unexplained() -> None:
    assert is_unexplained_rate_limit(bare_rate_limit_error())


def test_a_429_that_names_its_quota_is_explained() -> None:
    assert not is_unexplained_rate_limit(rate_limit_error("45s"))


def test_a_429_carrying_only_a_retry_delay_is_explained() -> None:
    exc = client_error(429, {"error": {"details": [{"@type": RETRY_INFO, "retryDelay": "30s"}]}})

    assert not is_unexplained_rate_limit(exc)


def test_a_non_429_is_never_an_unexplained_rate_limit() -> None:
    assert not is_unexplained_rate_limit(RuntimeError("boom"))


def test_rate_limit_delay_reads_the_servers_retry_info() -> None:
    assert rate_limit_delay(rate_limit_error("45s")) == 45.0


def test_rate_limit_delay_keeps_a_fractional_delay() -> None:
    assert rate_limit_delay(rate_limit_error("45.676528856s")) == pytest.approx(45.676528856)


def test_a_429_without_retry_info_waits_a_full_window() -> None:
    assert rate_limit_delay(rate_limit_error(None)) == RATE_WINDOW_SECONDS


def test_an_unreadable_retry_delay_waits_a_full_window() -> None:
    assert rate_limit_delay(rate_limit_error("soon")) == RATE_WINDOW_SECONDS


def test_rate_limit_delay_is_none_for_anything_but_a_429() -> None:
    assert rate_limit_delay(RuntimeError("boom")) is None


def test_the_window_and_margin_are_the_measured_ones() -> None:
    assert RATE_WINDOW_SECONDS == 60
    assert RATE_LIMIT_MARGIN_SECONDS == 1.0
