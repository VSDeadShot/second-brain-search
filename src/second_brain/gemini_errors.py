"""Reading a Gemini API error - shared by embedding and generation.

google-genai raises ClientError with `.code` and the response JSON in
`.details`; everything useful sits in the google.rpc detail entries. Nothing
here is specific to one API method: the same three 429s were seen embedding and
have to be told apart the same way wherever they land.

The taxonomy was measured against the live free tier in September 2026:
- a per-day 429 carries a QuotaFailure whose quotaId contains "PerDay". It also
  carries a RetryInfo delay, so only the quotaId separates it from a per-minute
  one - and waiting out a daily quota cannot help.
- a per-minute 429 names its quota and says how long to wait.
- a bare 429 carries neither: no quota named, no delay. Waiting didn't clear it.
"""

from __future__ import annotations

# One quota window. Also the fallback wait for a 429 that gives no delay.
RATE_WINDOW_SECONDS = 60
# Added to the server's retryDelay so the retry lands just after the window
# clears rather than racing its boundary.
RATE_LIMIT_MARGIN_SECONDS = 1.0


def error_details(exc: Exception) -> list[dict]:
    """The google.rpc detail entries of a google-genai ClientError, if any."""
    body = getattr(exc, "details", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    details = error.get("details", []) if isinstance(error, dict) else []
    return [d for d in details if isinstance(d, dict)]


def has_detail(exc: Exception, type_suffix: str) -> bool:
    return any(str(d.get("@type", "")).endswith(type_suffix) for d in error_details(exc))


def daily_quota_limit(exc: Exception) -> str | None:
    """The daily limit (e.g. "1000") if `exc` is a per-day quota 429, else None.

    The live per-day 429 still carried a ~58s RetryInfo delay, so the retry
    delay cannot tell the two apart - the QuotaFailure quotaId can
    ("...PerDay..." vs "...PerMinute...").
    """
    if getattr(exc, "code", None) != 429:
        return None
    for detail in error_details(exc):
        if not str(detail.get("@type", "")).endswith("QuotaFailure"):
            continue
        for violation in detail.get("violations", []):
            if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                return str(violation.get("quotaValue") or "the daily")
    return None


def is_unexplained_rate_limit(exc: Exception) -> bool:
    """A 429 with neither a QuotaFailure nor a RetryInfo - the bare 429 seen live."""
    return (
        getattr(exc, "code", None) == 429
        and not has_detail(exc, "QuotaFailure")
        and not has_detail(exc, "RetryInfo")
    )


def rate_limit_delay(exc: Exception) -> float | None:
    """Seconds the server asked us to wait, if `exc` is a 429; otherwise None.

    The delay lives in a google.rpc.RetryInfo entry as e.g. "45s". A 429 without
    a parseable RetryInfo waits a full window.
    """
    if getattr(exc, "code", None) != 429:
        return None
    for detail in error_details(exc):
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).strip().removesuffix("s")
            try:
                return float(raw)
            except ValueError:
                break
    return float(RATE_WINDOW_SECONDS)
