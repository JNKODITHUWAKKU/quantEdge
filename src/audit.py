"""Reproducible quality checks; warnings retain observations, errors block use.

This is a data-readiness audit, not a portfolio backtest. Full-history integrity
flags are recorded; distribution diagnostics are confined to development data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FRED_DATE_COLUMNS = ("observation_date", "DATE", "Date")
FRED_MISSING_TOKENS = (".", "")
MIN_AVAILABILITY_LAG = 1
PERCENT_DIVISOR = 100
ACT_365_DAYS = 365


@dataclass
class Issue:
    severity: str
    symbol: str
    check: str
    date: str
    detail: str


def iso_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def local_dates(values: pd.Series) -> pd.DatetimeIndex:
    # Keep the provider's exchange-local session date, not its UTC calendar date.
    return pd.DatetimeIndex([pd.Timestamp(item).date() for item in values], name="Date")


def find_fred_date_column(frame: pd.DataFrame) -> str | None:
    return next((col for col in FRED_DATE_COLUMNS if col in frame), None)


def prepare_rate_proxy(
    frame: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: dict
) -> tuple[pd.DataFrame, list[Issue]]:
    """Conservative, explicitly assumed availability and ACT/365 cash accrual.

    This is not point-in-time vintage data or an exact Treasury holding-period return.
    The two-session availability lag is managerial, not a claim about every release.
    """
    series_name = cfg["conventions"]["rate_series"]
    rates = _parse_fred_rates(frame, series_name)
    lag = cfg["conventions"]["rate_assumed_availability_lag_sessions"]
    if lag < MIN_AVAILABILITY_LAG:
        raise ValueError("Availability lag must be at least one session.")
    aligned_rates = _align_rates_to_sessions(rates, sessions, lag)
    merged = _compute_accrued_returns(
        aligned_rates, sessions, cfg["conventions"]["rate_max_age_calendar_days"]
    )
    issues = _collect_stale_rate_issues(merged, series_name)
    return merged, issues


def _parse_fred_rates(frame: pd.DataFrame, series_name: str) -> pd.DataFrame:
    date_col = find_fred_date_column(frame)
    if date_col is None or series_name not in frame:
        raise ValueError("FRED date or series field is absent.")
    rate_dates = pd.DatetimeIndex(pd.to_datetime(frame[date_col], errors="raise"))
    if rate_dates.has_duplicates or rate_dates.hasnans:
        raise ValueError("FRED dates are duplicated or missing.")
    values = pd.to_numeric(frame[series_name], errors="coerce")
    # FRED uses '.' for holiday/missing observations; reject any other non-numeric token.
    allowed_missing = frame[series_name].isna() | frame[series_name].astype(str).isin(
        FRED_MISSING_TOKENS
    )
    if (values.isna() & ~allowed_missing).any() or np.isinf(values).any():
        raise ValueError("FRED contains an unexpected non-numeric or infinite yield.")
    rates = pd.DataFrame({"observation_date": rate_dates, "yield_percent": values}).dropna()
    return rates.sort_values("observation_date")


def _align_rates_to_sessions(
    rates: pd.DataFrame, sessions: pd.DatetimeIndex, lag: int
) -> pd.DataFrame:
    positions = sessions.searchsorted(rates["observation_date"], side="right") + lag - 1
    valid = positions < len(sessions)
    aligned = rates.loc[valid].copy()
    aligned["assumed_available_session"] = sessions[positions[valid]].to_numpy()
    # Multiple non-session FRED dates can map to the same available session.
    return aligned.sort_values(["assumed_available_session", "observation_date"])


def _compute_accrued_returns(
    rates: pd.DataFrame, sessions: pd.DatetimeIndex, max_age_days: int
) -> pd.DataFrame:
    holding = pd.DataFrame({"return_date": sessions[1:], "holding_start": sessions[:-1]})
    merged = pd.merge_asof(
        holding,
        rates,
        left_on="holding_start",
        right_on="assumed_available_session",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["calendar_days"] = (merged["return_date"] - merged["holding_start"]).dt.days
    merged["quote_age_days"] = (merged["holding_start"] - merged["observation_date"]).dt.days
    stale = merged["quote_age_days"] > max_age_days
    merged["risk_free_simple_return"] = (
        merged["yield_percent"] / PERCENT_DIVISOR * merged["calendar_days"] / ACT_365_DAYS
    )
    merged.loc[stale, "risk_free_simple_return"] = np.nan
    return merged


def _collect_stale_rate_issues(merged: pd.DataFrame, series_name: str) -> list[Issue]:
    stale = merged["risk_free_simple_return"].isna() & merged["quote_age_days"].notna()
    return [
        Issue(
            "ERROR",
            series_name,
            "stale_rate",
            iso_date(date),
            "Cash-rate observation exceeds age limit.",
        )
        for date in merged.loc[stale, "return_date"]
    ]
