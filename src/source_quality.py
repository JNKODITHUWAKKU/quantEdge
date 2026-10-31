"""Recompute the original price-quality screens from the nine frozen CSV inputs.

The currency label is archived provider metadata, not independent verification.
Whole-history integrity screens retain flagged observations without changing prices.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit import Issue, iso_date, local_dates, prepare_rate_proxy
from src.common import session_calendar
from src.settings import SOURCE_SETTINGS

AUDIT_RULES = {
    "large_absolute_simple_return": 0.10,
    "flat_return_run": 5,
    "relative_adjustment_jump": 0.02,
}
REQUIRED_EQUITY_COLUMNS = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
}
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")
NONNEGATIVE_ACTION_COLUMNS = ("Volume", "Dividends", "Stock Splits")
OHLC_TOLERANCE = 1e-8
DIVIDEND_RECONCILIATION_TOLERANCE = 0.005


def returns_on_calendar(prices: pd.DataFrame | pd.Series) -> tuple:
    """Never turn a missing session into zero or bridge it as a one-day return."""
    simple = prices.pct_change(fill_method=None)
    log = np.log(prices).diff()
    return simple, log


def audit_equity(
    frame: pd.DataFrame,
    metadata: dict,
    symbol: str,
    sessions: pd.DatetimeIndex,
    cfg: dict,
) -> tuple[pd.Series | None, list[Issue], dict]:
    issues: list[Issue] = []

    def flag(level: str, check: str, date: object, detail: str) -> None:
        issues.append(Issue(level, symbol, check, str(date), detail))

    summary = {
        "symbol": symbol,
        "rows_returned": len(frame),
        "currency": metadata.get("currency"),
    }
    missing_columns = sorted(REQUIRED_EQUITY_COLUMNS - set(frame.columns))
    if missing_columns:
        flag("ERROR", "schema", "", f"Missing fields: {missing_columns}")
        return None, issues, summary
    if frame.empty:
        flag("ERROR", "empty_history", "", "No rows returned.")
        return None, issues, summary
    if metadata.get("currency") != cfg["data"]["currency"]:
        flag(
            "ERROR",
            "currency",
            "",
            "Expected independently returned USD metadata; absent or different.",
        )

    dates = _parse_and_validate_dates(frame["Date"], flag)
    if dates is None:
        return None, issues, summary

    data = frame.drop(columns="Date").copy()
    data.index = dates
    data = data.sort_index()
    _check_numeric_and_price_columns(data, sessions, flag)

    good = data["Adj Close"].where(
        np.isfinite(data["Adj Close"]) & data["Adj Close"].gt(0)
    )
    if good.dropna().empty:
        flag("ERROR", "no_valid_prices", "", "No valid adjusted closes.")
        return None, issues, summary

    expected = _check_session_calendar_coverage(good, data.index, sessions, summary, flag)
    _check_ohlc_and_actions(data, flag)
    calendar_prices = good.reindex(expected)
    _screen_return_and_adjustment_anomalies(data, calendar_prices, cfg["audit"], summary, flag)
    return calendar_prices.rename(symbol), issues, summary


def _parse_and_validate_dates(
    raw_dates: pd.Series,
    flag: Callable[[str, str, object, str], None],
) -> pd.DatetimeIndex | None:
    try:
        dates = local_dates(raw_dates)
    except (ValueError, TypeError, OverflowError) as exc:
        flag("ERROR", "date_parse", "", str(exc))
        return None
    if dates.isna().any():
        flag("ERROR", "missing_date", "", "A row has no valid session date.")
        return None
    if dates.duplicated().any():
        for date in dates[dates.duplicated(keep=False)].unique():
            flag(
                "ERROR",
                "duplicate_session",
                iso_date(date),
                "Duplicate session; no automatic deduplication.",
            )
        return None
    if not dates.is_monotonic_increasing:
        flag(
            "WARNING",
            "row_order",
            "",
            "Derived view sorted by date; raw file retained unchanged.",
        )
    return dates


def _check_numeric_and_price_columns(
    data: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    flag: Callable[[str, str, object, str], None],
) -> None:
    for column in sorted(REQUIRED_EQUITY_COLUMNS - {"Date"}):
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        for date in data.index[original.notna() & numeric.isna()]:
            flag("ERROR", "non_numeric", iso_date(date), f"Non-numeric {column}.")
        data[column] = numeric
    for column in PRICE_COLUMNS:
        invalid = ~np.isfinite(data[column]) | data[column].le(0)
        for date in data.index[invalid]:
            flag(
                "ERROR",
                "invalid_price",
                iso_date(date),
                f"{column} must be finite and positive.",
            )
    for date in data.index.difference(sessions):
        flag(
            "ERROR",
            "unexpected_session",
            iso_date(date),
            "Outside requested XNYS core-session calendar.",
        )


def _check_session_calendar_coverage(
    good_prices: pd.Series,
    observed_dates: pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
    summary: dict,
    flag: Callable[[str, str, object, str], None],
) -> pd.DatetimeIndex:
    first = good_prices.first_valid_index()
    last = good_prices.last_valid_index()
    expected = sessions[sessions >= first]
    for date in expected.difference(observed_dates):
        flag(
            "ERROR",
            "missing_session",
            iso_date(date),
            "Missing ETF quote; no price forward-fill.",
        )
    if last != sessions[-1]:
        flag(
            "ERROR",
            "stale_end",
            iso_date(last),
            f"Last valid quote precedes required {iso_date(sessions[-1])}.",
        )
    # No pre-inception padding: earliest returned date is not independently verified inception.
    summary.update(
        {
            "first_valid_session": iso_date(first),
            "last_valid_session": iso_date(last),
            "expected_sessions_from_first_quote": len(expected),
            "missing_sessions": len(expected.difference(observed_dates)),
            "duplicate_rows": 0,
        }
    )
    return expected


def _check_ohlc_and_actions(
    data: pd.DataFrame,
    flag: Callable[[str, str, object, str], None],
) -> None:
    invalid_range = (
        data["High"] < data[["Open", "Close", "Low"]].max(axis=1) - OHLC_TOLERANCE
    ) | (data["Low"] > data[["Open", "Close", "High"]].min(axis=1) + OHLC_TOLERANCE)
    for date in data.index[invalid_range]:
        flag(
            "ERROR",
            "ohlc_order",
            iso_date(date),
            "OHLC range is internally inconsistent.",
        )
    for field in NONNEGATIVE_ACTION_COLUMNS:
        invalid = ~np.isfinite(data[field]) | data[field].lt(0)
        for date in data.index[invalid]:
            flag("ERROR", "invalid_action_or_volume", iso_date(date), f"Invalid {field}.")
    for date in data.index[data["Volume"].eq(0)]:
        flag(
            "WARNING",
            "zero_volume",
            iso_date(date),
            "Zero reported volume; investigate quote quality.",
        )


def _screen_return_and_adjustment_anomalies(
    data: pd.DataFrame,
    calendar_prices: pd.Series,
    audit_cfg: dict,
    summary: dict,
    flag: Callable[[str, str, object, str], None],
) -> None:
    simple, _ = returns_on_calendar(calendar_prices)
    threshold = audit_cfg["large_absolute_simple_return"]
    for date, value in simple[simple.abs() >= threshold].items():
        flag(
            "WARNING",
            "large_return",
            iso_date(date),
            f"Simple return {value:.8f}; retained, not winsorised.",
        )
    flat = simple.eq(0)
    runs = flat.groupby((~flat).cumsum()).cumsum()
    for date in runs.index[runs.eq(audit_cfg["flat_return_run"])]:
        flag(
            "WARNING",
            "flat_run",
            iso_date(date),
            "Run of unchanged adjusted prices reaches screening length.",
        )
    factor = data["Adj Close"] / data["Close"]
    jumps = factor.pct_change(fill_method=None)
    for date in jumps.index[jumps.abs().ge(audit_cfg["relative_adjustment_jump"])]:
        flag(
            "WARNING",
            "adjustment_jump",
            iso_date(date),
            f"Adjustment factor changes by {jumps.loc[date]:.8f}; reconcile distributions/splits.",
        )
    # Approximate same-day dividend reconciliation. Yahoo Close is normally split-adjusted.
    # Do not multiply prices by splits again. Complex action days require human review.
    previous_close = data["Close"].shift(1)
    dividends = data["Dividends"]
    expected_factor = previous_close / (previous_close - dividends)
    observed_factor = factor / factor.shift(1)
    action = dividends.gt(0) & data["Stock Splits"].eq(0) & previous_close.gt(dividends)
    mismatch = action & (
        (observed_factor / expected_factor - 1).abs() > DIVIDEND_RECONCILIATION_TOLERANCE
    )
    for date in data.index[mismatch]:
        flag(
            "WARNING",
            "dividend_reconciliation",
            iso_date(date),
            "Adjustment-factor/dividend approximation differs by >0.5%; original values retained.",
        )
    summary.update(
        {
            "dividend_events": int(dividends.gt(0).sum()),
            "split_events": int(data["Stock Splits"].gt(0).sum()),
            "large_negative_return_flags": int(simple.le(-threshold).sum()),
            "large_absolute_return_flags": int(simple.abs().ge(threshold).sum()),
            "zero_volume_rows": int(data["Volume"].eq(0).sum()),
        }
    )


DEFAULT_EQUITY_METADATA = {
    "currency": "USD",
    "scope": "Archived same-provider metadata; not independently verified",
}


def audit_local_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed issues and inventory; never hard-code an observed count."""
    cfg = deepcopy(SOURCE_SETTINGS)
    cfg["audit"] = AUDIT_RULES.copy()
    sessions = session_calendar(
        cfg["data"]["start"], cfg["data"]["cutoff"], cfg["data"]["calendar"]
    )
    issues, profiles = [], []
    for symbol in cfg["data"]["tickers"]:
        frame = pd.read_csv(root / "data" / f"{symbol}.csv")
        _, found, profile = audit_equity(
            frame, dict(DEFAULT_EQUITY_METADATA), symbol, sessions, cfg
        )
        issues.extend(found)
        profiles.append(profile)
    _, rate_issues = prepare_rate_proxy(
        pd.read_csv(root / "data/DGS3MO.csv"), sessions, cfg
    )
    issues.extend(rate_issues)
    return pd.DataFrame(
        [asdict(item) for item in issues], columns=list(Issue.__dataclass_fields__)
    ), pd.DataFrame(profiles)
