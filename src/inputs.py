"""Load the original CSV inputs using explicit Python research settings."""

from __future__ import annotations

import hashlib
import importlib.metadata
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit import find_fred_date_column, local_dates, prepare_rate_proxy
from src.common import session_calendar
from src.settings import (
    ORIGINAL_FINAL_LOCK,
    ORIGINAL_RECOVERY_DATE,
    ORIGINAL_SELECTION_LOCK,
    SOURCE_SETTINGS,
    VALIDATION_SETTINGS,
    final_settings,
)
from src.source_quality import audit_local_sources
from src.storage import verify_inputs

EXPECTED_FINAL_BOUNDS = ("2020-01-01", "2026-10-23")
EXPECTED_CUTOFF_DATE = pd.Timestamp("2026-10-23")
EXPECTED_SELECTION = ("sample_gmv", "garch_t")
CPI_PERIOD_BOUNDS = ("2019-12", "2026-09")
EXPECTED_MISSING_CPI_MONTHS = ["2025-10"]
SHARED_RUN_FIELDS = (
    "strategies",
    "lookback_returns",
    "confidence_levels",
    "cost_scenarios_bps",
    "primary_cost_bps",
)
EXCLUDED_INTERFACE_MODULES = {"dashboard.py", "labels.py", "methodology.py"}
PINNED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "arch",
    "statsmodels",
    "scikit-learn",
    "exchange-calendars",
)
LOCK_HEX_LEN = 16


def settings(root: Path) -> dict:
    verify_inputs(root)
    cfg = final_settings()
    if (cfg["run"]["start"], cfg["run"]["end"]) != EXPECTED_FINAL_BOUNDS:
        raise ValueError("Final boundaries changed.")
    for field in SHARED_RUN_FIELDS:
        if cfg["run"][field] != VALIDATION_SETTINGS["run"][field]:
            raise ValueError(f"Original research convention changed: {field}")
    if (cfg["assessment"]["portfolio"], cfg["assessment"]["monitor"]) != EXPECTED_SELECTION:
        raise ValueError("No final-period model reselection is permitted.")
    return cfg


def load_cpi(root: Path, cfg: dict) -> pd.Series:
    frame = pd.read_csv(root / cfg["funding"]["cpi_file"])
    column = find_fred_date_column(frame)
    if column is None:
        raise ValueError("CPI date column is missing.")
    series = pd.Series(
        frame.CPIAUCNS.to_numpy(),
        index=pd.PeriodIndex(pd.to_datetime(frame[column]), freq="M"),
    )
    expected_index = pd.period_range(*CPI_PERIOD_BOUNDS, freq="M")
    missing_months = series[series.isna()].index.astype(str).tolist()
    if not series.index.equals(expected_index) or missing_months != EXPECTED_MISSING_CPI_MONTHS:
        raise ValueError("The original CPI coverage or missing-month convention changed.")
    return series


def load_final(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Recreate the same complete session panel and lagged rate proxy as before."""
    verify_inputs(root)
    source = deepcopy(SOURCE_SETTINGS)
    source_issues, source_inventory = audit_local_sources(root)
    if source_issues.severity.eq("ERROR").any():
        raise ValueError("Reproducible source audit found blocking input errors.")
    end = pd.Timestamp(cfg["run"]["end"])
    if end != EXPECTED_CUTOFF_DATE:
        raise ValueError("The fixed source cutoff changed.")
    sessions = session_calendar(
        source["data"]["start"], str(end.date()), source["data"]["calendar"]
    )
    prices = _load_aligned_prices(root, source["data"]["tickers"], sessions, end)
    rate = _load_aligned_rate_proxy(root, sessions, prices.index, source, end)
    return (
        prices,
        rate,
        {
            "warning_count": int(source_issues.severity.eq("WARNING").sum()),
            "issues": source_issues,
            "inventory": source_inventory,
            "scope": (
                "Recomputed from frozen CSV inputs; provider metadata is archived, "
                "not independently verified"
            ),
        },
    )


def _load_aligned_prices(
    root: Path,
    tickers: list[str],
    sessions: pd.DatetimeIndex,
    end: pd.Timestamp,
) -> pd.DataFrame:
    series_list = []
    for ticker in tickers:
        frame = pd.read_csv(root / "data" / f"{ticker}.csv")
        dates = local_dates(frame.Date)
        if dates.has_duplicates:
            raise ValueError(f"Duplicate source dates: {ticker}")
        series_list.append(
            pd.Series(frame["Adj Close"].to_numpy(), index=dates, name=ticker)
            .sort_index()
            .loc[:end]
        )
    first_session = max(values.first_valid_index() for values in series_list)
    # Trim to the common inception date across all 7 ETFs (GLD starts Nov 2004).
    prices = pd.concat(series_list, axis=1).reindex(sessions[sessions >= first_session])
    if (
        prices.isna().any().any()
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise ValueError("Source price panel must be complete, finite and positive.")
    return prices


def _load_aligned_rate_proxy(
    root: Path,
    sessions: pd.DatetimeIndex,
    price_dates: pd.DatetimeIndex,
    source: dict,
    end: pd.Timestamp,
) -> pd.Series:
    fred = pd.read_csv(root / "data/DGS3MO.csv")
    column = find_fred_date_column(fred)
    if column is None:
        raise ValueError("FRED date column is missing.")
    fred = fred[pd.to_datetime(fred[column]) <= end]
    rates, issues = prepare_rate_proxy(fred, sessions, source)
    if issues:
        raise ValueError("The lagged rate proxy has blocking source issues.")
    rate = rates.set_index("return_date").risk_free_simple_return.reindex(price_dates[1:])
    if rate.isna().any():
        raise ValueError("The risk-free proxy is incomplete.")
    return rate


def research_identity(root: Path) -> dict:
    """Bind numerical results to source, CSV inputs and numerical package versions."""
    settings(root)
    digest = hashlib.sha256()
    for path in sorted((root / "src").glob("*.py")):
        if path.name not in EXCLUDED_INTERFACE_MODULES:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    for path in sorted((root / "data").glob("*.csv")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in sorted((root / "evidence").iterdir()):
        if path.name.startswith("."):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for package in PINNED_PACKAGES:
        digest.update(f"{package}=={importlib.metadata.version(package)}".encode())
    return {
        "lock_id": "research_" + digest.hexdigest()[:LOCK_HEX_LEN],
        "source_recovery_contract_at_utc": ORIGINAL_RECOVERY_DATE,
        "original_final_lock_id": ORIGINAL_FINAL_LOCK,
        "identity": {"selection_lock_id": ORIGINAL_SELECTION_LOCK},
    }
