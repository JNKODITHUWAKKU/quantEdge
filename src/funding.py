"""Ex-post real wealth and path capacity; not a forward spending forecast."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DEFAULT_INITIAL_CAPITAL = 1e7
DEFAULT_BASE_MONTH = "2019-12"
DEFAULT_LAST_MONTH = "2026-09"
DEFAULT_COST_BPS = 5.0
BPS_DIVISOR = 10_000
DECEMBER_MONTH = 12
MAX_MONTH_END_GAP_DAYS = 4
DAYS_PER_YEAR = 365.25
EVIDENCE_CLASS = "ex_post_funding_diagnostic"


def real_funding(
    daily: pd.DataFrame,
    cpi: pd.Series,
    initial: float = DEFAULT_INITIAL_CAPITAL,
    base_month: str = DEFAULT_BASE_MONTH,
    last_month: str = DEFAULT_LAST_MONTH,
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[dict, pd.DataFrame]:
    cost_rate = cost_bps / BPS_DIVISOR
    nav = _validate_funding_inputs(daily, cpi, initial, cost_rate)
    periods = pd.period_range(base_month, last_month, freq="M")
    levels = _align_cpi_levels(cpi, periods)
    dates, growth = _align_monthly_wealth(nav, periods, initial)

    inflation = levels.to_numpy() / float(levels.iloc[0])
    payment = np.array([period.month == DECEMBER_MONTH and period > periods[0] for period in periods])
    coeff = _compute_withdrawal_coefficients(growth, inflation, payment, cost_rate)

    surplus = initial * (growth[-1] - inflation[-1])
    algebraic = surplus / coeff[-1]
    budget = max(0.0, float(algebraic))
    path = initial * growth - budget * coeff
    solvent = bool((path > 0).all())
    feasible = bool(surplus >= 0 and solvent)
    real = initial * growth / inflation
    all_dates = [periods[0].end_time.normalize(), *dates.to_list()]

    frame = pd.DataFrame(
        {
            "reference_month": periods.astype(str),
            "valuation_date": all_dates,
            "cpi": levels.to_numpy(),
            "cpi_ratio": inflation,
            "nominal_no_spending_usd": initial * growth,
            "real_no_spending_base_usd": real,
            "year_end_payment": payment,
            "withdrawal_terminal_coefficient": coeff,
            "algebraic_capacity_path_usd": path,
            "cpi_status": np.where(levels.isna(), "MISSING_NO_INTERPOLATION", "OBSERVED"),
            "evidence_class": EVIDENCE_CLASS,
        }
    )
    years = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days / DAYS_PER_YEAR
    result = {
        "base_month": base_month,
        "last_cpi_month": last_month,
        "last_valuation_date": str(pd.Timestamp(all_dates[-1]).date()),
        "initial_capital_usd": initial,
        "terminal_nominal_no_spending_usd": float(initial * growth[-1]),
        "terminal_real_no_spending_usd": float(real[-1]),
        "cumulative_inflation": float(inflation[-1] - 1),
        "real_total_return": float(real[-1] / initial - 1),
        "real_cagr": float((real[-1] / initial) ** (1 / years) - 1),
        "real_shortfall_usd": float(max(0, initial - real[-1])),
        "preserved_real_principal_before_spending": bool(surplus >= 0),
        "algebraic_annual_real_capacity_usd": float(algebraic),
        "feasible_nonnegative_annual_real_budget_usd": budget if feasible else None,
        "feasible_budget_fraction_of_initial": budget / initial if feasible else None,
        "completed_annual_payments": int(payment.sum()),
        "capacity_path_solvent": solvent,
        "cost_bps": cost_bps,
        "missing_nonpayment_cpi_months": periods[levels.isna()].astype(str).tolist(),
        "capacity_status": "NONNEGATIVE_PATH_CAPACITY"
        if feasible
        else "NO_NONNEGATIVE_BUDGET_PRESERVES_REAL_PRINCIPAL",
        "evidence_class": EVIDENCE_CLASS,
        "interpretation": (
            "Historical path capacity, not sustainable forward spending; "
            "no client payout rate assumed."
        ),
    }
    return result, frame


def _validate_funding_inputs(
    daily: pd.DataFrame, cpi: pd.Series, initial: float, cost_rate: float
) -> pd.Series:
    if not 0 <= cost_rate < 1 or initial <= 0 or daily.empty:
        raise ValueError("Invalid capital/cost/history.")
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and ordered.")
    nav = daily.nav_usd.astype(float)
    if not np.isfinite(nav).all() or (nav <= 0).any():
        raise ValueError("Invalid wealth.")
    if (
        not isinstance(cpi.index, pd.PeriodIndex)
        or cpi.index.freqstr != "M"
        or cpi.index.has_duplicates
    ):
        raise ValueError("Unique monthly CPI required.")
    return nav


def _align_cpi_levels(cpi: pd.Series, periods: pd.PeriodIndex) -> pd.Series:
    levels = cpi.reindex(periods)
    # Base month, terminal month, and December payment months must have observed CPI; non-payment gaps (Oct 2025) stay NaN.
    required = np.array(
        [
            idx == 0 or idx == len(periods) - 1 or period.month == DECEMBER_MONTH
            for idx, period in enumerate(periods)
        ]
    )
    if levels[required].isna().any() or np.isinf(levels).any() or (levels.dropna() <= 0).any():
        raise ValueError("Missing endpoint/payment CPI or invalid level.")
    return levels


def _align_monthly_wealth(
    nav: pd.Series, periods: pd.PeriodIndex, initial: float
) -> tuple[pd.Series, np.ndarray]:
    month_nav = nav.groupby(nav.index.to_period("M")).last()
    needed = periods[1:]
    if not needed.isin(month_nav.index).all():
        raise ValueError("Missing wealth month.")
    dates = nav.index.to_series().groupby(nav.index.to_period("M")).last().reindex(needed)
    for month, date in dates.items():
        if date.date() < month.end_time.date() - dt.timedelta(days=MAX_MONTH_END_GAP_DAYS):
            raise ValueError("Incomplete reference month.")
    growth = np.r_[1.0, month_nav.reindex(needed).to_numpy() / initial]
    return dates, growth


def _compute_withdrawal_coefficients(
    growth: np.ndarray,
    inflation: np.ndarray,
    payment: np.ndarray,
    cost_rate: float,
) -> np.ndarray:
    coeff = np.zeros(len(growth))
    for idx in range(1, len(growth)):
        coeff[idx] = coeff[idx - 1] * growth[idx] / growth[idx - 1]
        if payment[idx]:
            coeff[idx] += inflation[idx] / (1 - cost_rate)
    if coeff[-1] <= 0:
        raise ValueError("No completed annual payment date.")
    return coeff
