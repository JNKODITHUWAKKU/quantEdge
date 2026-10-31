"""Self-financing portfolio accounting with delayed, recorded execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.portfolio import target_for_strategy

WEIGHT_SUM_ATOL = 1e-10
COST_RESIDUAL_TOLERANCE = 1e-12
BRENTQ_XTOL = 1e-15
MIN_LOOKBACK = 3
BPS_DIVISOR = 10_000
DEFAULT_LOOKBACK = 1260
DEFAULT_FREQUENCY = "quarterly"
DEFAULT_COST_BPS = 5.0
DEFAULT_INITIAL_VALUE = 10_000_000.0
DEFAULT_ANNUALIZATION = 252
RECONCILIATION_BASE_TOLERANCE = 1e-6
SUPPORTED_FREQUENCIES = {"quarterly": "Q", "monthly": "M"}


def rebalance_cost(
    pre_weights: np.ndarray, target_weights: np.ndarray, rate: float
) -> dict:
    """Solve c = rate * sum(abs((1-c)*target - pre)) exactly.

    c is cost as a fraction of pre-trade wealth. pre sums to 1 for a rebalance
    or 0 for the initial purchase from cash. Both purchases and sales cost money.
    """
    pre, target = np.asarray(pre_weights, float), np.asarray(target_weights, float)
    _validate_rebalance_inputs(pre, target, rate)

    def equation(cost: float) -> float:
        return float(cost - rate * np.abs((1 - cost) * target - pre).sum())

    # Post-trade target dollars depend on the fee itself, so solve the fixed point on [0, 1].
    cost = 0.0 if rate == 0 else float(brentq(equation, 0.0, 1.0, xtol=BRENTQ_XTOL))
    trades = (1 - cost) * target - pre
    gross = float(np.abs(trades).sum())
    residual = abs(cost - rate * gross)
    if residual > COST_RESIDUAL_TOLERANCE:
        raise RuntimeError("Transaction-cost reconciliation failed.")
    return {
        "cost_fraction": cost,
        "gross_traded_fraction": gross,
        "trade_fractions": trades,
        "cost_equation_residual": residual,
    }


def _validate_rebalance_inputs(
    pre: np.ndarray, target: np.ndarray, rate: float
) -> None:
    if (
        pre.shape != target.shape
        or not np.isfinite(pre).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Invalid weight vectors.")
    if (
        (pre < 0).any()
        or (target < 0).any()
        or not np.isclose(target.sum(), 1, atol=WEIGHT_SUM_ATOL, rtol=0)
    ):
        raise ValueError("Targets must be long-only and sum to one.")
    if not (
        np.isclose(pre.sum(), 1, atol=WEIGHT_SUM_ATOL, rtol=0)
        or np.isclose(pre.sum(), 0, atol=WEIGHT_SUM_ATOL, rtol=0)
    ):
        raise ValueError(
            "Pre-trade positions must be fully invested or entirely initial cash."
        )
    if not 0 <= rate < 1:
        raise ValueError("Proportional cost must be in [0, 1).")


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    initial_execution_date: pd.Timestamp
    initial_value: float


def run_portfolio(
    prices: pd.DataFrame,
    strategy: str,
    lookback: int = DEFAULT_LOOKBACK,
    frequency: str = DEFAULT_FREQUENCY,
    cost_bps: float = DEFAULT_COST_BPS,
    initial_value: float = DEFAULT_INITIAL_VALUE,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> BacktestResult:
    rate = _validate_backtest_inputs(prices, lookback, frequency, cost_bps, initial_value)
    dates, assets = prices.index, list(prices.columns)
    returns = prices.pct_change(fill_method=None)
    scheduled = _build_rebalance_schedule(dates, lookback, frequency)
    signal_rows, trade_rows, daily_rows = [], [], []

    def decide(origin: int, execution: int) -> np.ndarray:
        history = returns.iloc[origin - lookback + 1 : origin + 1]
        if len(history) != lookback or history.index[-1] >= dates[execution]:
            raise RuntimeError("Forecast information/execution ordering violated.")
        target, details = target_for_strategy(history, strategy, annualization)
        signal_rows.append(
            {
                "strategy": strategy,
                "signal_date": dates[origin],
                "training_start": history.index[0],
                "training_end": history.index[-1],
                "execution_date": dates[execution],
                "first_affected_return": dates[execution + 1],
                "training_observations": len(history),
                **details,
                **{f"weight_{asset}": float(weight) for asset, weight in zip(assets, target)},
            }
        )
        return target

    execution = lookback + 1
    weights = decide(lookback, execution)
    initial = rebalance_cost(np.zeros(len(assets)), weights, rate)
    entry_dollars = initial_value * initial["cost_fraction"]
    value = initial_value - entry_dollars
    previous_reported_value = initial_value
    trade_rows.append(
        {
            "date": dates[execution],
            "kind": "initial_purchase",
            "cost_dollars": entry_dollars,
            "gross_traded_dollars": initial_value * initial["gross_traded_fraction"],
            "gross_traded_fraction": initial["gross_traded_fraction"],
            "cost_equation_residual": initial["cost_equation_residual"],
        }
    )
    max_reconciliation_error = RECONCILIATION_BASE_TOLERANCE * max(
        1.0, initial_value / DEFAULT_INITIAL_VALUE
    )

    for session_idx in range(execution + 1, len(dates)):
        start_weights = weights.copy()
        asset_return = returns.iloc[session_idx].to_numpy()
        gross_return = float(weights @ asset_return)
        invested_start = value
        market_pnl = invested_start * gross_return
        pre_trade_value = invested_start + market_pnl
        drift = weights * (1 + asset_return) / (1 + gross_return)
        cost_dollars = entry_dollars if session_idx == execution + 1 else 0.0
        gross_traded = 0.0
        if session_idx in scheduled:
            target = decide(scheduled[session_idx], session_idx)
            transaction = rebalance_cost(drift, target, rate)
            rebalance_dollars = pre_trade_value * transaction["cost_fraction"]
            cost_dollars += rebalance_dollars
            gross_traded = transaction["gross_traded_fraction"]
            value = pre_trade_value - rebalance_dollars
            weights = target
            trade_rows.append(
                {
                    "date": dates[session_idx],
                    "kind": "rebalance",
                    "cost_dollars": rebalance_dollars,
                    "gross_traded_dollars": pre_trade_value * gross_traded,
                    "gross_traded_fraction": gross_traded,
                    "cost_equation_residual": transaction["cost_equation_residual"],
                }
            )
        else:
            value, weights = pre_trade_value, drift
        net_return = value / previous_reported_value - 1
        residual = value - (previous_reported_value + market_pnl - cost_dollars)
        if abs(residual) > max_reconciliation_error:
            raise RuntimeError("Wealth/cost accounting failed to reconcile.")
        if not np.isclose(weights.sum(), 1, atol=WEIGHT_SUM_ATOL, rtol=0):
            raise RuntimeError("Post-trade weights do not sum to one.")
        daily_rows.append(
            {
                "Date": dates[session_idx],
                "nav_start_reported": previous_reported_value,
                "invested_start": invested_start,
                "market_pnl": market_pnl,
                "trading_cost_dollars": cost_dollars,
                "gross_traded_fraction": gross_traded,
                "gross_return": gross_return,
                "net_return": net_return,
                "nav_usd": value,
                "accounting_residual": residual,
                **{
                    f"start_weight_{asset}": float(weight)
                    for asset, weight in zip(assets, start_weights)
                },
                **{
                    f"end_weight_{asset}": float(weight)
                    for asset, weight in zip(assets, weights)
                },
            }
        )
        previous_reported_value = value

    daily = pd.DataFrame(daily_rows).set_index("Date")
    peak = np.maximum.accumulate(np.r_[initial_value, daily["nav_usd"].to_numpy()])[1:]
    daily["drawdown"] = daily["nav_usd"] / peak - 1
    return BacktestResult(
        daily,
        pd.DataFrame(signal_rows),
        pd.DataFrame(trade_rows),
        dates[execution],
        initial_value,
    )


def _validate_backtest_inputs(
    prices: pd.DataFrame,
    lookback: int,
    frequency: str,
    cost_bps: float,
    initial_value: float,
) -> float:
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Prices need unique ascending session dates.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Price panel must be finite, positive and complete.")
    if lookback < MIN_LOOKBACK or len(prices) < lookback + MIN_LOOKBACK or initial_value <= 0:
        raise ValueError("Insufficient warmup/holding observations or invalid capital.")
    if frequency not in SUPPORTED_FREQUENCIES:
        raise ValueError("Unsupported rebalance frequency.")
    rate = cost_bps / BPS_DIVISOR
    if not 0 <= rate < 1:
        raise ValueError("Invalid transaction-cost assumption.")
    return rate


def _build_rebalance_schedule(
    dates: pd.DatetimeIndex, lookback: int, frequency: str
) -> dict[int, int]:
    periods = dates.to_period(SUPPORTED_FREQUENCIES[frequency])
    origins = [lookback] + [
        idx
        for idx in range(lookback + 1, len(dates) - 2)
        if periods[idx] != periods[idx + 1]
    ]
    # Decisions are evaluated as the simulation reaches their execution dates.
    return {origin + 1: origin for origin in origins}
