"""Forecast inputs from origin-time holdings; outcomes are attached separately."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.risk import (
    gaussian_risk,
    historical_risk,
    normal_mc_coefficients,
    projected_mc,
    seeded_identifier,
)

START_WEIGHT_ATOL = 1e-11
HPL_RECONCILIATION_ATOL = 1e-11
VAR_ES_TOLERANCE = 1e-12
SUPPORTED_MEAN_MODES = {"sample", "zero"}
DEFAULT_DESIGN_VARIANCE_BUFFER = 1.5
MC_SE_KEYS = ("var_mc_se", "es_mc_se", "var_mc_halfwidth95", "es_mc_halfwidth95")
ATTRIBUTION_METHODS = {"historical", "gaussian"}


def origin_book(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    assets: list[str],
) -> pd.DataFrame:
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise ValueError("Holdings history must have unique ascending dates.")
    first = signals.iloc[0]
    origin = pd.Timestamp(first["execution_date"])
    if pd.Timestamp(first["signal_date"]) >= origin:
        raise ValueError("Initial signal must precede execution.")
    initial_trades = trades[trades["kind"] == "initial_purchase"]
    if len(initial_trades) != 1 or pd.Timestamp(initial_trades.iloc[0]["date"]) != origin:
        raise ValueError("Initial execution ledger is ambiguous.")
    weights = first[[f"weight_{asset}" for asset in assets]].to_numpy(float)
    wealth = float(
        daily.iloc[0]["nav_start_reported"] - initial_trades.iloc[0]["cost_dollars"]
    )
    rows = []
    for date, row in daily.iterrows():
        if sessions[sessions.get_loc(origin) + 1] != date:
            raise ValueError("Risk outcomes must be the next eligible session.")
        # Origin state is built from the initial executed signal or the PREVIOUS
        # day's ending holdings, never from today's realized return.
        record = {
            "forecast_origin": origin,
            "realization_date": date,
            "origin_value_usd": wealth,
            **{f"weight_{asset}": float(weight) for asset, weight in zip(assets, weights)},
        }
        rows.append(record)
        # This comparison is an accounting invariant, not a forecast input.
        if not np.allclose(
            weights,
            row[[f"start_weight_{asset}" for asset in assets]].to_numpy(float),
            atol=START_WEIGHT_ATOL,
            rtol=0,
        ):
            raise ValueError(
                "Origin holdings differ from the independently recorded beginning positions."
            )
        origin, wealth = date, float(row["nav_usd"])
        weights = row[[f"end_weight_{asset}" for asset in assets]].to_numpy(float)
    return pd.DataFrame(rows)


def forecast_cross_section(
    returns: pd.DataFrame,
    origin: pd.Timestamp,
    exposures: dict[str, dict],
    lookback: int,
    levels: list[float],
    mc_settings: dict,
    base_seed: int,
    mean_mode: str = "sample",
) -> tuple:
    training, data, mean, covariance = _prepare_training_moments(
        returns, origin, lookback, mean_mode
    )
    # Draw one common standard-normal MC batch sized to the widest portfolio sigma at this origin.
    max_sigma = max(
        gaussian_risk(mean, covariance, exposure["weights"], levels[0])[
            "standard_deviation"
        ]
        for exposure in exposures.values()
    )
    seed = seeded_identifier(
        base_seed, "forecast", str(origin.date()), lookback, mean_mode
    )
    coeff = normal_mc_coefficients(
        levels,
        seed,
        max_sigma,
        mc_settings["batches"],
        mc_settings["initial_draws_per_batch"],
        mc_settings["maximum_draws_per_batch"],
        mc_settings["target_ci95_halfwidth_return"],
        mc_settings.get("normal_design_variance_buffer", DEFAULT_DESIGN_VARIANCE_BUFFER),
    )
    records, contributions = [], []
    assets = list(training.columns)
    for name, exposure in exposures.items():
        weights = np.asarray(exposure["weights"], float)
        weight_hash = hashlib.sha256(
            json.dumps(weights.tolist(), separators=(",", ":")).encode()
        ).hexdigest()
        for level in levels:
            historical = historical_risk(data, weights, level)
            analytic = gaussian_risk(mean, covariance, weights, level)
            mc = projected_mc(
                coeff[level], analytic["mean_return"], analytic["standard_deviation"]
            )
            models = {
                "historical": historical,
                "gaussian": analytic,
                "monte_carlo": mc,
            }
            for method, result in models.items():
                records.append(
                    _build_cross_section_record(
                        name=name,
                        method=method,
                        level=level,
                        origin=origin,
                        exposure=exposure,
                        training=training,
                        lookback=lookback,
                        mean_mode=mean_mode,
                        weight_hash=weight_hash,
                        weights=weights,
                        assets=assets,
                        result=result,
                        analytic=analytic,
                        seed=seed,
                        mc_settings=mc_settings,
                    )
                )
                if method in ATTRIBUTION_METHODS:
                    contributions.extend(
                        _build_contribution_rows(
                            name=name,
                            method=method,
                            level=level,
                            origin=origin,
                            exposure=exposure,
                            assets=assets,
                            result=result,
                            mean_mode=mean_mode,
                            lookback=lookback,
                        )
                    )
    return pd.DataFrame(records), pd.DataFrame(contributions)


def _prepare_training_moments(
    returns: pd.DataFrame,
    origin: pd.Timestamp,
    lookback: int,
    mean_mode: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be unique and ascending.")
    training = returns.loc[:origin].tail(lookback)
    if (
        len(training) != lookback
        or training.index[-1] != origin
        or not np.isfinite(training.to_numpy()).all()
    ):
        raise ValueError("The exact past-only risk window is not available.")
    if mean_mode not in SUPPORTED_MEAN_MODES:
        raise ValueError("Unsupported daily mean convention.")
    data = training.to_numpy(float)
    mean = data.mean(axis=0)
    covariance = np.atleast_2d(np.cov(data, rowvar=False, ddof=1))
    if mean_mode == "zero":
        data = data - mean
        mean = np.zeros_like(mean)
    return training, data, mean, covariance


def _build_cross_section_record(
    name: str,
    method: str,
    level: float,
    origin: pd.Timestamp,
    exposure: dict,
    training: pd.DataFrame,
    lookback: int,
    mean_mode: str,
    weight_hash: str,
    weights: np.ndarray,
    assets: list[str],
    result: dict,
    analytic: dict,
    seed: int,
    mc_settings: dict,
) -> dict:
    if result["es"] < result["var"] - VAR_ES_TOLERANCE:
        raise RuntimeError("Inconsistent VaR/ES pair.")
    origin_value = float(exposure["origin_value_usd"])
    record = {
        "strategy": name,
        "method": method,
        "confidence": level,
        "forecast_origin": origin,
        "realization_date": exposure["realization_date"],
        "training_start": training.index[0],
        "training_end": training.index[-1],
        "training_observations": len(training),
        "lookback": lookback,
        "mean_mode": mean_mode,
        "horizon_sessions": 1,
        "pnl_type": "frozen_exposure_market_loss",
        "holdings_sha256": weight_hash,
        "origin_value_usd": origin_value,
        "var": float(result["var"]),
        "es": float(result["es"]),
        "var_usd": float(origin_value * result["var"]),
        "es_usd": float(origin_value * result["es"]),
        "fit_status": "OK",
        "actual_model_used": method,
        "evidence_class": "exploratory_development_only",
        "empirical_tail_mass": lookback * (1 - level),
        "forecast_above_100pct_loss": bool(result["var"] > 1 or result["es"] > 1),
        "var_mc_se": 0.0,
        "es_mc_se": 0.0,
        "var_mc_halfwidth95": 0.0,
        "es_mc_halfwidth95": 0.0,
        "mc_draws": 0,
        "mc_seed": str(seed) if method == "monte_carlo" else "",
        "mc_precision_reached": True,
        "mc_agreement_with_analytic": True,
        **{f"weight_{asset}": float(weight) for asset, weight in zip(assets, weights)},
    }
    if method == "monte_carlo":
        for key in MC_SE_KEYS:
            record[key] = result[key]
        record.update(
            {
                "mc_draws": result["draws"],
                "mc_batches": result["batches"],
                "mc_precision_reached": bool(result["precision_reached"]),
                "mc_attempts": result["attempts"],
            }
        )
        record["mc_agreement_with_analytic"] = all(
            abs(result[field] - analytic[field])
            <= mc_settings["agreement_standard_errors"] * result[field + "_mc_se"]
            + VAR_ES_TOLERANCE
            for field in ("var", "es")
        )
    if method == "gaussian":
        record["probability_loss_above_100pct"] = analytic[
            "probability_loss_above_100pct"
        ]
    return record


def _build_contribution_rows(
    name: str,
    method: str,
    level: float,
    origin: pd.Timestamp,
    exposure: dict,
    assets: list[str],
    result: dict,
    mean_mode: str,
    lookback: int,
) -> list[dict]:
    origin_value = float(exposure["origin_value_usd"])
    return [
        {
            "strategy": name,
            "method": method,
            "confidence": level,
            "forecast_origin": origin,
            "realization_date": exposure["realization_date"],
            "asset": asset,
            "es_component": float(value),
            "es_component_usd": float(value * origin_value),
            "portfolio_es": result["es"],
            "reconciliation_residual": result["contribution_residual"],
            "mean_mode": mean_mode,
            "lookback": lookback,
        }
        for asset, value in zip(assets, result["es_components"])
    ]


def attach_outcomes(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    daily_by_strategy: dict[str, pd.DataFrame],
    assets: list[str],
) -> pd.DataFrame:
    """Called after forecast CSV is written; no forecasting happens in this function."""
    records = []
    weight_cols = [f"weight_{asset}" for asset in assets]
    for (name, date), block in forecasts.groupby(
        ["strategy", "realization_date"], sort=False
    ):
        realization_ts = pd.Timestamp(date)
        row = block.iloc[0]
        weights = row[weight_cols].to_numpy(float)
        realized = float(-weights @ returns.loc[realization_ts, assets].to_numpy(float))
        daily = daily_by_strategy[name].loc[realization_ts]
        if abs(realized + daily["gross_return"]) > HPL_RECONCILIATION_ATOL:
            raise RuntimeError(
                "Frozen-exposure P&L does not reconcile with the holdings ledger."
            )
        records.append(
            {
                "strategy": name,
                "realization_date": realization_ts,
                "realized_hpl_loss": realized,
                "realized_actual_net_loss_reported": float(-daily["net_return"]),
                "actual_trading_cost_usd": float(daily["trading_cost_dollars"]),
            }
        )
    result = forecasts.merge(
        pd.DataFrame(records),
        on=["strategy", "realization_date"],
        how="left",
        validate="many_to_one",
    )
    if result.realized_hpl_loss.isna().any():
        raise ValueError("Realized data missing; no forecast is dropped.")
    result["exception"] = result.realized_hpl_loss > result["var"]
    result["realized_hpl_loss_usd"] = result.realized_hpl_loss * result.origin_value_usd
    result["mc_exception_boundary_uncertain"] = (result.method == "monte_carlo") & (
        np.abs(result.realized_hpl_loss - result["var"]) <= result.var_mc_halfwidth95
    )
    return result
