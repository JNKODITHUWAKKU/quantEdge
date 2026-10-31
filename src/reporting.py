"""Derive dashboard tables from the newly computed research results."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.release_decision import (
    architecture,
    committee_actions,
    funding_map,
    matched_baselines,
)
from src.risk import gaussian_risk
from src.storage import csv, read_assessment

DEFAULT_LOOKBACK = 1260
REFERENCE_CAPITAL_USD = 1e7
WEALTH_ERROR_LIMIT = 1e-5
SELF_FINANCING_ERROR_LIMIT = 1e-6
ATTRIBUTION_ATOL = 1e-12
CONFIDENCE_LEVELS = (0.9, 0.95, 0.99)
SPENDING_RATES = (0, 0.02, 0.03, 0.04)


def _verify_accounting_reconciliation(daily: pd.DataFrame) -> dict[str, float]:
    checks = {
        "wealth_max_error_usd": float(
            np.max(
                np.abs(REFERENCE_CAPITAL_USD * np.cumprod(1 + daily.net_return) - daily.nav_usd)
            )
        ),
        "self_financing_max_error_usd": float(
            np.max(
                np.abs(
                    daily.nav_usd
                    - daily.nav_start_reported
                    - daily.market_pnl
                    + daily.trading_cost_dollars
                )
            )
        ),
    }
    if (
        checks["wealth_max_error_usd"] >= WEALTH_ERROR_LIMIT
        or checks["self_financing_max_error_usd"] >= SELF_FINANCING_ERROR_LIMIT
    ):
        raise ValueError("Independent wealth reconstruction failed.")
    return checks


def _build_gaussian_euler_table(
    training: pd.DataFrame, weights: np.ndarray, saved: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    mean_vec = training.mean().to_numpy()
    cov_mat = training.cov().to_numpy()
    for confidence in CONFIDENCE_LEVELS:
        risk = gaussian_risk(mean_vec, cov_mat, weights, confidence)
        reference = saved[np.isclose(saved.confidence, confidence)].iloc[0]
        if not np.isclose(
            risk["var"], reference["var"], atol=ATTRIBUTION_ATOL, rtol=0
        ) or not np.isclose(risk["es"], reference.es, atol=ATTRIBUTION_ATOL, rtol=0):
            raise ValueError("Gaussian attribution does not reconcile to its forecast.")
        marginal = (
            -mean_vec
            + norm.ppf(confidence) * (cov_mat @ weights) / risk["standard_deviation"]
        )
        # By Euler's theorem for homogeneous degree-1 risk measures, sum(w_i * marginal_i) == VaR.
        for idx, asset in enumerate(training.columns):
            rows.append(
                {
                    "asset": asset,
                    "confidence": confidence,
                    "weight": weights[idx],
                    "marginal_var": marginal[idx],
                    "var_component": risk["var_components"][idx],
                    "var_component_usd": REFERENCE_CAPITAL_USD * risk["var_components"][idx],
                    "es_component": risk["es_components"][idx],
                    "portfolio_var": risk["var"],
                    "portfolio_es": risk["es"],
                }
            )
    return pd.DataFrame(rows)


def _build_spending_sensitivity_table(real: pd.DataFrame) -> pd.DataFrame:
    spending = []
    for fraction in SPENDING_RATES:
        nominal = (
            real.nominal_no_spending_usd
            - REFERENCE_CAPITAL_USD * fraction * real.withdrawal_terminal_coefficient
        )
        terminal = float(nominal.iloc[-1] / real.cpi_ratio.iloc[-1])
        spending.append(
            {
                "annual_real_spending": fraction,
                "terminal_real_usd": terminal,
                "preserves_real_principal": terminal >= REFERENCE_CAPITAL_USD,
                "path_solvent": bool((nominal > 0).all()),
            }
        )
    return pd.DataFrame(spending)


def build_display(output: Path, returns: pd.DataFrame) -> None:
    """Reconcile accounting, attribution and funding before exposing results."""
    folder = output

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(output / f"{name}.csv", low_memory=False)

    metrics, funding, stress = map(
        read, ["portfolio_metrics", "funding_diagnostics", "historical_stress"]
    )
    assessment = read_assessment(output)
    tables = {
        "matched_baselines": matched_baselines(metrics, funding, stress),
        "funding_decision_map": funding_map(
            read("real_wealth_paths"),
            metrics,
            stress,
            read("hypothetical_stress"),
            assessment,
        ),
        "committee_actions": committee_actions(),
        "model_architecture": architecture(),
    }
    daily = read("sample_gmv_cost_5bps_daily")
    checks = _verify_accounting_reconciliation(daily)
    training = returns.tail(DEFAULT_LOOKBACK)
    weights = (
        read("deployment_weights")
        .query("strategy == 'sample_gmv'")
        .set_index("asset")
        .weight.reindex(training.columns)
        .to_numpy()
    )
    saved = read("deployment_risk").query(
        "strategy == 'sample_gmv' and method == 'gaussian'"
    )
    tables["gaussian_euler_contributions"] = _build_gaussian_euler_table(
        training, weights, saved
    )
    real = read("real_wealth_paths").query("strategy == 'sample_gmv' and cost_bps == 5")
    tables["spending_sensitivity"] = _build_spending_sensitivity_table(real)
    for name, frame in tables.items():
        csv(frame, folder / f"{name}.csv")
    selected = metrics.query("strategy == 'sample_gmv' and cost_bps == 5").iloc[0]
    selected_funding = funding.query("strategy == 'sample_gmv' and cost_bps == 5").iloc[
        0
    ]
    headline = {
        "portfolio": "sample_gmv",
        "monitor": "garch_t",
        "decision": "HOLD / NO TRADE",
        "cagr": selected.cagr,
        "annualized_volatility": selected.annualized_volatility,
        "maximum_drawdown_loss": selected.maximum_drawdown_loss,
        "real_total_return": selected_funding.real_total_return,
        "terminal_real_usd": selected_funding.terminal_real_no_spending_usd,
        "average_held_AGG": float(daily.end_weight_AGG.mean()),
        **checks,
    }
    csv(pd.DataFrame([headline]), folder / "headline.csv")
