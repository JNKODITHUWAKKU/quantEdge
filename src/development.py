"""Recreate the original development frontier without using later prices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import mean_uncertainty
from src.portfolio import efficient_frontier, estimate_moments, solve_markowitz
from src.storage import csv

ESTIMATION_METHODS = ("sample", "shrinkage")
PERTURBATION_DIRECTIONS = (-1, 1)
FEASIBILITY_TOLERANCE = 1e-8
SENSITIVITY_EVIDENCE_CLASS = "development_sensitivity_not_a_return_forecast"


def build_development(prices: pd.DataFrame, cfg: dict, output: Path) -> None:
    """Build only the development diagnostics consumed by the final dashboard."""
    output.mkdir(parents=True, exist_ok=True)
    training = (
        prices.loc[: cfg["run"]["end"]]
        .pct_change(fill_method=None)
        .iloc[1:]
        .tail(cfg["run"]["lookback_returns"])
    )
    annual, hac = cfg["run"]["annualization"], cfg["run"]["hac_lags"]
    uncertainty = mean_uncertainty(training, annual, hac)
    csv(uncertainty, output / "mean_uncertainty.csv")
    frontier_rows, sensitivity_rows = [], []
    assets = list(training.columns)
    estimate_date = str(training.index[-1].date())

    for method in ESTIMATION_METHODS:
        moments = estimate_moments(training, method, annual)
        for number, point in enumerate(
            efficient_frontier(moments, cfg["frontier"]["points"])
        ):
            frontier_rows.append(
                {
                    "method": method,
                    "point": number,
                    "estimate_date": estimate_date,
                    "shrinkage_intensity": moments["shrinkage_intensity"],
                    **{key: value for key, value in point.items() if key != "weights"},
                    **{
                        f"weight_{asset}": weight
                        for asset, weight in zip(assets, point["weights"])
                    },
                }
            )
        gmv = solve_markowitz(moments["mean"], moments["covariance"])
        # Anchor the sensitivity test halfway between GMV and the highest single-asset mean.
        target = gmv["estimated_return_annual"] + cfg["frontier"][
            "mean_sensitivity_target_fraction"
        ] * (moments["mean"].max() - gmv["estimated_return_annual"])
        base = solve_markowitz(moments["mean"], moments["covariance"], float(target))
        for asset_idx, asset in enumerate(assets):
            standard_error = float(
                uncertainty.iloc[asset_idx]["mean_hac_standard_error_annual"]
            )
            for direction in PERTURBATION_DIRECTIONS:
                sensitivity_rows.append(
                    _evaluate_perturbation(
                        method=method,
                        asset=asset,
                        asset_idx=asset_idx,
                        assets=assets,
                        direction=direction,
                        standard_error=standard_error,
                        moments=moments,
                        target=float(target),
                        base_weights=base["weights"],
                    )
                )
    csv(pd.DataFrame(frontier_rows), output / "frontier.csv")
    csv(pd.DataFrame(sensitivity_rows), output / "mean_sensitivity.csv")


def _evaluate_perturbation(
    method: str,
    asset: str,
    asset_idx: int,
    assets: list[str],
    direction: int,
    standard_error: float,
    moments: dict,
    target: float,
    base_weights: np.ndarray,
) -> dict:
    perturbed_mean = moments["mean"].copy()
    perturbed_mean[asset_idx] += direction * standard_error
    row = {
        "method": method,
        "asset_perturbed": asset,
        "direction": direction,
        "mean_shift_annual": direction * standard_error,
        "fixed_target_annual": target,
        "evidence_class": SENSITIVITY_EVIDENCE_CLASS,
    }
    if target > perturbed_mean.max() + FEASIBILITY_TOLERANCE:
        row["status"] = "INFEASIBLE"
        return row
    fitted = solve_markowitz(perturbed_mean, moments["covariance"], target)
    weight_diff = np.abs(fitted["weights"] - base_weights)
    row.update(
        {
            "status": "FEASIBLE",
            "weight_l1_change": float(weight_diff.sum()),
            "maximum_absolute_weight_change": float( np.max(weight_diff)),
            **{
                f"weight_{symbol}": weight
                for symbol, weight in zip(assets, fitted["weights"])
            },
        }
    )
    return row
