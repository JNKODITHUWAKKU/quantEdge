"""Transparent convex Markowitz baselines and recorded numerical checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

MIN_OBSERVATIONS = 3
DEFAULT_ANNUALIZATION = 252
SYMMETRY_ATOL = 1e-12
MIN_EIGENVALUE = 1e-12
DEFAULT_FEASIBILITY_TOLERANCE = 1e-8
DEFAULT_GAP_TOLERANCE = 1e-6
SLSQP_FTOL = 1e-12
SLSQP_MAXITER = 500
DEFAULT_FRONTIER_POINTS = 11
POLICY_BENCHMARK_WEIGHTS = (("SPY", 0.4), ("AGG", 0.6))
GMV_STRATEGIES = {"sample_gmv", "shrinkage_gmv"}


def estimate_moments(
    returns: pd.DataFrame,
    method: str = "sample",
    annualization: int = DEFAULT_ANNUALIZATION,
) -> dict:
    matrix = returns.to_numpy(dtype=float)
    if (
        len(matrix) < MIN_OBSERVATIONS
        or not np.isfinite(matrix).all()
        or returns.index.has_duplicates
    ):
        raise ValueError("Moment estimation needs finite, uniquely dated returns.")
    if method == "sample":
        covariance = np.cov(matrix, rowvar=False, ddof=1)
        intensity = 0.0
    elif method == "shrinkage":
        from sklearn.covariance import LedoitWolf

        fit = LedoitWolf(assume_centered=False).fit(matrix)
        # Match the n-1 covariance normalization used in the sample baseline.
        covariance = fit.covariance_ * len(matrix) / (len(matrix) - 1)
        intensity = float(fit.shrinkage_)
    else:
        raise ValueError("Unknown covariance estimator.")
    return {
        "mean": matrix.mean(axis=0) * annualization,
        "covariance": np.atleast_2d(covariance) * annualization,
        "shrinkage_intensity": intensity,
        "method": method,
        "n": len(matrix),
        "last_observation": str(returns.index[-1].date()),
        "assets": list(returns.columns),
    }


def solve_markowitz(
    mean: np.ndarray,
    covariance: np.ndarray,
    target: float | None = None,
    feasibility_tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
) -> dict:
    mean_vec, cov_mat = np.asarray(mean, float), np.asarray(covariance, float)
    eigenvalues = _validate_markowitz_inputs(
        mean_vec, cov_mat, target, feasibility_tolerance
    )
    asset_count = len(mean_vec)
    scale = float(np.trace(cov_mat) / asset_count)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov_mat @ weights / scale)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2 * cov_mat @ weights / scale

    constraints, aub, bub = _build_constraints(mean_vec, asset_count, target)
    initial = np.ones(asset_count) / asset_count
    if target is not None and mean_vec @ initial < target:
        initial = np.eye(asset_count)[np.argmax(mean_vec)]

    fit = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * asset_count,
        constraints=constraints,
        options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER},
    )
    weights, violation = _clean_and_verify_weights(
        fit, mean_vec, target, feasibility_tolerance
    )
    gap = _verify_optimality_certificate(
        gradient(weights), weights, aub, bub, asset_count, gap_tolerance
    )
    return {
        "weights": weights,
        "estimated_return_annual": float(mean_vec @ weights),
        "estimated_volatility_annual": float(np.sqrt(weights @ cov_mat @ weights)),
        "target_return_annual": target,
        "feasibility_residual": violation,
        "scaled_optimality_gap": gap,
        "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "iterations": int(fit.nit),
        "maximum_weight": float(weights.max()),
    }


def _validate_markowitz_inputs(
    mean: np.ndarray,
    covariance: np.ndarray,
    target: float | None,
    feasibility_tolerance: float,
) -> np.ndarray:
    asset_count = len(mean)
    if (
        covariance.shape != (asset_count, asset_count)
        or not np.isfinite(covariance).all()
        or not np.isfinite(mean).all()
    ):
        raise ValueError("Invalid moment dimensions or non-finite estimates.")
    if not np.allclose(covariance, covariance.T, atol=SYMMETRY_ATOL, rtol=0):
        raise ValueError("Covariance is not symmetric.")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= MIN_EIGENVALUE:
        raise ValueError(
            "Covariance is not sufficiently positive definite; no silent eigenvalue repair."
        )
    if target is not None and (
        not np.isfinite(target) or target > mean.max() + feasibility_tolerance
    ):
        raise ValueError("Target return is infeasible for this long-only universe.")
    return eigenvalues


def _build_constraints(
    mean: np.ndarray, asset_count: int, target: float | None
) -> tuple[list[dict], np.ndarray | None, np.ndarray | None]:
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones(asset_count),
        }
    ]
    aub, bub = None, None
    if target is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: mean @ w - target, "jac": lambda w: mean}
        )
        aub, bub = -mean[None, :], np.array([-target])
    return constraints, aub, bub


def _clean_and_verify_weights(
    fit: object,
    mean: np.ndarray,
    target: float | None,
    feasibility_tolerance: float,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(fit.x)
    violation = max(
        abs(raw.sum() - 1),
        float(max(0.0, -raw.min(), raw.max() - 1)),
        0.0 if target is None else float(max(0.0, target - mean @ raw)),
    )
    if not fit.success or violation > feasibility_tolerance:
        raise RuntimeError(
            f"Optimization failed: {fit.message}; feasibility residual={violation:.3g}"
        )
    # Only numerical dust is clipped; the raw feasibility residual is retained.
    weights = np.maximum(raw, 0)
    weights /= weights.sum()
    if target is not None and mean @ weights < target - feasibility_tolerance:
        raise RuntimeError("Numerical cleanup violated the target constraint.")
    return weights, violation


def _verify_optimality_certificate(
    grad: np.ndarray,
    weights: np.ndarray,
    aub: np.ndarray | None,
    bub: np.ndarray | None,
    asset_count: int,
    gap_tolerance: float,
) -> float:
    # Independent convex optimality certificate: min linearized objective over
    # the same feasible polytope. The Frank-Wolfe gap bounds objective suboptimality.
    certificate = linprog(
        grad,
        A_ub=aub,
        b_ub=bub,
        A_eq=np.ones((1, asset_count)),
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * asset_count,
        method="highs",
    )
    if not certificate.success:
        raise RuntimeError("Independent optimality certificate failed.")
    gap = max(0.0, float(grad @ (weights - certificate.x)))
    if gap > gap_tolerance:
        raise RuntimeError(
            f"Optimizer returned an insufficiently optimal solution: gap={gap:.3g}"
        )
    return gap


def target_for_strategy(
    history: pd.DataFrame, strategy: str, annualization: int = DEFAULT_ANNUALIZATION
) -> tuple[np.ndarray, dict]:
    assets = list(history.columns)
    weights = np.zeros(len(assets))
    if strategy == "equal_weight":
        weights[:] = 1 / len(assets)
    elif strategy == "policy_40_60":
        for symbol, weight in POLICY_BENCHMARK_WEIGHTS:
            weights[assets.index(symbol)] = weight
    elif strategy == "agg_only":
        weights[assets.index("AGG")] = 1.0
    elif strategy in GMV_STRATEGIES:
        moments = estimate_moments(
            history,
            "sample" if strategy == "sample_gmv" else "shrinkage",
            annualization,
        )
        result = solve_markowitz(moments["mean"], moments["covariance"])
        return result["weights"], {
            key: value for key, value in result.items() if key != "weights"
        } | {
            "shrinkage_intensity": moments["shrinkage_intensity"],
            "covariance_estimator": moments["method"],
        }
    else:
        raise ValueError("Unknown strategy.")
    return weights, {
        "covariance_estimator": "not_applicable",
        "maximum_weight": float(weights.max()),
    }


def efficient_frontier(
    moments: dict, points: int = DEFAULT_FRONTIER_POINTS
) -> list[dict]:
    gmv = solve_markowitz(moments["mean"], moments["covariance"])
    targets = np.linspace(
        gmv["estimated_return_annual"], float(np.max(moments["mean"])), points
    )
    return [
        solve_markowitz(moments["mean"], moments["covariance"], float(target))
        for target in targets
    ]
