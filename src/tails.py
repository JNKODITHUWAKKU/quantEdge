"""GPD likelihood, coherent empirical-body tail splice, and explicit stability rule."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto

from src.risk import confidence, empirical_tail

INFEASIBLE_NLL = 1e100
TAYLOR_XI_THRESHOLD = 1e-6
DEFAULT_MIN_EXCEEDANCES = 50
DEFAULT_SHAPE_BOUNDS = (-0.95, 2.0)
INITIAL_SHAPES = (0.0, 0.2, -0.2)
LOG_SCALE_BOUNDS = (-5.0, 5.0)
LOG_SCALE_BOUND_LIMIT = 5.0
SUPPORT_MARGIN = 1e-8
BOUNDARY_TOLERANCE = 1e-5
SLSQP_FTOL = 1e-9
SLSQP_MAXITER = 250
REGULAR_LIKELIHOOD_XI = -0.5
FINITE_VARIANCE_XI = 0.5
FINITE_ES_XI = 1.0
QUANTILE_XI_THRESHOLD = 1e-8
JOIN_TOLERANCE = 1e-12
ES_VAR_TOLERANCE = 1e-10
MIN_EVT_OBSERVATIONS = 100
GRID_CONFIDENCE_LEVELS = (0.9, 0.95, 0.99)
RELATIVE_SPAN_FLOOR = 1e-10


def gpd_objective(theta: np.ndarray, excess: np.ndarray) -> tuple[float, np.ndarray]:
    """Negative log likelihood and analytic gradient in (shape, log scale)."""
    xi, eta = map(float, theta)
    scaled = np.asarray(excess, float) / math.exp(eta)
    if np.any(1 + xi * scaled <= 0):
        return INFEASIBLE_NLL, np.zeros(2)
    sample_size = len(scaled)
    if abs(xi) < TAYLOR_XI_THRESHOLD:
        # Use 2nd-order Taylor expansion around xi=0 (exponential limit) to avoid 0/0 cancellation.
        first_order = scaled - scaled * scaled / 2
        second_order = scaled**3 / 3 - scaled * scaled / 2
        value = sample_size * eta + np.sum(
            scaled + xi * first_order + xi * xi * second_order
        )
        grad = np.array(
            [
                np.sum(first_order + 2 * xi * second_order),
                sample_size
                - np.sum(
                    scaled
                    + xi * (scaled - scaled * scaled)
                    + xi * xi * (scaled**3 - scaled * scaled)
                ),
            ]
        )
    else:
        logs = np.log1p(xi * scaled)
        ratio = scaled / (1 + xi * scaled)
        value = sample_size * eta + (1 + 1 / xi) * logs.sum()
        grad = np.array(
            [
                -logs.sum() / xi**2 + (1 + 1 / xi) * ratio.sum(),
                sample_size - (1 + xi) * ratio.sum(),
            ]
        )
    return float(value), grad


def fit_gpd(
    excess: np.ndarray,
    minimum: int = DEFAULT_MIN_EXCEEDANCES,
    bounds: tuple = DEFAULT_SHAPE_BOUNDS,
) -> dict:
    excess_vec = np.asarray(excess, float)
    if (
        excess_vec.ndim != 1
        or len(excess_vec) < minimum
        or not np.isfinite(excess_vec).all()
        or np.any(excess_vec <= 0)
    ):
        return {
            "fit_status": "INSUFFICIENT_OR_INVALID_EXCESSES",
            "exceedances": len(excess_vec),
        }
    unit = float(excess_vec.mean())
    normalized = excess_vec / unit
    solutions = _solve_gpd_candidates(normalized, bounds)
    if not solutions:
        return {"fit_status": "OPTIMIZATION_FAILED", "exceedances": len(excess_vec)}
    best = min(solutions, key=lambda fit: fit.fun)
    xi, eta = map(float, best.x)
    beta = float(np.exp(eta) * unit)
    if (
        min(
            xi - bounds[0],
            bounds[1] - xi,
            eta + LOG_SCALE_BOUND_LIMIT,
            LOG_SCALE_BOUND_LIMIT - eta,
        )
        < BOUNDARY_TOLERANCE
    ):
        return {
            "fit_status": "NUMERIC_BOUNDARY",
            "exceedances": len(excess_vec),
            "shape": xi,
            "scale": beta,
        }
    fitted = genpareto.cdf(np.sort(excess_vec), xi, scale=beta)
    exceedances = len(excess_vec)
    ks = float(
        max(
            np.max(np.arange(1, exceedances + 1) / exceedances - fitted),
            np.max(fitted - np.arange(exceedances) / exceedances),
        )
    )
    return {
        "fit_status": "OK",
        "exceedances": exceedances,
        "shape": xi,
        "scale": beta,
        "nll": float(best.fun + exceedances * np.log(unit)),
        "successful_starts": len(solutions),
        "iterations": int(best.nit),
        "support_min": float(np.min(1 + xi * excess_vec / beta)),
        "ks_statistic": ks,
        "likelihood_regular": bool(xi > REGULAR_LIKELIHOOD_XI),
        "finite_es": bool(xi < FINITE_ES_XI),
        "finite_variance": bool(xi < FINITE_VARIANCE_XI),
    }


def _solve_gpd_candidates(normalized: np.ndarray, bounds: tuple) -> list:
    solutions = []
    max_norm = normalized.max()
    for shape in INITIAL_SHAPES:
        scale = max(1 - shape, -shape * max_norm + 0.1)
        fit = minimize(
            lambda params: gpd_objective(params, normalized),
            [shape, np.log(scale)],
            jac=True,
            method="SLSQP",
            bounds=[bounds, LOG_SCALE_BOUNDS],
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda params: (
                        1 + params[0] * max_norm / np.exp(params[1]) - SUPPORT_MARGIN
                    ),
                    "jac": lambda params: np.array(
                        [
                            max_norm / np.exp(params[1]),
                            -params[0] * max_norm / np.exp(params[1]),
                        ]
                    ),
                }
            ],
            options={"ftol": SLSQP_FTOL, "maxiter": SLSQP_MAXITER},
        )
        if (
            fit.success
            and np.isfinite(fit.fun)
            and np.min(1 + fit.x[0] * normalized / np.exp(fit.x[1])) > SUPPORT_MARGIN
        ):
            solutions.append(fit)
    return solutions


def tail_quantile(
    u: float, pu: float, xi: float, beta: float, c: float
) -> float:
    level = confidence(c)
    if beta <= 0 or not 0 < pu < 1 or level < 1 - pu - JOIN_TOLERANCE:
        raise ValueError(
            "Quantile must lie in the fitted tail, with valid scale and exceedance mass."
        )
    logratio = math.log(pu / (1 - level))
    return float(
        u
        + beta
        * (
            logratio
            if abs(xi) < QUANTILE_XI_THRESHOLD
            else np.expm1(xi * logratio) / xi
        )
    )


def hybrid_risk(losses: np.ndarray, fit: dict, c: float) -> dict:
    """Each body atom keeps mass 1/n; fitted continuous tail has mass k/n."""
    loss_vec = np.asarray(losses, float)
    level = confidence(c)
    if fit.get("fit_status") != "OK":
        raise ValueError("A supported GPD fit is required.")
    threshold, xi, beta, tail_prob = (
        float(fit[key])
        for key in ("threshold", "shape", "scale", "tail_probability")
    )
    body = np.sort(loss_vec[loss_vec <= threshold])
    sample_size = len(loss_vec)
    join = 1 - tail_prob
    if len(body) != sample_size - fit["exceedances"] or not np.isclose(
        len(body) / sample_size, join
    ):
        raise ValueError("Body/tail probability does not reconcile.")
    if level > join + JOIN_TOLERANCE:
        quantile = tail_quantile(threshold, tail_prob, xi, beta, level)
        es = (
            (quantile + beta - xi * threshold) / (1 - xi)
            if xi < FINITE_ES_XI
            else np.inf
        )
        segment = "GPD_TAIL"
    else:
        index = max(0, int(math.ceil(sample_size * level - JOIN_TOLERANCE)) - 1)
        quantile = float(body[min(index, len(body) - 1)])
        mass = np.maximum(
            0,
            np.arange(1, len(body) + 1) / sample_size
            - np.maximum(level, np.arange(len(body)) / sample_size),
        )
        tail_mean = threshold + beta / (1 - xi) if xi < FINITE_ES_XI else np.inf
        es = float((mass @ body + tail_prob * tail_mean) / (1 - level))
        segment = "EMPIRICAL_BODY_WITH_GPD_ES"
    if np.isfinite(es) and es < quantile - ES_VAR_TOLERANCE:
        raise RuntimeError("Hybrid ES is smaller than VaR.")
    return {"var": quantile, "es": es, "quantile_segment": segment}


def threshold_grid(losses: np.ndarray, settings: dict) -> list[dict]:
    loss_vec = np.asarray(losses, float)
    if (
        loss_vec.ndim != 1
        or len(loss_vec) < MIN_EVT_OBSERVATIONS
        or not np.isfinite(loss_vec).all()
    ):
        raise ValueError(
            "EVT needs a finite loss series; no missing values are dropped."
        )
    rows = []
    for percentile in settings["threshold_probabilities"]:
        threshold = empirical_tail(loss_vec, percentile)["var"]
        excess = loss_vec[loss_vec > threshold] - threshold
        fit = fit_gpd(
            excess,
            settings["minimum_exceedances"],
            (
                settings["shape_lower_numeric_bound"],
                settings["shape_upper_numeric_bound"],
            ),
        )
        fit.update(
            threshold_percentile=percentile,
            threshold=threshold,
            tail_probability=len(excess) / len(loss_vec),
            observations=len(loss_vec),
            mean_residual_life=float(excess.mean()) if len(excess) else np.nan,
            at_least_75_exceedances=len(excess) >= 75,
            at_least_100_exceedances=len(excess) >= 100,
        )
        if fit["fit_status"] == "OK":
            fit["modified_scale"] = fit["scale"] - fit["shape"] * threshold
            for level in GRID_CONFIDENCE_LEVELS:
                risk = hybrid_risk(loss_vec, fit, level)
                fit[f"var_{int(100 * level)}"] = risk["var"]
                fit[f"es_{int(100 * level)}"] = risk["es"]
        rows.append(fit)
    return rows


def choose_threshold(rows: list[dict], settings: dict) -> dict:
    minimum = settings["minimum_stable_thresholds"]
    regions = []
    for size in range(len(rows), minimum - 1, -1):
        for start in range(len(rows) - size + 1):
            region = rows[start : start + size]
            if _is_stable_region(region, settings):
                regions.append((start, size))
        if regions:
            break
    if not regions:
        return {"selection_status": "NO_STABLE_REGION", "selected": None}
    start, size = regions[0]
    chosen = start + (size - 1) // 2
    return {
        "selection_status": "DIAGNOSTIC_RULE_SUPPORTED",
        "selected": rows[chosen],
        "region_low": rows[start]["threshold_percentile"],
        "region_high": rows[start + size - 1]["threshold_percentile"],
        "region_size": size,
    }


def _is_stable_region(region: list[dict], settings: dict) -> bool:
    if any(
        row.get("fit_status") != "OK" or not row.get("finite_es", False)
        for row in region
    ):
        return False
    shape = np.array([row["shape"] for row in region])
    modified = np.array([row["modified_scale"] for row in region])
    scale = np.array([row["scale"] for row in region])
    var_99 = np.array([row["var_99"] for row in region])
    return bool(
        np.ptp(shape) <= settings["maximum_shape_span"]
        and np.ptp(modified)
        <= settings["maximum_modified_scale_span_over_median_scale"] * np.median(scale)
        and np.ptp(var_99)
        <= settings["maximum_q99_relative_span"]
        * max(RELATIVE_SPAN_FLOOR, np.median(np.abs(var_99)))
    )


def gpd_qqpp(excess: np.ndarray, fit: dict) -> list[dict]:
    ordered = np.sort(excess)
    prob = (np.arange(len(ordered)) + 0.5) / len(ordered)
    return [
        {
            "empirical_excess": float(emp_val),
            "theoretical_excess": float(theo_val),
            "empirical_probability": float(emp_prob),
            "fitted_probability": float(fit_prob),
        }
        for emp_val, theo_val, emp_prob, fit_prob in zip(
            ordered,
            genpareto.ppf(prob, fit["shape"], scale=fit["scale"]),
            prob,
            genpareto.cdf(ordered, fit["shape"], scale=fit["scale"]),
        )
    ]


def refit_ks_bootstrap(fit: dict, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    statistics = []
    failures = 0
    for _ in range(replicates):
        simulated_excess = genpareto.rvs(
            fit["shape"],
            scale=fit["scale"],
            size=fit["exceedances"],
            random_state=rng,
        )
        simulated = fit_gpd(simulated_excess)
        if simulated["fit_status"] != "OK":
            failures += 1
        else:
            statistics.append(simulated["ks_statistic"])
    # No reported p-value if bootstrap fit failures would be silently excluded.
    pvalue = (
        (1 + np.sum(np.asarray(statistics) >= fit["ks_statistic"])) / (replicates + 1)
        if failures == 0
        else None
    )
    return {
        "ks_bootstrap_pvalue": None if pvalue is None else float(pvalue),
        "ks_bootstrap_replicates": replicates,
        "ks_bootstrap_fit_failures": failures,
        "ks_reference": (
            "iid fitted excesses; parameter refitted; dependence/selection not covered"
        ),
    }
