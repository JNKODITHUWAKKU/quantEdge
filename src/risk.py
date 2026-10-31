"""Matched one-day loss risk, finite-sample ES, and explicit simulation error."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import norm, t

ROUND_BOUNDARY_TOLERANCE = 1e-10
QUANTILE_CEIL_EPSILON = 1e-12
REMAINING_MASS_TOLERANCE = 1e-9
WEIGHT_RECONCILIATION_ATOL = 1e-11
WEIGHT_SUM_ATOL = 1e-10
SYMMETRY_ATOL = 1e-12
SYMMETRY_RTOL = 1e-10
EIGENVALUE_TOLERANCE = -1e-12
MIN_HISTORICAL_ROWS = 2
MIN_BATCHES = 4
MIN_DRAWS_PER_BATCH = 20
CI95_T_QUANTILE = 0.975
DEFAULT_DESIGN_VARIANCE_BUFFER = 1.5
MIN_BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_LOWER_QUANTILE = 0.025
BOOTSTRAP_UPPER_QUANTILE = 0.975
MC_SE_KEYS = ("var_mc_se", "es_mc_se", "var_mc_halfwidth95", "es_mc_halfwidth95")


def confidence(value: float) -> float:
    level = float(value)
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("Confidence must be strictly between zero and one.")
    return level


def _vector(values: np.ndarray) -> np.ndarray:
    vec = np.asarray(values, dtype=float)
    if vec.ndim != 1 or len(vec) == 0 or not np.isfinite(vec).all():
        raise ValueError("A nonempty finite vector is required; missing values are not dropped.")
    return vec


def _tail_mass(sample_size: int, level: float) -> float:
    mass = sample_size * (1 - confidence(level))
    # Remove roundoff at an integer boundary, not economic tail observations.
    nearest = round(mass)
    return (
        float(nearest)
        if nearest > 0 and abs(mass - nearest) < ROUND_BOUNDARY_TOLERANCE
        else mass
    )


def empirical_tail(losses: np.ndarray, c: float) -> dict:
    """Inverse empirical CDF and integral ES, including symmetric tied mass."""
    loss_vec, level = _vector(losses), confidence(c)
    sample_size = len(loss_vec)
    index = max(0, int(math.ceil(sample_size * level - QUANTILE_CEIL_EPSILON)) - 1)
    quantile = float(np.partition(loss_vec, index)[index])
    mass = _tail_mass(sample_size, level)
    above, boundary = loss_vec > quantile, loss_vec == quantile
    remaining = mass - int(above.sum())
    if (
        remaining < -REMAINING_MASS_TOLERANCE
        or remaining > int(boundary.sum()) + REMAINING_MASS_TOLERANCE
    ):
        raise RuntimeError("Empirical tail probability does not reconcile.")
    tail = above.astype(float)
    # Split any remaining fractional tail mass evenly across tied boundary observations.
    tail[boundary] = max(0.0, remaining) / int(boundary.sum())
    tail /= mass
    if not np.isclose(tail.sum(), 1.0, atol=WEIGHT_RECONCILIATION_ATOL, rtol=0):
        raise RuntimeError("ES tail weights must sum to one.")
    es = float(tail @ loss_vec)
    if es < quantile - QUANTILE_CEIL_EPSILON:
        raise RuntimeError("ES cannot be smaller than its loss VaR.")
    return {
        "var": quantile,
        "es": es,
        "tail_mass_observations": mass,
        "strict_exceedances_in_sample": int(above.sum()),
        "tail_weights": tail,
    }


def validate_weights(weights: np.ndarray, assets: int) -> np.ndarray:
    weight_vec = _vector(weights)
    if (
        len(weight_vec) != assets
        or (weight_vec < -QUANTILE_CEIL_EPSILON).any()
        or not np.isclose(weight_vec.sum(), 1.0, atol=WEIGHT_SUM_ATOL, rtol=0)
    ):
        raise ValueError(
            "These risk baselines require aligned, long-only, fully invested weights."
        )
    return weight_vec


def historical_risk(returns: np.ndarray, weights: np.ndarray, c: float) -> dict:
    return_mat = np.asarray(returns, dtype=float)
    if (
        return_mat.ndim != 2
        or len(return_mat) < MIN_HISTORICAL_ROWS
        or not np.isfinite(return_mat).all()
        or (return_mat < -1).any()
    ):
        raise ValueError(
            "Historical simple returns must be finite, aligned and at least -100%."
        )
    weight_vec = validate_weights(weights, return_mat.shape[1])
    position_losses = -return_mat * weight_vec
    answer = empirical_tail(position_losses.sum(axis=1), c)
    components = answer.pop("tail_weights") @ position_losses
    answer["es_components"] = components
    answer["contribution_residual"] = float(components.sum() - answer["es"])
    if abs(answer["contribution_residual"]) > WEIGHT_RECONCILIATION_ATOL:
        raise RuntimeError("Historical ES contributions do not reconcile.")
    return answer


def normal_parameters(
    mean: np.ndarray, covariance: np.ndarray, weights: np.ndarray
) -> tuple:
    mu = _vector(mean)
    cov = np.asarray(covariance, dtype=float)
    weight_vec = validate_weights(weights, len(mu))
    if cov.shape != (len(mu), len(mu)) or not np.isfinite(cov).all():
        raise ValueError("Covariance dimensions/values are invalid.")
    if not np.allclose(cov, cov.T, atol=SYMMETRY_ATOL, rtol=SYMMETRY_RTOL):
        raise ValueError("Covariance must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < EIGENVALUE_TOLERANCE:
        raise ValueError(
            "Covariance must be positive semidefinite; no hidden repair is applied."
        )
    variance = float(weight_vec @ cov @ weight_vec)
    if variance < EIGENVALUE_TOLERANCE:
        raise ValueError("Portfolio variance cannot be negative.")
    return float(weight_vec @ mu), math.sqrt(max(0.0, variance)), cov @ weight_vec


def gaussian_risk(
    mean: np.ndarray, covariance: np.ndarray, weights: np.ndarray, c: float
) -> dict:
    level = confidence(c)
    weight_vec, mu = np.asarray(weights, float), np.asarray(mean, float)
    mu_p, sigma, cov_w = normal_parameters(mu, covariance, weight_vec)
    z_score = float(norm.ppf(level))
    tail_factor = float(norm.pdf(z_score) / (1 - level))
    variance_components = (
        np.zeros_like(weight_vec) if sigma == 0 else weight_vec * cov_w / sigma
    )
    quantile = -mu_p + z_score * sigma
    es = -mu_p + tail_factor * sigma
    q_components = -weight_vec * mu + z_score * variance_components
    e_components = -weight_vec * mu + tail_factor * variance_components
    residual = float(e_components.sum() - es)
    if abs(residual) > WEIGHT_RECONCILIATION_ATOL:
        raise RuntimeError("Gaussian Euler contributions do not reconcile.")
    probability = float(norm.sf((1 + mu_p) / sigma)) if sigma > 0 else float(-mu_p > 1)
    return {
        "var": quantile,
        "es": es,
        "mean_return": mu_p,
        "standard_deviation": sigma,
        "var_components": q_components,
        "es_components": e_components,
        "contribution_residual": residual,
        "probability_loss_above_100pct": probability,
    }


def seeded_identifier(base_seed: int, *parts: object) -> int:
    key = "|".join([str(base_seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def _sorted_tail(ordered: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Empirical quantile/integral ES along last axis of already sorted draws."""
    sample_size = ordered.shape[-1]
    index = max(0, int(math.ceil(sample_size * c - QUANTILE_CEIL_EPSILON)) - 1)
    mass = _tail_mass(sample_size, c)
    full_count = int(math.floor(mass))
    fraction = mass - full_count
    total = np.zeros(ordered.shape[:-1], dtype=float)
    if full_count:
        total += ordered[..., -full_count:].sum(axis=-1)
    if fraction:
        total += fraction * ordered[..., -(full_count + 1)]
    return ordered[..., index], total / mass


def simulation_summary(draws: np.ndarray, levels: list[float]) -> dict:
    """Pooled empirical estimates; approximate SE from independent batch estimates."""
    draw_matrix = np.asarray(draws, dtype=float)
    if (
        draw_matrix.ndim != 2
        or draw_matrix.shape[0] < MIN_BATCHES
        or draw_matrix.shape[1] < MIN_DRAWS_PER_BATCH
        or not np.isfinite(draw_matrix).all()
    ):
        raise ValueError(
            "Simulation needs >=4 independent batches and >=20 finite draws per batch."
        )
    ordered = np.sort(draw_matrix, axis=1)
    pooled = np.sort(draw_matrix.reshape(-1))
    critical = float(t.ppf(CI95_T_QUANTILE, draw_matrix.shape[0] - 1))
    output = {}
    for raw in levels:
        level = confidence(raw)
        quantile, es = _sorted_tail(pooled, level)
        batch_q, batch_es = _sorted_tail(ordered, level)
        se_q = float(np.std(batch_q, ddof=1) / math.sqrt(len(batch_q)))
        se_e = float(np.std(batch_es, ddof=1) / math.sqrt(len(batch_es)))
        output[level] = {
            "var": float(quantile),
            "es": float(es),
            "var_mc_se": se_q,
            "es_mc_se": se_e,
            "var_mc_halfwidth95": critical * se_q,
            "es_mc_halfwidth95": critical * se_e,
            "draws": int(draw_matrix.size),
            "batches": int(draw_matrix.shape[0]),
            "draws_per_batch": int(draw_matrix.shape[1]),
            "tail_draw_mass": _tail_mass(draw_matrix.size, level),
        }
    return output


def normal_mc_coefficients(
    levels: list[float],
    seed: int,
    maximum_sigma: float,
    batches: int,
    initial: int,
    maximum: int,
    tolerance: float,
    design_variance_buffer: float = DEFAULT_DESIGN_VARIANCE_BUFFER,
) -> dict:
    """Simulate normal innovations shared across portfolios at a given origin.

    Each projection -mu + sigma*Z is an actual MC loss simulation. Independent
    batches estimate numerical error. Precision stopping gives approximate,
    not guaranteed sequential-confidence coverage.
    """
    if not np.isfinite(maximum_sigma) or maximum_sigma < 0 or tolerance <= 0:
        raise ValueError("Invalid MC precision inputs.")
    if batches < MIN_BATCHES or initial < MIN_DRAWS_PER_BATCH or maximum < initial:
        raise ValueError("Invalid simulation budget.")
    if design_variance_buffer < 1:
        raise ValueError("MC design variance buffer must be at least one.")
    # A normal-model pilot precision budget uses known asymptotic influence
    # variances. This chooses computational effort, never the investment model.
    # Actual draws and independent-batch error estimates still follow.
    required = _pilot_required_draws(
        levels, batches, maximum_sigma, tolerance, design_variance_buffer
    )
    size = initial
    while size * batches < required and size < maximum:
        size = min(maximum, size * 2)
    attempts = 0
    while True:
        rng = np.random.default_rng(seeded_identifier(seed, "normal", size))
        output = simulation_summary(rng.standard_normal((batches, size)), levels)
        half = maximum_sigma * max(
            max(item["var_mc_halfwidth95"], item["es_mc_halfwidth95"])
            for item in output.values()
        )
        attempts += 1
        reached = half <= tolerance
        if reached or size >= maximum:
            for value in output.values():
                value.update(
                    {"seed": seed, "precision_reached": reached, "attempts": attempts}
                )
            return output
        size = min(maximum, 2 * size)


def _pilot_required_draws(
    levels: list[float],
    batches: int,
    maximum_sigma: float,
    tolerance: float,
    design_variance_buffer: float,
) -> int:
    variances = []
    for raw_level in levels:
        level = confidence(raw_level)
        tail_prob = 1 - level
        z_score = norm.ppf(level)
        density = norm.pdf(z_score)
        first = density - z_score * tail_prob
        second = (1 + z_score * z_score) * tail_prob - z_score * density
        variances.extend(
            [
                level * tail_prob / (density * density),
                (second - first * first) / (tail_prob * tail_prob),
            ]
        )
    return math.ceil(
        design_variance_buffer
        * (t.ppf(CI95_T_QUANTILE, batches - 1) * maximum_sigma / tolerance) ** 2
        * max(variances)
    )


def projected_mc(coefficients: dict, mean_return: float, sigma: float) -> dict:
    result = dict(coefficients)
    for key in ("var", "es"):
        result[key] = -mean_return + sigma * coefficients[key]
    for key in MC_SE_KEYS:
        result[key] *= sigma
    return result


def multivariate_gaussian_check(
    mean: np.ndarray,
    covariance: np.ndarray,
    weights: np.ndarray,
    levels: list[float],
    seed: int,
    batches: int,
    batch_size: int,
) -> dict:
    """Independent asset-level implementation checks the linear projection route."""
    normal_parameters(mean, covariance, weights)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError(
            "This multivariate reference check requires positive-definite covariance."
        )
    factor = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)
    losses = np.empty((batches, batch_size))
    for batch_idx in range(batches):
        asset_returns = (
            np.asarray(mean)
            + rng.standard_normal((batch_size, len(mean))) @ factor.T
        )
        losses[batch_idx] = -asset_returns @ np.asarray(weights)
    return simulation_summary(losses, levels)


def circular_block_indices(
    n: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    if not 1 <= length <= n:
        raise ValueError("Block length must be between one and sample size.")
    starts = rng.integers(0, n, size=math.ceil(n / length))
    return ((starts[:, None] + np.arange(length)) % n).reshape(-1)[:n]


def historical_bootstrap(
    returns: np.ndarray,
    weights: np.ndarray,
    levels: list[float],
    length: int,
    repetitions: int,
    seed: int,
) -> list[dict]:
    if repetitions < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError("Bootstrap sensitivity needs at least 100 replicates.")
    # Row indices are shared across all assets: contemporaneous dependence survives.
    historical_risk(returns, weights, levels[0])
    rng = np.random.default_rng(seed)
    records = {level: [] for level in levels}
    for _ in range(repetitions):
        indices = circular_block_indices(len(returns), length, rng)
        losses = -np.asarray(returns)[indices] @ weights
        for level in levels:
            result = empirical_tail(losses, level)
            records[level].append([result["var"], result["es"]])
    rows = []
    for level, values in records.items():
        draws = np.asarray(values)
        rows.append(
            {
                "confidence": level,
                "block_length": length,
                "replicates": repetitions,
                "var_low": float(np.quantile(draws[:, 0], BOOTSTRAP_LOWER_QUANTILE)),
                "var_high": float(np.quantile(draws[:, 0], BOOTSTRAP_UPPER_QUANTILE)),
                "es_low": float(np.quantile(draws[:, 1], BOOTSTRAP_LOWER_QUANTILE)),
                "es_high": float(np.quantile(draws[:, 1], BOOTSTRAP_UPPER_QUANTILE)),
                "interval_type": (
                    "block_bootstrap_percentile_sensitivity_not_universal_coverage"
                ),
            }
        )
    return rows
