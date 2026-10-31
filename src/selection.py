"""Predeclared paired validation comparisons; no access to final-period prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk import circular_block_indices, seeded_identifier

MIN_BOOTSTRAP_REPLICATES = 99
DEFAULT_COVERAGE = 0.95
DEGENERATE_SE_FLOOR = 1e-14
DEGENERATE_MEAN_SCALE = 1e-12
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
ES_CALIBRATION_CONFIDENCES = (0.95, 0.99)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Holm requires finite p-values in [0,1].")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def bootstrap_means(matrix: np.ndarray, block: int, replicates: int, seed: int) -> np.ndarray:
    x = np.asarray(matrix, float)
    if (
        x.ndim != 2
        or len(x) < block
        or block < 1
        or replicates < MIN_BOOTSTRAP_REPLICATES
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid paired block-bootstrap inputs.")
    rng = np.random.default_rng(seed)
    return np.array(
        [x[circular_block_indices(len(x), block, rng)].mean(axis=0) for _ in range(replicates)]
    )


def _initialize_interval_rows(
    matrix: np.ndarray, valid: np.ndarray, block: int, replicates: int, coverage: float
) -> list[dict]:
    return [
        {
            "column": col_idx,
            "mean_difference": float(matrix[:, col_idx].mean()) if valid[col_idx] else None,
            "bootstrap_se": None,
            "ci_low": None,
            "ci_high": None,
            "status": "INVALID_SCORE_DOMAIN",
            "block_length": block,
            "replicates": replicates,
            "coverage": coverage,
            "family_size": matrix.shape[1],
            "max_statistic_critical": None,
        }
        for col_idx in range(matrix.shape[1])
    ]


def simultaneous_mean_intervals(
    matrix: np.ndarray,
    block: int,
    replicates: int,
    seed: int,
    coverage: float = DEFAULT_COVERAGE,
) -> list[dict]:
    """Single-step centered max-|standardized mean| intervals, not RW stepdown."""
    x = np.asarray(matrix, float)
    if x.ndim != 2 or not 0 < coverage < 1:
        raise ValueError("Invalid interval input.")
    valid = np.isfinite(x).all(axis=0)
    rows = _initialize_interval_rows(x, valid, block, replicates, coverage)
    if not valid.any():
        return rows
    means = x[:, valid].mean(axis=0)
    draws = bootstrap_means(x[:, valid], block, replicates, seed)
    se = draws.std(axis=0, ddof=1)
    active = se > np.maximum(DEGENERATE_SE_FLOOR, np.abs(means) * DEGENERATE_MEAN_SCALE)
    critical = (
        float(
            np.quantile(
                np.max(np.abs((draws[:, active] - means[active]) / se[active]), axis=1),
                coverage,
            )
        )
        if active.any()
        else None
    )
    for valid_idx, col_idx in enumerate(np.flatnonzero(valid)):
        rows[col_idx]["bootstrap_se"] = float(se[valid_idx])
        rows[col_idx]["status"] = "OK" if active[valid_idx] else "DEGENERATE_NO_ADOPTION_EVIDENCE"
        if active[valid_idx]:
            rows[col_idx].update(
                ci_low=float(means[valid_idx] - critical * se[valid_idx]),
                ci_high=float(means[valid_idx] + critical * se[valid_idx]),
                max_statistic_critical=critical,
            )
    return rows


def causal_states(returns: pd.DataFrame, book: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spec = cfg["risk_selection"]
    vol = returns.SPY.rolling(spec["state_volatility_window"]).std(ddof=1)
    # Shift by 1 session before rolling median so state classification is strictly pre-origin.
    median = vol.shift(1).rolling(spec["state_median_window"]).median()
    origins = pd.DatetimeIndex(book.forecast_origin)
    current = vol.reindex(origins).to_numpy()
    reference = median.reindex(origins).to_numpy()
    if not np.isfinite(np.c_[current, reference]).all():
        raise ValueError("Incomplete past-only volatility state.")
    return pd.DataFrame(
        {
            "forecast_origin": origins,
            "realization_date": pd.to_datetime(book.realization_date).to_numpy(),
            "spy_trailing_volatility": current,
            "past_median_volatility": reference,
            "volatility_state": np.where(current > reference, "HIGH", "LOW"),
        }
    )


def _compute_es_moment_vectors(
    forecasts: pd.DataFrame, candidate_order: list[str]
) -> tuple[list[tuple], np.ndarray]:
    keys = []
    vectors = []
    confidence_mask = np.isclose(forecasts.confidence, ES_CALIBRATION_CONFIDENCES[0]) | np.isclose(
        forecasts.confidence, ES_CALIBRATION_CONFIDENCES[1]
    )
    selected = forecasts[forecasts.method.isin(candidate_order) & confidence_mask]
    for key, group in selected.groupby(["strategy", "method", "confidence"], sort=True):
        group = group.sort_values("realization_date")
        confidence = key[2]
        realized_loss = group.realized_hpl_loss.to_numpy()
        var_forecast = group["var"].to_numpy()
        es_forecast = group.es.to_numpy()
        moment = (
            var_forecast + np.maximum(realized_loss - var_forecast, 0) / (1 - confidence) - es_forecast
        ) / es_forecast
        keys.append(key)
        vectors.append(moment)
    return keys, np.column_stack(vectors)


def es_calibration(forecasts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spec = cfg["calibration_diagnostics"]
    keys, matrix = _compute_es_moment_vectors(forecasts, cfg["risk_selection"]["candidate_order"])
    valid = np.isfinite(matrix).all(axis=0)
    draws = (
        bootstrap_means(
            matrix[:, valid],
            spec["es_moment_block_length"],
            spec["es_moment_bootstrap_replicates"],
            seeded_identifier(cfg["run"]["seed"], "es_moment"),
        )
        if valid.any()
        else np.empty((0, 0))
    )
    rows = []
    valid_idx = 0
    for col_idx, (strategy, method, confidence) in enumerate(keys):
        row = {
            "strategy": strategy,
            "method": method,
            "confidence": confidence,
            "observations": len(matrix),
            "status": "OK_POINTWISE_DIAGNOSTIC" if valid[col_idx] else "INVALID_DOMAIN",
            "mean_normalized_es_moment": float(matrix[:, col_idx].mean())
            if valid[col_idx]
            else np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "block_length": spec["es_moment_block_length"],
            "replicates": spec["es_moment_bootstrap_replicates"],
            "selection_use": "NONE",
        }
        if valid[col_idx]:
            row.update(
                ci95_low=float(np.quantile(draws[:, valid_idx], LOWER_QUANTILE)),
                ci95_high=float(np.quantile(draws[:, valid_idx], UPPER_QUANTILE)),
            )
            valid_idx += 1
        rows.append(row)
    return pd.DataFrame(rows)
