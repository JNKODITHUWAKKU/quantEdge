"""Fixed-threshold tail bootstrap with observed-path conditioning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.conditional import fixed_garch_forecast, garch_filter
from src.risk import circular_block_indices, seeded_identifier
from src.tails import hybrid_risk, threshold_grid

MIN_USABLE_REPLICATE_FRACTION = 0.9
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
INTERVAL_SCOPE_NOTE = (
    "Fixed percentile; filter/GPD refit; refilter observed origin history; "
    "selection uncertainty omitted"
)


def snapshot_uncertainty(
    training: pd.DataFrame,
    w: np.ndarray,
    family: str,
    fit: dict,
    cfg: dict,
    strategy: str,
) -> list[dict]:
    rng = np.random.default_rng(
        seeded_identifier(cfg["run"]["seed"], "tail_bootstrap", strategy, family)
    )
    levels = cfg["run"]["confidence_levels"]
    records = {level: [] for level in levels}
    failures = 0
    nonfinite = {level: 0 for level in levels}
    spec = dict(cfg["evt"])
    spec["threshold_probabilities"] = [fit["threshold_percentile"]]
    return_mat = training.to_numpy()
    observed = return_mat @ w
    sample_size = len(return_mat)
    reps = cfg["evt"]["bootstrap_replicates"]
    block_length = cfg["evt"]["bootstrap_block_length"]

    for _ in range(reps):
        draw = return_mat[circular_block_indices(sample_size, block_length, rng)] @ w
        replicate_input = _prepare_bootstrap_losses(draw, observed, family, cfg["filter"])
        if replicate_input is None:
            failures += 1
            continue
        losses, location, scale = replicate_input
        fitted = threshold_grid(losses, spec)[0]
        if fitted["fit_status"] != "OK":
            failures += 1
            continue
        for level in levels:
            result = hybrid_risk(losses, fitted, level)
            quantile = location + scale * result["var"]
            es = location + scale * result["es"]
            if not np.isfinite(es):
                nonfinite[level] += 1
            records[level].append([quantile, es, fitted["shape"]])

    return [
        _build_uncertainty_row(
            strategy=strategy,
            family=family,
            level=level,
            threshold_percentile=fit["threshold_percentile"],
            reps=reps,
            failures=failures,
            nonfinite_count=nonfinite[level],
            block_length=block_length,
            draws=np.array(records[level]),
        )
        for level in levels
    ]


def _prepare_bootstrap_losses(
    draw: np.ndarray,
    observed: np.ndarray,
    family: str,
    filter_cfg: dict,
) -> tuple[np.ndarray, float, float] | None:
    if family != "garch":
        return -draw, 0.0, 1.0
    mean = float(draw.mean())
    filtered = garch_filter(draw - mean, filter_cfg)
    if filtered["status"] != "OK":
        return None
    losses = -filtered["residuals"]
    location = -mean
    # Resampling must not replace the known current conditioning path.
    # Fit parameters/tail on each bootstrap history, then apply those
    # parameters to the ACTUAL origin-ending innovations.
    try:
        scale = fixed_garch_forecast(observed - mean, filtered)
    except ValueError:
        return None
    return losses, location, scale


def _build_uncertainty_row(
    strategy: str,
    family: str,
    level: float,
    threshold_percentile: float,
    reps: int,
    failures: int,
    nonfinite_count: int,
    block_length: int,
    draws: np.ndarray,
) -> dict:
    # Require at least 90% successful bootstrap fits before reporting percentile bands.
    usable = len(draws) >= MIN_USABLE_REPLICATE_FRACTION * reps
    row = {
        "strategy": strategy,
        "filter": family,
        "confidence": level,
        "threshold_percentile": threshold_percentile,
        "replicates": reps,
        "fit_failures": failures,
        "nonfinite_es_replicates": nonfinite_count,
        "block_length": block_length,
        "interval_available": usable,
        "interval_scope": INTERVAL_SCOPE_NOTE,
        "conditioning_history": "OBSERVED_ORIGIN_HISTORY_FIXED",
    }
    if usable:
        row.update(
            var_low=float(np.quantile(draws[:, 0], LOWER_QUANTILE)),
            var_high=float(np.quantile(draws[:, 0], UPPER_QUANTILE)),
            shape_low=float(np.quantile(draws[:, 2], LOWER_QUANTILE)),
            shape_high=float(np.quantile(draws[:, 2], UPPER_QUANTILE)),
        )
        if nonfinite_count == 0:
            row.update(
                es_low=float(np.quantile(draws[:, 1], LOWER_QUANTILE)),
                es_high=float(np.quantile(draws[:, 1], UPPER_QUANTILE)),
            )
    return row
