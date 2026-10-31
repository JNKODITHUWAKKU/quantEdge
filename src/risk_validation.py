"""Coverage, independence and joint risk scores with explicit sparse-data limits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlog1py, xlogy
from scipy.stats import binomtest, chi2

from src.risk import confidence

CONFIDENCE_LEVEL_95 = 0.95
BASEL_CONFIDENCE = 0.99
BASEL_CONFIDENCE_ATOL = 1e-12
BASEL_WINDOW = 250
BASEL_GREEN_MAX = 4
BASEL_AMBER_MAX = 9
MIN_SPARSE_COUNT = 5
FZ0_DOMAIN_TOLERANCE = 1e-12
FROZEN_PNL_TYPE = "frozen_exposure_market_loss"
REQUIRED_FORECAST_COLUMNS = {
    "realization_date",
    "forecast_origin",
    "confidence",
    "var",
    "es",
    "realized_hpl_loss",
    "pnl_type",
}


def _exceptions(values: np.ndarray) -> np.ndarray:
    binary = np.asarray(values)
    if binary.ndim != 1 or len(binary) == 0 or not np.isin(binary, [0, 1, False, True]).all():
        raise ValueError(
            "Exceptions must be a nonempty binary vector; missing days cannot be dropped."
        )
    return binary.astype(int)


def _bernoulli_ll(success: int, failure: int, probability: float) -> float:
    return float(xlogy(success, probability) + xlog1py(failure, -probability))


def kupiec(exceptions: np.ndarray, c: float) -> dict:
    binary, nominal_rate = _exceptions(exceptions), 1 - confidence(c)
    observations, count = len(binary), int(binary.sum())
    observed_rate = count / observations
    statistic = max(
        0.0,
        -2
        * (
            _bernoulli_ll(count, observations - count, nominal_rate)
            - _bernoulli_ll(count, observations - count, observed_rate)
        ),
    )
    exact = binomtest(count, observations, p=nominal_rate)
    interval = exact.proportion_ci(confidence_level=CONFIDENCE_LEVEL_95, method="exact")
    return {
        "observations": observations,
        "exceptions": count,
        "expected_exceptions": observations * nominal_rate,
        "nominal_exception_rate": nominal_rate,
        "observed_exception_rate": observed_rate,
        "kupiec_lr": statistic,
        "kupiec_pvalue": float(chi2.sf(statistic, 1)),
        "exact_binomial_pvalue": float(exact.pvalue),
        "exception_rate_ci95_low": float(interval.low),
        "exception_rate_ci95_high": float(interval.high),
        "coverage_note": (
            "Asymptotic LR and iid-binomial reference; clustering weakens iid inference."
        ),
    }


def christoffersen(
    exceptions: np.ndarray, consecutive: np.ndarray | None = None
) -> dict:
    binary = _exceptions(exceptions)
    pair_count = max(0, len(binary) - 1)
    mask = (
        np.ones(pair_count, dtype=bool)
        if consecutive is None
        else np.asarray(consecutive, dtype=bool)
    )
    if mask.shape != (pair_count,):
        raise ValueError("Transition eligibility must match adjacent observation pairs.")
    counts = np.zeros((2, 2), dtype=int)
    for previous, current in zip(binary[:-1][mask], binary[1:][mask]):
        counts[previous, current] += 1
    n00, n01, n10, n11 = map(int, counts.reshape(-1))
    output = {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "eligible_transitions": int(counts.sum()),
        "christoffersen_lr": None,
        "christoffersen_pvalue": None,
    }
    if not _has_all_transition_states(counts):
        output["independence_status"] = "INSUFFICIENT_TRANSITION_INFORMATION"
        return output
    p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
    pooled = (n01 + n11) / counts.sum()
    null = _bernoulli_ll(n01 + n11, n00 + n10, pooled)
    alternative = _bernoulli_ll(n01, n00, p01) + _bernoulli_ll(n11, n10, p11)
    statistic = max(0.0, -2 * (null - alternative))
    output.update(
        {
            "christoffersen_lr": statistic,
            "christoffersen_pvalue": float(chi2.sf(statistic, 1)),
            "independence_status": "SPARSE_TRANSITIONS_ASYMPTOTIC_CAUTION"
            if counts.min() < MIN_SPARSE_COUNT
            else "ASYMPTOTIC_REFERENCE",
            "p01": p01,
            "p11": p11,
        }
    )
    return output


def _has_all_transition_states(counts: np.ndarray) -> bool:
    return bool(
        counts.sum() > 0
        and not (counts.sum(axis=1) == 0).any()
        and not (counts.sum(axis=0) == 0).any()
    )


def basel_traffic_light(
    exceptions: np.ndarray, c: float, consecutive: bool = True
) -> dict:
    binary = _exceptions(exceptions)
    if not np.isclose(confidence(c), BASEL_CONFIDENCE, atol=BASEL_CONFIDENCE_ATOL, rtol=0):
        return {
            "basel_zone": "NOT_APPLICABLE",
            "basel_note": "Classic panel requires 99% one-day VaR.",
        }
    if len(binary) != BASEL_WINDOW or not consecutive:
        return {
            "basel_zone": "INSUFFICIENT_OR_NONCONTIGUOUS",
            "basel_note": "Requires exactly 250 consecutive forecasts.",
        }
    count = int(binary.sum())
    zone = (
        "GREEN"
        if count <= BASEL_GREEN_MAX
        else "AMBER"
        if count <= BASEL_AMBER_MAX
        else "RED"
    )
    return {
        "basel_zone": zone,
        "basel_exceptions": count,
        "basel_note": (
            "Classic 99%/250-observation diagnostic; no endowment regulatory certification."
        ),
    }


def quantile_loss(loss: np.ndarray, var: np.ndarray, c: float) -> np.ndarray:
    level = confidence(c)
    loss_vec, var_vec = np.asarray(loss, float), np.asarray(var, float)
    if (
        loss_vec.shape != var_vec.shape
        or not np.isfinite(loss_vec).all()
        or not np.isfinite(var_vec).all()
    ):
        raise ValueError("Score inputs must be aligned and finite.")
    return (level - (loss_vec < var_vec)) * (loss_vec - var_vec)


def joint_fz0(
    loss: np.ndarray, var: np.ndarray, es: np.ndarray, c: float
) -> np.ndarray:
    """Patton/Ziegel/Chen FZ0 mapped from negative returns to positive losses.

    Domain: ES > 0 and ES >= VaR. Invalid domain is missing, never clipped.
    """
    level = confidence(c)
    loss_vec, var_vec, es_vec = (
        np.asarray(loss, float),
        np.asarray(var, float),
        np.asarray(es, float),
    )
    if (
        loss_vec.shape != var_vec.shape
        or var_vec.shape != es_vec.shape
        or not np.isfinite(loss_vec).all()
        or not np.isfinite(var_vec).all()
        or not np.isfinite(es_vec).all()
    ):
        raise ValueError("Joint score inputs must be aligned and finite.")
    output = np.full(loss_vec.shape, np.nan)
    valid = (es_vec > 0) & (es_vec >= var_vec - FZ0_DOMAIN_TOLERANCE)
    output[valid] = (
        (var_vec[valid] + np.maximum(loss_vec[valid] - var_vec[valid], 0) / (1 - level))
        / es_vec[valid]
        + np.log(es_vec[valid])
        - 1
    )
    return output


def evaluate_forecasts(frame: pd.DataFrame) -> dict:
    data, dates, origins = _validate_evaluation_frame(frame)
    loss_vec, var_vec, es_vec = (
        data.realized_hpl_loss.to_numpy(float),
        data["var"].to_numpy(float),
        data.es.to_numpy(float),
    )
    if not np.isfinite(np.column_stack([loss_vec, var_vec, es_vec])).all():
        raise ValueError(
            "Missing or failed forecasts must be reported, not silently excluded."
        )
    level = float(data.confidence.iloc[0])
    exceptions = (loss_vec > var_vec).astype(int)
    # Only count transitions where the previous realization date is this forecast's origin.
    adjacent = origins[1:] == dates[:-1]
    answer = {**kupiec(exceptions, level), **christoffersen(exceptions, adjacent)}
    q_scores = quantile_loss(loss_vec, var_vec, level)
    fz0_scores = joint_fz0(loss_vec, var_vec, es_vec, level)
    answer.update(
        {
            "confidence": level,
            "window_start": str(pd.Timestamp(dates[0]).date()),
            "window_end": str(pd.Timestamp(dates[-1]).date()),
            "mean_quantile_loss": float(q_scores.mean()),
            "mean_fz0": (
                float(np.nanmean(fz0_scores)) if np.isfinite(fz0_scores).any() else None
            ),
            "fz0_valid_observations": int(np.isfinite(fz0_scores).sum()),
            "forecast_gaps": int((~adjacent).sum()),
            "interpretation": (
                "Development diagnostic; p-values are not model-selection rankings."
            ),
        }
    )
    if len(data) >= BASEL_WINDOW:
        last = exceptions[-BASEL_WINDOW:]
        contig = bool(
            (origins[-(BASEL_WINDOW - 1) :] == dates[-BASEL_WINDOW:-1]).all()
        )
        answer.update(basel_traffic_light(last, level, contig))
        answer.update(
            {
                "basel_window_start": str(pd.Timestamp(dates[-BASEL_WINDOW]).date()),
                "basel_window_end": str(pd.Timestamp(dates[-1]).date()),
            }
        )
    else:
        answer.update(basel_traffic_light(exceptions, level, bool(adjacent.all())))
    if answer["christoffersen_lr"] is not None:
        cc_lr = answer["kupiec_lr"] + answer["christoffersen_lr"]
        answer.update(
            {
                "conditional_coverage_lr": cc_lr,
                "conditional_coverage_pvalue": float(chi2.sf(cc_lr, 2)),
                "conditional_coverage_note": (
                    "Usual asymptotic sum; coverage uses n, independence n-1 eligible transitions."
                ),
            }
        )
    return answer


def _validate_evaluation_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not REQUIRED_FORECAST_COLUMNS.issubset(frame.columns) or len(frame) == 0:
        raise ValueError("A nonempty matched forecast/outcome ledger is required.")
    data = frame.sort_values("realization_date")
    if data.realization_date.duplicated().any() or data.confidence.nunique() != 1:
        raise ValueError("Evaluate one method, portfolio and confidence at a time.")
    if data.pnl_type.nunique() != 1 or data.pnl_type.iloc[0] != FROZEN_PNL_TYPE:
        raise ValueError("Coverage requires the matching frozen-exposure market P&L.")
    if "method" in data and data.method.nunique() != 1:
        raise ValueError("Do not mix methods in a coverage test.")
    if "strategy" in data and data.strategy.nunique() != 1:
        raise ValueError("Do not mix strategies in a coverage test.")
    dates = pd.to_datetime(data.realization_date).to_numpy()
    origins = pd.to_datetime(data.forecast_origin).to_numpy()
    if not (origins < dates).all():
        raise ValueError("Forecast origins must precede outcomes.")
    return data, dates, origins
