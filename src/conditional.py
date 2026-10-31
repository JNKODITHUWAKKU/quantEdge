"""Past-window volatility filters and matched conditional loss distributions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.risk import empirical_tail, validate_weights
from src.tails import choose_threshold, hybrid_risk, threshold_grid

METHODS = [
    "ewma_normal",
    "ewma_fhs",
    "garch_normal",
    "garch_t",
    "garch_fhs",
    "raw_pot_guarded",
    "garch_pot_guarded",
]
DEFAULT_EWMA_DECAY = 0.94
DEFAULT_BURN_SESSIONS = 60
MIN_BURN_SESSIONS = 2
MIN_EWMA_VARIANCE = 1e-16
PERCENT_SCALE = 100.0
VARIANCE_SCALE = 10_000.0
PERSISTENCE_TOLERANCE = 1e-7
RECURRENCE_TOLERANCE = 1e-10
NU_LOWER_BOUND = 2.0
NU_NEAR_LOWER = 2.06
NU_NEAR_UPPER = 499.0
MIN_FIXED_HISTORY = 100
FHS_RECONCILIATION_ATOL = 1e-11
LJUNG_BOX_LAGS = [10]
EXTREMOGRAM_QUANTILE = 0.95
EXTREMOGRAM_LAGS = (1, 2, 5, 10, 20)
GARCH_METHODS = ("garch_normal", "garch_t", "garch_fhs")


def ewma_filter(
    innovations: np.ndarray,
    decay: float = DEFAULT_EWMA_DECAY,
    burn: int = DEFAULT_BURN_SESSIONS,
) -> dict:
    series = np.asarray(innovations, float)
    if (
        series.ndim != 1
        or len(series) <= burn
        or not np.isfinite(series).all()
        or not 0 < decay < 1
        or burn < MIN_BURN_SESSIONS
    ):
        raise ValueError("Invalid EWMA series, initialization or decay.")
    variance = float(np.mean(series[:burn] ** 2))
    if variance <= MIN_EWMA_VARIANCE:
        raise ValueError("EWMA initialization has no usable variance.")
    sigma = np.full(len(series), np.nan)
    for idx in range(burn, len(series)):
        # Record sigma_t before updating with innovation_t so filter variance stays strictly pre-outcome.
        sigma[idx] = np.sqrt(variance)
        variance = decay * variance + (1 - decay) * series[idx] ** 2
    return {
        "status": "OK",
        "conditional_sigma": sigma,
        "next_sigma": float(np.sqrt(variance)),
        "residuals": series[burn:] / sigma[burn:],
        "burn": burn,
        "decay": decay,
    }


def _build_student_garch_model(innovations: np.ndarray):
    from arch import arch_model

    # Fit on percentage returns (x 100) for optimizer conditioning; variance outputs are divided by 10,000.
    return arch_model(
        PERCENT_SCALE * innovations,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="StudentsT",
        rescale=False,
    )


def garch_filter(innovations: np.ndarray, settings: dict) -> dict:
    series = np.asarray(innovations, float)
    burn = settings["initialization_sessions"]
    if series.ndim != 1 or len(series) <= burn or not np.isfinite(series).all():
        raise ValueError("Invalid GARCH input series.")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = _build_student_garch_model(series)
            result = model.fit(
                disp="off",
                show_warning=False,
                tol=settings["garch_tolerance"],
                options={"maxiter": settings["garch_max_iterations"]},
            )
        pars = result.params.to_dict()
        alpha = pars["alpha[1]"]
        beta = pars["beta[1]"]
        nu = pars["nu"]
        cond_var = np.asarray(result.conditional_volatility, float) ** 2 / VARIANCE_SCALE
        next_var = (
            float(result.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
            / VARIANCE_SCALE
        )
        recurrence = (
            pars["omega"] / VARIANCE_SCALE + alpha * series[-1] ** 2 + beta * cond_var[-1]
        )
        if (
            result.convergence_flag != 0
            or not np.isfinite(cond_var).all()
            or np.min(cond_var) <= 0
            or not np.isfinite(next_var)
        ):
            raise RuntimeError("Convergence or conditional variance check failed.")
        if (
            alpha < 0
            or beta < 0
            or alpha + beta > 1 + PERSISTENCE_TOLERANCE
            or nu <= NU_LOWER_BOUND
            or abs(next_var - recurrence) > RECURRENCE_TOLERANCE
        ):
            raise RuntimeError("GARCH parameter/one-step recurrence check failed.")
        return {
            "status": "OK",
            "conditional_sigma": np.sqrt(cond_var),
            "next_sigma": np.sqrt(next_var),
            "residuals": series[burn:] / np.sqrt(cond_var[burn:]),
            "burn": burn,
            "nu": nu,
            "omega_return_units": pars["omega"] / VARIANCE_SCALE,
            "alpha": alpha,
            "beta": beta,
            "persistence": alpha + beta,
            "near_integrated": bool(
                alpha + beta >= settings["garch_near_integration"]
            ),
            "nu_near_boundary": bool(nu < NU_NEAR_LOWER or nu > NU_NEAR_UPPER),
            "convergence_flag": int(result.convergence_flag),
            "iterations": int(result.optimization_result.nit),
            "warnings": " | ".join(str(item.message) for item in caught),
        }
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        return {"status": "FIT_FAILED", "reason": str(exc)}


def student_risk(mean: float, sigma: float, nu: float, c: float) -> dict:
    if (
        nu <= NU_LOWER_BOUND
        or sigma < 0
        or not np.isfinite([mean, sigma, nu, c]).all()
        or not 0 < c < 1
    ):
        raise ValueError("Standardized Student-t needs nu>2 and valid risk inputs.")
    quantile = t.ppf(c, nu)
    # Standard Student-t has variance nu/(nu-2); scale by sqrt((nu-2)/nu) to match arch's unit-variance innovations.
    unit = np.sqrt((nu - 2) / nu)
    return {
        "var": float(-mean + sigma * unit * quantile),
        "es": float(
            -mean
            + sigma * unit * (nu + quantile * quantile) / (nu - 1) * t.pdf(quantile, nu) / (1 - c)
        ),
    }


def fixed_garch_forecast(innovations: np.ndarray, parameters: dict) -> float:
    """Refilter the observed history using estimated parameters, without refitting."""
    series = np.asarray(innovations, float)
    if (
        series.ndim != 1
        or len(series) < MIN_FIXED_HISTORY
        or not np.isfinite(series).all()
    ):
        raise ValueError("A finite observed origin history is required.")
    pars = [
        parameters["omega_return_units"] * VARIANCE_SCALE,
        parameters["alpha"],
        parameters["beta"],
        parameters["nu"],
    ]
    fixed = _build_student_garch_model(series).fix(pars)
    variance = (
        float(fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        / VARIANCE_SCALE
    )
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Invalid fixed-state forecast variance.")
    return float(np.sqrt(variance))


def fhs_risk(
    asset_returns: np.ndarray,
    w: np.ndarray,
    asset_mean: np.ndarray,
    filtered: dict,
    c: float,
) -> dict:
    start = filtered["burn"]
    ratio = filtered["next_sigma"] / filtered["conditional_sigma"][start:]
    scenarios = asset_mean + (asset_returns[start:] - asset_mean) * ratio[:, None]
    position_loss = -scenarios * w
    risk = empirical_tail(position_loss.sum(axis=1), c)
    components = risk.pop("tail_weights") @ position_loss
    if not np.isclose(components.sum(), risk["es"], atol=FHS_RECONCILIATION_ATOL, rtol=0):
        raise RuntimeError("FHS components do not reconcile.")
    return {**risk, "components": components}


def residual_diagnostics(residuals: np.ndarray) -> tuple[dict, list[dict]]:
    res_vec = np.asarray(residuals, float)
    lb = acorr_ljungbox(res_vec, lags=LJUNG_BOX_LAGS, return_df=True)
    sq = acorr_ljungbox(res_vec * res_vec, lags=LJUNG_BOX_LAGS, return_df=True)
    cutoff = empirical_tail(res_vec, EXTREMOGRAM_QUANTILE)["var"]
    exceeded = res_vec > cutoff
    rate = float(exceeded.mean())
    rows = []
    for lag in EXTREMOGRAM_LAGS:
        previous = exceeded[:-lag]
        events = int(previous.sum())
        rows.append(
            {
                "lag": lag,
                "conditioning_events": events,
                "conditional_exceedance_rate": (
                    float(exceeded[lag:][previous].mean()) if events else np.nan
                ),
                "unconditional_exceedance_rate": rate,
            }
        )
    return {
        "residual_mean": float(res_vec.mean()),
        "residual_std": float(res_vec.std(ddof=1)),
        "ljung_box_p_lag10": float(lb.lb_pvalue.iloc[0]),
        "squared_ljung_box_p_lag10": float(sq.lb_pvalue.iloc[0]),
        "diagnostic_note": (
            "In-sample fitted residuals; exploratory asymptotic references, not proof of iid."
        ),
    }, rows


def forecast_portfolio(
    training: pd.DataFrame,
    origin: pd.Timestamp,
    exposure: dict,
    strategy: str,
    cfg: dict,
    diagnostics: bool = False,
) -> tuple:
    return_mat, weights = _validate_training_window(training, origin, exposure, cfg)
    asset_mean = return_mat.mean(axis=0)
    mu = float(asset_mean @ weights)
    centered = return_mat @ weights - mu
    ew = ewma_filter(
        centered, cfg["filter"]["ewma_lambda"], cfg["filter"]["initialization_sessions"]
    )
    ga = garch_filter(centered, cfg["filter"])
    losses = {"raw": -return_mat @ weights}
    if ga["status"] == "OK":
        losses["garch"] = -ga["residuals"]
    grids = {name: threshold_grid(series, cfg["evt"]) for name, series in losses.items()}
    selections = {
        name: choose_threshold(grid, cfg["evt"]) for name, grid in grids.items()
    }
    tail_rows = _build_tail_rows(grids, selections, strategy, origin)
    common = _build_common_metadata(
        training, origin, exposure, strategy, weights, mu
    )

    records, components = [], []
    for level in cfg["run"]["confidence_levels"]:
        level_records, level_components = _forecast_at_confidence(
            return_mat=return_mat,
            weights=weights,
            assets=list(training.columns),
            asset_mean=asset_mean,
            mu=mu,
            ew=ew,
            ga=ga,
            losses=losses,
            selections=selections,
            common=common,
            origin_value_usd=exposure["origin_value_usd"],
            level=level,
        )
        records.extend(level_records)
        components.extend(level_components)

    filter_rows, extremograms = _collect_filter_diagnostics(
        ew, ga, losses["raw"], strategy, origin, diagnostics
    )
    return (
        records,
        components,
        tail_rows,
        filter_rows,
        extremograms,
        {
            "losses": losses,
            "filters": {"ewma": ew, "garch": ga},
            "selections": selections,
        },
    )


def _validate_training_window(
    training: pd.DataFrame, origin: pd.Timestamp, exposure: dict, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    if training.index[-1] != origin or len(training) != cfg["run"]["lookback_returns"]:
        raise ValueError("Exact origin-ending training window required.")
    if training.index.has_duplicates or not training.index.is_monotonic_increasing:
        raise ValueError("Training dates must be unique and ordered.")
    return_mat = training.to_numpy(float)
    weights = validate_weights(exposure["weights"], return_mat.shape[1])
    if not np.isfinite(return_mat).all():
        raise ValueError("Missing training returns.")
    return return_mat, weights


def _build_tail_rows(
    grids: dict[str, list[dict]],
    selections: dict[str, dict],
    strategy: str,
    origin: pd.Timestamp,
) -> list[dict]:
    tail_rows = []
    for family, grid in grids.items():
        selection = selections[family]
        for row in grid:
            tail_rows.append(
                {
                    "strategy": strategy,
                    "forecast_origin": origin,
                    "filter": family,
                    **row,
                    "selection_status": selection["selection_status"],
                    "selected": selection["selected"] is row,
                }
            )
    return tail_rows


def _build_common_metadata(
    training: pd.DataFrame,
    origin: pd.Timestamp,
    exposure: dict,
    strategy: str,
    weights: np.ndarray,
    mu: float,
) -> dict:
    return {
        "strategy": strategy,
        "forecast_origin": origin,
        "realization_date": exposure["realization_date"],
        "training_start": training.index[0],
        "training_end": training.index[-1],
        "training_observations": len(training),
        "pnl_type": "frozen_exposure_market_loss",
        "horizon_sessions": 1,
        "origin_value_usd": float(exposure["origin_value_usd"]),
        "mean_return": mu,
        "evidence_class": "exploratory_development_only",
        "var_mc_halfwidth95": 0.0,
        **{f"weight_{asset}": float(val) for asset, val in zip(training.columns, weights)},
    }


def _forecast_at_confidence(
    return_mat: np.ndarray,
    weights: np.ndarray,
    assets: list[str],
    asset_mean: np.ndarray,
    mu: float,
    ew: dict,
    ga: dict,
    losses: dict[str, np.ndarray],
    selections: dict[str, dict],
    common: dict,
    origin_value_usd: float,
    level: float,
) -> tuple[list[dict], list[dict]]:
    ef = fhs_risk(return_mat, weights, asset_mean, ew, level)
    gf = fhs_risk(return_mat, weights, asset_mean, ga, level) if ga["status"] == "OK" else ef
    z_score = norm.ppf(level)
    tail_factor = norm.pdf(z_score) / (1 - level)
    ewma_gaussian = {
        "var": -mu + ew["next_sigma"] * z_score,
        "es": -mu + ew["next_sigma"] * tail_factor,
    }
    answers = {
        "ewma_normal": (ewma_gaussian, "ewma_normal", "OK", ""),
        "ewma_fhs": (ef, "ewma_fhs", "OK", ""),
    }
    if ga["status"] == "OK":
        garch_gaussian = {
            "var": -mu + ga["next_sigma"] * z_score,
            "es": -mu + ga["next_sigma"] * tail_factor,
        }
        answers.update(
            garch_normal=(garch_gaussian, "garch_normal", "OK", ""),
            garch_t=(
                student_risk(mu, ga["next_sigma"], ga["nu"], level),
                "garch_t",
                "OK",
                "",
            ),
            garch_fhs=(gf, "garch_fhs", "OK", ""),
        )
    else:
        for name in GARCH_METHODS:
            answers[name] = (ef, "ewma_fhs", "FALLBACK", ga["reason"])

    for family, method, fallback_risk, actual in [
        ("raw", "raw_pot_guarded", empirical_tail(losses["raw"], level), "historical"),
        (
            "garch",
            "garch_pot_guarded",
            gf,
            "garch_fhs" if ga["status"] == "OK" else "ewma_fhs",
        ),
    ]:
        answers[method] = _resolve_pot_forecast(
            family=family,
            method=method,
            fallback_risk=fallback_risk,
            actual=actual,
            losses=losses,
            selections=selections,
            mu=mu,
            ga=ga,
            level=level,
        )

    records = []
    for method in METHODS:
        result, actual, status, reason = answers[method]
        records.append(
            {
                **common,
                "confidence": level,
                "method": method,
                "actual_model_used": actual,
                "fit_status": status,
                "fallback_reason": reason,
                "var": float(result["var"]),
                "es": float(result["es"]),
                "var_usd": float(result["var"] * origin_value_usd),
                "es_usd": float(result["es"] * origin_value_usd),
                "quantile_segment": result.get("quantile_segment", ""),
                "threshold_percentile": result.get("threshold_percentile", np.nan),
                "tail_shape": result.get("tail_shape", np.nan),
                "forecast_above_100pct_loss": bool(
                    result["var"] > 1 or result["es"] > 1
                ),
            }
        )

    components = []
    for method, result in [("ewma_fhs", ef), ("garch_fhs", gf)]:
        for asset, value in zip(assets, result["components"]):
            components.append(
                {
                    "strategy": common["strategy"],
                    "forecast_origin": common["forecast_origin"],
                    "realization_date": common["realization_date"],
                    "method": method,
                    "actual_filter": (
                        "ewma" if method == "ewma_fhs" or ga["status"] != "OK" else "garch"
                    ),
                    "confidence": level,
                    "asset": asset,
                    "es_component": float(value),
                    "portfolio_es": result["es"],
                }
            )
    return records, components


def _resolve_pot_forecast(
    family: str,
    method: str,
    fallback_risk: dict,
    actual: str,
    losses: dict[str, np.ndarray],
    selections: dict[str, dict],
    mu: float,
    ga: dict,
    level: float,
) -> tuple[dict, str, str, str]:
    selection = selections.get(
        family, {"selected": None, "selection_status": "FILTER_FAILED"}
    )
    fit = selection["selected"]
    if fit is None:
        return fallback_risk, actual, "FALLBACK", selection["selection_status"]
    tail = hybrid_risk(losses[family], fit, level)
    mean, scale = (0.0, 1.0) if family == "raw" else (-mu, ga["next_sigma"])
    estimate = {
        **tail,
        "var": mean + scale * tail["var"],
        "es": mean + scale * tail["es"],
        "threshold_percentile": fit["threshold_percentile"],
        "tail_shape": fit["shape"],
    }
    if not np.isfinite(estimate["es"]) or estimate["es"] > 1 or estimate["var"] > 1:
        return fallback_risk, actual, "FALLBACK", "NONFINITE_OR_ECONOMIC_SUPPORT"
    return estimate, method, "OK", ""


def _collect_filter_diagnostics(
    ew: dict,
    ga: dict,
    raw_losses: np.ndarray,
    strategy: str,
    origin: pd.Timestamp,
    diagnostics: bool,
) -> tuple[list[dict], list[dict]]:
    filter_rows, extremograms = [], []
    for name, filtered in [("ewma", ew), ("garch", ga)]:
        small = {
            key: val
            for key, val in filtered.items()
            if key not in ("conditional_sigma", "residuals")
        }
        if diagnostics and filtered["status"] == "OK":
            diagnostics_row, extreme = residual_diagnostics(-filtered["residuals"])
            small.update(diagnostics_row)
            extremograms.extend(
                {"strategy": strategy, "forecast_origin": origin, "filter": name, **row}
                for row in extreme
            )
        filter_rows.append(
            {"strategy": strategy, "forecast_origin": origin, "filter": name, **small}
        )
    if diagnostics:
        diag_summary, extreme_rows = residual_diagnostics(raw_losses)
        filter_rows.append(
            {
                "strategy": strategy,
                "forecast_origin": origin,
                "filter": "raw",
                "status": "OK",
                **diag_summary,
            }
        )
        extremograms.extend(
            {"strategy": strategy, "forecast_origin": origin, "filter": "raw", **row}
            for row in extreme_rows
        )
    return filter_rows, extremograms
