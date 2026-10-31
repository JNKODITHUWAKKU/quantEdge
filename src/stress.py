"""Counterfactual buy-and-hold replay and explicit non-probabilistic scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk import validate_weights

DEFAULT_STRESS_VALUE = 1e7
ATTRIBUTION_ATOL = 1e-7
ATTRIBUTION_RTOL = 1e-12
EIGENVALUE_TOLERANCE = -1e-12
DEVELOPMENT_END_DATE = pd.Timestamp("2014-12-31")
LOOKBACK_SESSIONS = 1260
ROLLING_SHOCK_WINDOW = 20

WINDOWS = [
    {
        "scenario": "GFC_2008",
        "start": "2008-01-01",
        "end": "2008-12-31",
        "mechanism": "Equity/credit contagion and funding stress",
    },
    {
        "scenario": "GFC_extended",
        "start": "2007-01-01",
        "end": "2009-12-31",
        "mechanism": "Pre-crisis peak, crisis and partial recovery",
    },
    {
        "scenario": "COVID_March_2020",
        "start": "2020-03-01",
        "end": "2020-03-31",
        "mechanism": "Equity/energy shock and liquidity pressure",
    },
    {
        "scenario": "COVID_extended",
        "start": "2020-02-01",
        "end": "2020-04-30",
        "mechanism": "Onset, crash and early rebound",
    },
    {
        "scenario": "Rates_2022",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "mechanism": "Rates, duration and equity/bond joint repricing",
    },
]


def _validate_replay_inputs(
    prices: pd.DataFrame, start: str, end: str, value: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if value <= 0 or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Replay needs complete positive prices and capital.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Replay prices must have unique ascending dates.")
    window_prices = prices.loc[start:end]
    prior_prices = prices.loc[prices.index < pd.Timestamp(start)]
    if len(window_prices) < 2 or len(prior_prices) == 0 or prices.index[-1] < pd.Timestamp(end):
        raise ValueError(
            "Replay needs a prior close, complete requested end coverage and at least two observations."
        )
    return window_prices, prior_prices


def _longest_underwater_run(underwater: np.ndarray) -> int:
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def replay(
    prices: pd.DataFrame,
    weights: np.ndarray,
    start: str,
    end: str,
    value: float = DEFAULT_STRESS_VALUE,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    w = validate_weights(weights, prices.shape[1])
    window_prices, prior_prices = _validate_replay_inputs(prices, start, end, value)
    base = prior_prices.iloc[-1]
    base_date = prior_prices.index[-1]
    # Anchor buy-and-hold shares to the last close strictly before the stress window starts.
    levels = window_prices.divide(base, axis=1)
    wealth = levels @ w * value
    full = np.r_[value, wealth.to_numpy()]
    dates = pd.DatetimeIndex([base_date, *window_prices.index])
    peaks = np.maximum.accumulate(full)
    drawdowns = 1 - full / peaks
    trough = int(np.argmax(drawdowns))
    peak = int(np.argmax(full[: trough + 1]))
    recovery = np.flatnonzero(full[trough + 1 :] >= full[peak])
    no_drawdown = bool(drawdowns[trough] == 0)
    recovered = int(trough + 1 + recovery[0]) if len(recovery) and not no_drawdown else None
    longest = _longest_underwater_run(full < peaks)
    components = -w * (levels.iloc[-1].to_numpy() - 1) * value
    total = float(value - wealth.iloc[-1])
    if not np.isclose(components.sum(), total, atol=ATTRIBUTION_ATOL, rtol=ATTRIBUTION_RTOL):
        raise RuntimeError("Buy-and-hold terminal attribution failed.")
    summary = {
        "requested_start": start,
        "requested_end": end,
        "base_date": str(base_date.date()),
        "first_return_date": str(window_prices.index[0].date()),
        "last_date": str(window_prices.index[-1].date()),
        "initial_value_usd": value,
        "terminal_loss_usd": total,
        "terminal_loss_fraction": total / value,
        "maximum_drawdown_loss": float(drawdowns[trough]),
        "peak_date": str(dates[peak].date()),
        "trough_date": str(dates[trough].date()),
        "recovery_date": str(dates[recovered].date()) if recovered is not None else "",
        "recovery_status": "NO_DRAWDOWN"
        if no_drawdown
        else (
            "RECOVERED_WITHIN_WINDOW" if recovered is not None else "NOT_RECOVERED_WITHIN_WINDOW"
        ),
        "recovery_sessions_from_trough": None if recovered is None else recovered - trough,
        "longest_underwater_sessions": longest,
        "sessions": len(window_prices),
        "gain_needed_after_max_drawdown": float(1 / (1 - drawdowns[trough]) - 1),
        "replay_type": "Counterfactual fixed initial holdings; drift allowed; no intra-window trades/costs",
        "evidence_class": "stress_scenario",
    }
    path = pd.DataFrame({"Date": dates, "wealth_usd": full, "drawdown_loss": drawdowns})
    parts = pd.DataFrame(
        {
            "asset": prices.columns,
            "initial_weight": w,
            "terminal_loss_usd": components,
            "asset_cumulative_return": levels.iloc[-1].to_numpy() - 1,
        }
    )
    return summary, path, parts


def scenario_impact(
    weights: np.ndarray,
    shocks: np.ndarray,
    value: float = DEFAULT_STRESS_VALUE,
    scale: float = 1.0,
) -> dict:
    s = np.asarray(shocks, float)
    w = validate_weights(weights, len(s))
    if (
        not np.isfinite(s).all()
        or not np.isfinite(scale)
        or scale < 0
        or np.any(scale * s < -1)
        or value <= 0
    ):
        raise ValueError("Scenario violates return/capital bounds.")
    components = -value * w * scale * s
    return {
        "loss_fraction": float(components.sum() / value),
        "loss_usd": float(components.sum()),
        "components_usd": components,
        "horizon_note": "Terminal shock only; drawdown/recovery not identified",
    }


def correlation_sensitivity(
    covariance: np.ndarray, weights: np.ndarray, eta: float, multiplier: float
) -> dict:
    cov = np.asarray(covariance, float)
    w = validate_weights(weights, len(cov))
    if (
        not 0 <= eta <= 1
        or multiplier <= 0
        or not np.allclose(cov, cov.T)
        or np.linalg.eigvalsh(cov).min() < EIGENVALUE_TOLERANCE
    ):
        raise ValueError("Invalid covariance or scenario parameters.")
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        raise ValueError("Positive asset variances are required.")
    corr = cov / np.outer(sd, sd)
    stress = (1 - eta) * corr + eta * np.ones_like(corr)
    shifted = stress * np.outer(sd * multiplier, sd * multiplier)
    eigen = float(np.linalg.eigvalsh(shifted).min())
    if eigen < EIGENVALUE_TOLERANCE:
        raise RuntimeError("Stress covariance lost positive semidefiniteness.")
    vol = float(np.sqrt(w @ shifted @ w))
    return {
        "correlation_mixture": eta,
        "volatility_multiplier": multiplier,
        "portfolio_daily_volatility": vol,
        "diversification_ratio": float(w @ (sd * multiplier) / vol),
        "minimum_covariance_eigenvalue": eigen,
        "evidence_class": "stress_sensitivity_not_probability",
    }


def _build_hypothetical_context(
    prices: pd.DataFrame, assets: list[str], shock: np.ndarray
) -> list[dict]:
    rolling = prices.pct_change(ROLLING_SHOCK_WINDOW, fill_method=None).dropna()
    context = []
    for asset, asset_shock in zip(assets, shock):
        series = rolling[asset]
        context.append(
            {
                "asset": asset,
                "hypothetical_return": asset_shock,
                "observations": len(series),
                "development_min": float(series.min()),
                "development_p01": float(series.quantile(0.01)),
                "development_p05": float(series.quantile(0.05)),
                "development_median": float(series.median()),
                "development_p95": float(series.quantile(0.95)),
                "development_p99": float(series.quantile(0.99)),
                "development_max": float(series.max()),
                "historical_fraction_below_shock": float((series <= asset_shock).mean()),
                "interpretation": "Overlapping marginal historical context; not a joint event probability or calibrated forecast",
            }
        )
    return context


def build_stresses(
    prices: pd.DataFrame, exposures: dict, cfg: dict, origin: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    if prices.index[-1] > DEVELOPMENT_END_DATE:
        raise ValueError(
            "Stage 4 cannot release final-period stress results before selection freeze."
        )
    assets = list(prices.columns)
    if assets != cfg["stress"]["assets"]:
        raise ValueError("Scenario asset alignment is invalid.")
    definitions = []
    summaries = []
    paths = []
    parts = []
    hypothetical = []
    hparts = []
    correlations = []
    value = cfg["stress"]["value_usd"]
    shock = np.array(cfg["stress"]["hypothetical_returns"])
    covariance = (
        prices.pct_change(fill_method=None).loc[:origin].tail(LOOKBACK_SESSIONS).cov().to_numpy()
    )
    for definition in WINDOWS:
        release = pd.Timestamp(definition["end"]) <= DEVELOPMENT_END_DATE
        definitions.append(
            {
                **definition,
                "status": "REPLAY_AVAILABLE" if release else "WITHHELD_UNTIL_SELECTION_FREEZE",
            }
        )
        if not release:
            continue
        for name, exposure in exposures.items():
            summary, path, component = replay(
                prices,
                exposure["weights"],
                definition["start"],
                definition["end"],
                value,
            )
            summary.update(
                strategy=name,
                scenario=definition["scenario"],
                weights_origin=str(origin.date()),
            )
            summaries.append(summary)
            paths.append(path.assign(strategy=name, scenario=definition["scenario"]))
            parts.append(component.assign(strategy=name, scenario=definition["scenario"]))
    for name, exposure in exposures.items():
        w = exposure["weights"]
        for scale in cfg["stress"]["hypothetical_scale_factors"]:
            result = scenario_impact(w, shock, value, scale)
            hypothetical.append(
                {
                    "strategy": name,
                    "scenario_scale": scale,
                    "weights_origin": str(origin.date()),
                    "horizon_sessions": cfg["stress"]["hypothetical_horizon_sessions"],
                    "assumption_category": "D managerial scenario",
                    "initial_value_usd": value,
                    **{k: v for k, v in result.items() if k != "components_usd"},
                }
            )
            hparts.extend(
                {
                    "strategy": name,
                    "scenario_scale": scale,
                    "asset": asset,
                    "shock": float(scale * asset_shock),
                    "loss_usd": float(loss),
                }
                for asset, asset_shock, loss in zip(assets, shock, result["components_usd"])
            )
        for eta in cfg["stress"]["correlation_mixture"]:
            for scale in cfg["stress"]["volatility_multipliers"]:
                correlations.append(
                    {
                        "strategy": name,
                        **correlation_sensitivity(covariance, w, eta, scale),
                    }
                )
    return {
        "stress_definitions": pd.DataFrame(definitions),
        "historical_stress": pd.DataFrame(summaries),
        "historical_stress_paths": pd.concat(paths, ignore_index=True),
        "historical_stress_components": pd.concat(parts, ignore_index=True),
        "hypothetical_stress": pd.DataFrame(hypothetical),
        "hypothetical_components": pd.DataFrame(hparts),
        "hypothetical_context": pd.DataFrame(_build_hypothetical_context(prices, assets, shock)),
        "correlation_stress": pd.DataFrame(correlations),
    }
