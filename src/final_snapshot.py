"""Cutoff targets, matching risk and scenario replay; no executed allocation claim."""

import numpy as np
import pandas as pd

from src.conditional import forecast_portfolio
from src.forecasting import forecast_cross_section
from src.portfolio import target_for_strategy
from src.risk import seeded_identifier
from src.storage import csv
from src.stress import WINDOWS, correlation_sensitivity, replay, scenario_impact
from src.tails import gpd_qqpp, refit_ks_bootstrap
from src.uncertainty import snapshot_uncertainty

REFERENCE_THRESHOLD_PERCENTILE = 0.9
HYPOTHETICAL_HORIZON_SESSIONS = 20


def _compute_cutoff_targets(
    training: pd.DataFrame,
    origin: pd.Timestamp,
    strategies: list[str],
    capital: float,
    output,
) -> dict[str, dict]:
    assets = list(training.columns)
    exposures = {}
    weights_rows = []
    optimizer_rows = []
    for name in strategies:
        weights, meta = target_for_strategy(training, name)
        exposures[name] = {
            "weights": weights,
            "origin_value_usd": capital,
            "realization_date": pd.NaT,
        }
        weights_rows.extend(
            {
                "strategy": name,
                "asset": asset,
                "weight": float(weight),
                "reference_notional_usd": float(capital * weight),
                "valuation_date": origin,
                "role": "INDICATIVE_FRESH_CAPITAL_TARGET_NOT_EXECUTED",
            }
            for asset, weight in zip(assets, weights)
        )
        optimizer_rows.append({"strategy": name, "valuation_date": origin, **meta})
    csv(pd.DataFrame(weights_rows), output / "deployment_weights.csv")
    csv(pd.DataFrame(optimizer_rows), output / "deployment_optimization.csv")
    return exposures


def _collect_selected_evt_diagnostics(
    training: pd.DataFrame,
    origin: pd.Timestamp,
    name: str,
    exposure: dict,
    tail_rows: list[dict],
    state: dict,
    conditional: dict,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    qq_rows = []
    gof_rows = []
    boot_rows = []
    for family, losses in state["losses"].items():
        fit = state["selections"][family]["selected"]
        reference = fit is None
        if reference:
            # When the stability rule rejects POT, still compute 90th-percentile diagnostics as a reference.
            match = [
                row
                for row in tail_rows
                if row["filter"] == family
                and np.isclose(row["threshold_percentile"], REFERENCE_THRESHOLD_PERCENTILE)
                and row["fit_status"] == "OK"
            ]
            if len(match) != 1:
                continue
            fit = match[0]
        excess = losses[losses > fit["threshold"]] - fit["threshold"]
        qq_rows.extend(
            {
                "strategy": name,
                "filter": family,
                "forecast_origin": origin,
                "reference_only": reference,
                **row,
            }
            for row in gpd_qqpp(excess, fit)
        )
        gof_rows.append(
            {
                "strategy": name,
                "filter": family,
                "forecast_origin": origin,
                "reference_only": reference,
                **refit_ks_bootstrap(
                    fit,
                    conditional["evt"]["gof_replicates"],
                    seeded_identifier(seed, "final_gof", family),
                ),
            }
        )
        boot_rows.extend(
            {**row, "forecast_origin": origin, "reference_only": reference}
            for row in snapshot_uncertainty(
                training, exposure["weights"], family, fit, conditional, name
            )
        )
    return qq_rows, gof_rows, boot_rows


def _export_cutoff_stress_tables(
    prices: pd.DataFrame,
    training: pd.DataFrame,
    origin: pd.Timestamp,
    exposures: dict[str, dict],
    capital: float,
    conditional: dict,
    output,
) -> None:
    assets = list(training.columns)
    summaries = []
    paths = []
    components = []
    hypos = []
    hparts = []
    correlations = []
    shock = np.asarray(conditional["stress"]["hypothetical_returns"])
    covariance = training.cov().to_numpy()
    for name, exposure in exposures.items():
        weights = exposure["weights"]
        for definition in WINDOWS:
            result, path, parts = replay(
                prices, weights, definition["start"], definition["end"], capital
            )
            summaries.append(
                {
                    **result,
                    "strategy": name,
                    "scenario": definition["scenario"],
                    "mechanism": definition["mechanism"],
                    "weights_origin": origin,
                }
            )
            paths.append(path.assign(strategy=name, scenario=definition["scenario"]))
            components.append(parts.assign(strategy=name, scenario=definition["scenario"]))
        for scale in conditional["stress"]["hypothetical_scale_factors"]:
            result = scenario_impact(weights, shock, capital, scale)
            hypos.append(
                {
                    "strategy": name,
                    "scenario_scale": scale,
                    "weights_origin": origin,
                    "horizon_sessions": HYPOTHETICAL_HORIZON_SESSIONS,
                    "assumption_category": "D",
                    "probability": "NOT_ASSIGNED",
                    **{k: v for k, v in result.items() if k != "components_usd"},
                }
            )
            hparts.extend(
                {
                    "strategy": name,
                    "scenario_scale": scale,
                    "asset": asset,
                    "shock": float(scale * asset_shock),
                    "loss_usd": float(loss_usd),
                }
                for asset, asset_shock, loss_usd in zip(assets, shock, result["components_usd"])
            )
        for eta in conditional["stress"]["correlation_mixture"]:
            for multiple in conditional["stress"]["volatility_multipliers"]:
                correlations.append(
                    {
                        "strategy": name,
                        **correlation_sensitivity(covariance, weights, eta, multiple),
                    }
                )
    for title, frame in [
        ("historical_stress", pd.DataFrame(summaries)),
        ("historical_stress_paths", pd.concat(paths)),
        ("historical_stress_components", pd.concat(components)),
        ("hypothetical_stress", pd.DataFrame(hypos)),
        ("hypothetical_components", pd.DataFrame(hparts)),
        ("correlation_stress", pd.DataFrame(correlations)),
    ]:
        csv(frame, output / (title + ".csv"))


def deployment_snapshot(prices, returns, cfg, base_risk, conditional, output):
    origin = returns.index[-1]
    training = returns.tail(cfg["run"]["lookback_returns"])
    capital = cfg["run"]["reference_capital_usd"]
    exposures = _compute_cutoff_targets(
        training, origin, cfg["run"]["strategies"], capital, output
    )
    base, parts = forecast_cross_section(
        returns,
        origin,
        exposures,
        cfg["run"]["lookback_returns"],
        cfg["run"]["confidence_levels"],
        base_risk["monte_carlo"],
        base_risk["run"]["seed"],
    )
    frows = base.to_dict("records")
    components = parts.to_dict("records")
    grids = []
    filters = []
    extremes = []
    qq = []
    gof = []
    boot = []
    for name, exposure in exposures.items():
        rows, contrib, tail_rows, fil, ext, state = forecast_portfolio(
            training, origin, exposure, name, conditional, diagnostics=True
        )
        frows.extend(rows)
        grids.extend(tail_rows)
        filters.extend(fil)
        extremes.extend(ext)
        for row in contrib:
            row["es_component_usd"] = row["es_component"] * capital
        components.extend(contrib)
        if name != cfg["assessment"]["portfolio"]:
            continue
        qq_rows, gof_rows, boot_rows = _collect_selected_evt_diagnostics(
            training,
            origin,
            name,
            exposure,
            tail_rows,
            state,
            conditional,
            cfg["run"]["seed"],
        )
        qq.extend(qq_rows)
        gof.extend(gof_rows)
        boot.extend(boot_rows)
    forecast = pd.DataFrame(frows).assign(
        evidence_class="cutoff_static_exposure_diagnostic",
        horizon_note="One-session hypothetical market loss; target not executed; no future outcome",
    )
    for title, frame in [
        ("deployment_risk", forecast),
        ("deployment_es_contributions", pd.DataFrame(components)),
        ("deployment_threshold_diagnostics", pd.DataFrame(grids)),
        ("deployment_filter_diagnostics", pd.DataFrame(filters)),
        ("deployment_extremograms", pd.DataFrame(extremes)),
        ("deployment_evt_qqpp", pd.DataFrame(qq)),
        ("deployment_evt_gof", pd.DataFrame(gof)),
        ("deployment_evt_bootstrap", pd.DataFrame(boot)),
    ]:
        csv(
            frame if len(frame.columns) else pd.DataFrame(columns=["status"]),
            output / (title + ".csv"),
        )
    _export_cutoff_stress_tables(
        prices, training, origin, exposures, capital, conditional, output
    )
    return forecast
