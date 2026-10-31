"""Saved research presentation; numerical models remain in src."""

import streamlit as st
from dashboard_ui.data import subset
from dashboard_ui.charts import bars
from dashboard_ui.components import remembered_selectbox, section, table

HISTORICAL_REPLAY_CHART_HEIGHT = 260
HYPOTHETICAL_STRESS_TABLES = ("hypothetical_stress", "hypothetical_components")
HISTORICAL_ATTRIBUTION_TABLES = (
    "historical_stress_components",
    "correlation_stress",
)


def render(ctx) -> None:
    data = ctx.data
    strategy = ctx.strategy
    stress = subset(data["historical_stress"], strategy)
    bars(
        stress,
        "scenario",
        ["terminal_loss_fraction", "maximum_drawdown_loss"],
        title="HISTORICAL REPLAY / TERMINAL LOSS VS MAXIMUM DRAWDOWN",
        percent=True,
        height=HISTORICAL_REPLAY_CHART_HEIGHT,
    )
    st.caption(
        "Counterfactual replay of the unexecuted 23 October 2026 target. Positive values denote losses; negative terminal losses represent gains."
    )
    section("Historical crises · counterfactual cutoff holdings")
    table(stress, hide_index=True, width="stretch")
    st.caption(
        "Fixed initial cutoff holdings drift; no intra-window trading or transaction costs. Terminal loss and maximum drawdown differ; recovery is censored at the scenario end."
    )
    st.write("Causal strategy during crises")
    table(subset(data["causal_strategy_crises"], strategy), hide_index=True)
    st.caption(
        "Causal results use then-known quarterly allocations and net costs. No causal 2008 return is invented before the five-year formation period."
    )
    section("Hypothetical stagflation and liquidity squeeze")
    st.write(
        "Rates and inflation pressure hurt duration and equities; energy rises with supply pressure; gold initially falls under forced liquidation. Shock magnitudes/scalings are managerial scenarios with no assigned probability."
    )
    for title in HYPOTHETICAL_STRESS_TABLES:
        table(subset(data[title], strategy), hide_index=True, width="stretch")
    with st.expander("Historical attribution and diversification failure"):
        for title in HISTORICAL_ATTRIBUTION_TABLES:
            table(subset(data[title], strategy), hide_index=True, width="stretch")

    section("Asset-level historical stress attribution")
    scenario = remembered_selectbox(
        "Historical replay attribution",
        stress.scenario.tolist(),
        key="stress_scenario",
    )
    parts = subset(data["historical_stress_components"], strategy)
    bars(
        parts[parts.scenario.eq(scenario)],
        "asset",
        ["terminal_loss_usd"],
        title="TERMINAL LOSS CONTRIBUTION / USD",
        usd=True,
    )
    hypothetical = subset(data["hypothetical_stress"], strategy)
    bars(
        hypothetical.assign(
            Scale=hypothetical.scenario_scale.map(lambda x: f"{x:g}× base shock")
        ),
        "Scale",
        ["loss_usd"],
        title="HYPOTHETICAL STAGFLATION / 20-SESSION STIPULATED LOSS",
        usd=True,
    )
