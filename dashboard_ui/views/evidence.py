"""Saved research presentation; numerical models remain in src."""

import streamlit as st
from dashboard_ui.data import subset
from dashboard_ui.components import MetricItem, metric_strip, section, table

PRIMARY_EVIDENCE_TABLES = (
    "backtest_diagnostics",
    "deployment_optimization",
    "deployment_filter_diagnostics",
)
EVT_EVIDENCE_TABLES = (
    "deployment_threshold_diagnostics",
    "deployment_evt_gof",
    "deployment_evt_bootstrap",
    "deployment_evt_qqpp",
)


def render(ctx) -> None:
    data = ctx.data
    strategy = ctx.strategy
    summary = data["summary"]
    metric_strip(
        [
            MetricItem("Saved CSV tables", str(summary["csv_table_count"]), "neutral"),
            MetricItem("Forecast records", f"{summary['forecast_rows']:,}", "neutral"),
            MetricItem(
                "Source warnings",
                str(summary["source_warning_count"]),
                "caution",
            ),
            MetricItem("Human review", "PENDING", "caution"),
        ]
    )
    st.warning(
        "51 large-move warnings and 12 adjustment events remain subject to independent human source review. Same-provider reconciliation does not establish independent accuracy. Current-vintage data are not point-in-time vintages."
    )
    section("Quantitative evidence and recovery provenance")
    st.caption(
        "Historically reconstructed pre-outcome forecasts, not proven real-time live issuances or a live trading record."
    )
    for title in PRIMARY_EVIDENCE_TABLES:
        st.write(title)
        table(subset(data[title], strategy), hide_index=True, width="stretch")
    table(data["stationarity_diagnostics"], hide_index=True)
    st.caption("ADF rejection does not establish iid or Gaussian returns.")
    st.write("Monte Carlo agreement flags are retained without seed changes.")
    table(data["monte_carlo_agreement_flags"], hide_index=True, width="stretch")
    table(ctx.release("run_summary"), hide_index=True)
    st.caption(
        "The original final evaluation had already been inspected before the disclosed recovery. This rerun is not a new holdout or a live trading record. Original source/model review remains pending."
    )

    with st.expander("Development uncertainty / frozen 2014 evidence"):
        table(ctx.release("mean_uncertainty"), hide_index=True)
        table(ctx.release("mean_sensitivity"), hide_index=True)
    with st.expander(
        "EVT threshold, fitted-null diagnostics and bootstrap uncertainty"
    ):
        for title in EVT_EVIDENCE_TABLES:
            st.caption(title.replace("_", " "))
            table(subset(data[title], strategy), hide_index=True)
        st.caption(
            "Fixed-threshold intervals omit threshold-selection and model uncertainty; unstable reference fits are not operational estimates."
        )
