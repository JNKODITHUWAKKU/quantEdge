"""Saved research presentation; numerical models remain in src."""

import streamlit as st
from dashboard_ui.disclosures import METHODS
from dashboard_ui.data import subset
from dashboard_ui.charts import bars, evt_threshold_charts
from dashboard_ui.components import (
    MetricItem,
    fallback_notice,
    metric_strip,
    remembered_selectbox,
    section,
    table,
)

RISK_COMPARISON_HEIGHT = 300
ES_CONTRIBUTION_HEIGHT = 220
DEFAULT_CONTRIBUTION_METHOD_INDEX = 3

CONTRIBUTION_METHODS = ("historical", "gaussian", "ewma_fhs", "garch_fhs")
EVT_FAMILIES = ("garch", "raw")
EVT_DETAIL_TABLES = (
    "deployment_evt_bootstrap",
    "deployment_evt_gof",
    "deployment_evt_qqpp",
)
RISK_TABLE_COLUMNS = [
    "method",
    "var",
    "es",
    "var_usd",
    "es_usd",
    "fit_status",
    "actual_model_used",
    "fallback_reason",
]


def _render_selected_risk_summary(ctx) -> None:
    st.info(
        "VaR is a loss threshold at the selected confidence. ES / CVaR measures average loss in the remaining tail. Losses can exceed both estimates."
    )
    metric_strip(
        [
            MetricItem(
                "Selected VaR",
                f"${ctx.chosen.var_usd:,.0f}",
                "caution",
                f"{ctx.chosen['var']:.2%} of USD 10m",
            ),
            MetricItem(
                "Selected Expected Shortfall",
                f"${ctx.chosen.es_usd:,.0f}",
                "negative",
                f"{ctx.chosen.es:.2%} of USD 10m",
            ),
        ]
    )
    st.caption(
        f"Requested distribution: {METHODS[ctx.method]} · Actual model: {ctx.chosen.actual_model_used} · Fit status: {ctx.chosen.fit_status}. One-session static cutoff exposure; no model is refitted by this control."
    )
    fallback_notice(ctx.chosen)


def _render_matched_risk_methods(ctx) -> None:
    section("Matched risk methods")
    bars(
        ctx.risk.assign(Model=ctx.risk.method.map(METHODS)),
        "Model",
        ["var_usd", "es_usd"],
        title="ONE-SESSION LOSS / USD 10M REFERENCE EXPOSURE",
        usd=True,
        height=RISK_COMPARISON_HEIGHT,
    )
    table(
        ctx.risk[RISK_TABLE_COLUMNS],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Historical risk inherits sample tails. Gaussian analytic and Monte Carlo share a fitted distribution. Conditional models change volatility; qualified EVT changes tail shape. Simulation error is separate from model uncertainty."
    )
    with st.expander("Loss definitions and conditional-tail mathematics"):
        st.latex(
            r"L=-w^\top R,\quad VaR_c=F_L^{-1}(c),\quad ES_c=\frac{1}{1-c}\int_c^1 VaR_u\,du"
        )
        st.latex(
            r"\sigma_t^2=\omega+\alpha\epsilon_{t-1}^2+\beta\sigma_{t-1}^2,\quad z_t=\sqrt{(\nu-2)/\nu}\ T_\nu"
        )
        st.latex(
            r"q_c=u+\frac{\beta_{GPD}}{\xi}\left[\left(\frac{p_u}{1-c}\right)^\xi-1\right],\quad ES_c=\frac{q_c+\beta_{GPD}-\xi u}{1-\xi},\quad\xi<1"
        )
        st.write(
            "Fixed POT grid 80/85/90/92.5/95%, at least 50 exceedances and three contiguous stable thresholds. Raw POT falls back to historical simulation; filtered POT to FHS; failed GARCH to EWMA-FHS. Fixed-threshold bootstrap intervals omit selection/model uncertainty. Actual supplying model is disclosed. Gaussian Monte Carlo shares the analytic Gaussian law."
        )


def _render_es_contributions_and_evt(ctx) -> None:
    contribution_method = remembered_selectbox(
        "ES contribution distribution",
        list(CONTRIBUTION_METHODS),
        index=DEFAULT_CONTRIBUTION_METHOD_INDEX,
        format_func=METHODS.get,
        key="final_contribution_method",
    )
    parts = subset(
        ctx.data["deployment_es_contributions"], ctx.strategy, ctx.confidence
    )
    parts = parts[parts.method.eq(contribution_method)]
    bars(
        parts,
        "asset",
        ["es_component"],
        title="ES CONTRIBUTION / MATCHED DISTRIBUTION",
        percent=True,
        height=ES_CONTRIBUTION_HEIGHT,
    )
    st.caption(
        f"Sum {parts.es_component.sum():.4%}; matching total {parts.portfolio_es.iloc[0]:.4%}. These allocate that distribution's ES, not GARCH-t or EVT ES."
    )
    family = remembered_selectbox(
        "EVT diagnostic loss series", list(EVT_FAMILIES), key="final_evt_family"
    )
    grid = subset(ctx.data["deployment_threshold_diagnostics"], ctx.strategy)
    grid = grid[grid["filter"].eq(family)]
    table(grid, hide_index=True, width="stretch")
    evt_threshold_charts(grid)
    with st.expander("Selected-portfolio QQ/PP, fitted-null KS and uncertainty"):
        st.caption(
            "Cutoff bootstrap details are for selected sample GMV. Reference-only unstable fits are not operational models. Fixed-percentile intervals omit threshold-selection uncertainty."
        )
        for title in EVT_DETAIL_TABLES:
            st.write(title)
            table(subset(ctx.data[title], ctx.strategy), hide_index=True, width="stretch")


def render(ctx) -> None:
    _render_selected_risk_summary(ctx)
    _render_matched_risk_methods(ctx)
    _render_es_contributions_and_evt(ctx)
