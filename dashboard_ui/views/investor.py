"""Plain-language investor overview using the frozen primary research case."""

import altair as alt
import pandas as pd
import streamlit as st
from dashboard_ui.charts import add_nan_segments, show
from dashboard_ui.data import subset
from dashboard_ui.components import MetricItem, metric_strip, table
from dashboard_ui.disclosures import ASSET_NAMES

WEALTH_CHART_HEIGHT = 310
WEALTH_STROKE_WIDTH = 2
PERFORMANCE_ALLOCATION_RATIOS = (2, 1)
RISK_COLUMN_RATIOS = (1, 1, 2)
WEALTH_SERIES_DOMAIN = ["Portfolio value", "After inflation"]
WEALTH_SERIES_COLORS = ["#2868a5", "#af7047"]


def wealth_chart(ctx) -> None:
    nominal = ctx.daily.reset_index()[["Date", "nav_usd"]].rename(
        columns={"nav_usd": "Value"}
    )
    nominal["Series"] = "Portfolio value"
    real = subset(ctx.data["real_wealth_paths"], ctx.strategy)
    real = real[real.cost_bps.eq(ctx.bps)][
        ["valuation_date", "real_no_spending_base_usd"]
    ].rename(columns={"valuation_date": "Date", "real_no_spending_base_usd": "Value"})
    real["Series"] = "After inflation"
    frame = pd.concat([nominal, real], ignore_index=True)
    frame["Date"] = pd.to_datetime(frame.Date)
    frame["segment"] = add_nan_segments(frame)
    chart = (
        alt.Chart(frame)
        .mark_line(strokeWidth=WEALTH_STROKE_WIDTH)
        .encode(
            x=alt.X(
                "Date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")
            ),
            y=alt.Y(
                "Value:Q",
                title="USD",
                axis=alt.Axis(format="$~s"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=WEALTH_SERIES_DOMAIN,
                    range=WEALTH_SERIES_COLORS,
                ),
                title=None,
            ),
            detail=["Series:N", "segment:N"],
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                "Series:N",
                alt.Tooltip("Value:Q", title="Value", format="$,.0f"),
            ],
        )
    )
    show(chart, WEALTH_CHART_HEIGHT)


def _render_overview_metrics(ctx) -> None:
    metrics = ctx.metrics
    funding = ctx.funding
    metric_strip(
        [
            MetricItem(
                "Portfolio value",
                f"${metrics.terminal_wealth_usd:,.0f}",
                "neutral",
                "At 23 October 2026",
            ),
            MetricItem(
                "Annual return",
                f"{metrics.cagr:.2%}",
                "neutral",
                "Annualized growth after assumed trading costs",
            ),
            MetricItem(
                "Purchasing power",
                f"{funding.real_total_return:.2%}",
                "neutral",
                "Change after inflation, through September 2026",
            ),
            MetricItem(
                "Largest fall",
                f"{metrics.maximum_drawdown_loss:.2%}",
                "neutral",
                "Largest peak-to-trough loss in the study",
            ),
        ]
    )


def _render_performance_and_allocation(ctx) -> None:
    performance_col, allocation_col = st.columns(
        PERFORMANCE_ALLOCATION_RATIOS, gap="large"
    )
    with performance_col:
        st.subheader("Portfolio value over time")
        wealth_chart(ctx)
        st.caption(
            "After-inflation values use December 2019 purchasing power and end on 30 September 2026. The missing October 2025 CPI observation is left as a gap."
        )
    with allocation_col:
        st.subheader("Portfolio allocation")
        weights = ctx.weights[["asset", "weight"]].copy()
        weights["Holding"] = weights.asset + " · " + weights.asset.map(ASSET_NAMES)
        table(
            weights[["Holding", "weight"]].rename(columns={"weight": "Weight"}),
            hide_index=True,
            column_config={"Weight": st.column_config.NumberColumn(format="percent")},
        )
        st.caption(
            "Indicative target at 23 October 2026. These weights are unexecuted; no allocation has been approved."
        )


def _render_tail_risk_summary(ctx) -> None:
    st.subheader("What could a bad day cost?")
    left_col, right_col, explanation_col = st.columns(RISK_COLUMN_RATIOS, gap="large")
    with left_col:
        st.metric("Daily loss threshold (VaR)", f"${ctx.chosen.var_usd:,.0f}")
        st.caption(f"{ctx.chosen['var']:.2%} of the USD 10m reference portfolio")
    with right_col:
        st.metric("Average loss beyond it (ES)", f"${ctx.chosen.es_usd:,.0f}")
        st.caption(f"{ctx.chosen.es:.2%} of the USD 10m reference portfolio")
    with explanation_col:
        st.write(
            "At 99% confidence, VaR is the model's estimated one-day loss threshold. Expected Shortfall estimates the average loss in the remaining 1% tail. Losses can be larger than either number."
        )
        st.caption(
            "Selected GARCH Student-t monitor; its final calibration failed. These estimates require review."
        )
    if ctx.chosen.fit_status == "FALLBACK":
        st.warning(
            f"The requested monitor uses {ctx.chosen.actual_model_used}: {ctx.chosen.fallback_reason}."
        )


def render(ctx, switch_mode) -> None:
    st.title("Portfolio overview")
    st.caption(
        "Retrospective research · 2 January 2020 – 23 October 2026 · USD 10 million starting capital"
    )
    st.markdown(
        '<div class="decision-note"><strong>HOLD / NO TRADE</strong><p>The current evidence does not support investing. The portfolio lost purchasing power, and the selected risk model failed its final calibration checks. Investment approval is pending.</p></div>',
        unsafe_allow_html=True,
    )
    _render_overview_metrics(ctx)
    _render_performance_and_allocation(ctx)
    _render_tail_risk_summary(ctx)
    with st.expander("A few things to understand before investing"):
        st.write(
            "Nominal growth did not keep up with inflation. Even with no spending, the saved research fails to preserve initial real capital. Raising a loss tolerance does not resolve this funding shortfall or the model-calibration failure."
        )
        st.write(
            "The selected Sample GMV rule minimizes estimated nominal variance. It does not directly optimize the university's liabilities, income needs or long-term purchasing power. Human source/model review and institutional spending, liquidity and loss limits are still required."
        )
        st.write(
            "This is retrospective research with disclosed recovery after final results had been seen. It is not an untouched holdout or a live investment record."
        )
    st.caption(
        "Summary uses the fixed primary research case: 5 bp assumed trading costs, no withdrawals and 99% confidence. Risk is a one-session static exposure on USD 10m and excludes execution costs. Detailed methods, scenarios, downloads and all disclosures are available in Advanced mode."
    )
    st.button(
        "Assumptions & limitations",
        key="investor_limitations",
        on_click=switch_mode,
        args=("advanced", "Assumptions & Limitations"),
    )
