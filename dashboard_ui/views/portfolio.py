"""Saved research presentation; numerical models remain in src."""

import streamlit as st
from dashboard_ui.data import subset
from dashboard_ui.charts import bars, comparison_paths, frontier_chart, lines
from dashboard_ui.components import section, table

TARGET_BAR_HEIGHT = 240
DRAWDOWN_LINE_HEIGHT = 160
ALLOCATION_COLUMN_RATIOS = (1, 1.4)
SUMMARY_METRIC_COLUMNS = [
    "strategy",
    "cagr",
    "annualized_volatility",
    "maximum_drawdown_loss",
    "sharpe",
]


def _render_target_allocation(ctx) -> None:
    left_col, right_col = st.columns(ALLOCATION_COLUMN_RATIOS)
    with left_col:
        section("Indicative target / latest held exposure")
        held_weights = subset(ctx.data["latest_held_weights"], ctx.strategy)[
            ["asset", "weight"]
        ].rename(columns={"weight": "historical_held_weight"})
        table(
            ctx.weights[["asset", "weight", "reference_notional_usd"]].merge(
                held_weights, on="asset"
            ),
            hide_index=True,
            width="stretch",
        )
    with right_col:
        bars(
            ctx.weights,
            "asset",
            ["weight"],
            title="UNEXECUTED CUTOFF TARGET / SEVEN-ASSET UNIVERSE",
            percent=True,
            height=TARGET_BAR_HEIGHT,
        )
    st.caption(
        "Fresh-capital target at 23 October 2026; historical holdings drift between quarterly trades. No allocation is approved."
    )


def _render_historical_performance_and_objective(ctx, daily_frame) -> None:
    section("Historical portfolio / net performance")
    lines(daily_frame, "Date", ["nav_usd"], usd=True)
    lines(daily_frame, "Date", ["drawdown"], percent=True, height=DRAWDOWN_LINE_HEIGHT)
    st.latex(
        r"r_s^{(t)}=w_t^\top R_s,\quad x_s^{(t)}=r_s^{(t)}-\bar r^{(t)},\quad s\leq t"
    )
    st.caption(
        "Quarterly allocation uses sample covariance. Risk models refit at each historical origin on a single recast exposure series. Historical origins use the known held exposure; cutoff snapshots use the explicitly unexecuted fresh-capital target. This training series is distinct from the strategy's realized dynamic-weight return path."
    )
    section("Sample covariance and the minimum-variance objective")
    st.latex(
        r"\min_w w^\top\widehat\Sigma w\quad \mathbf{1}^\top w=1,\quad 0\leq w_i\leq1"
    )
    st.write(
        "Sample covariance uses 1,260 past daily simple returns, annualized by 252. No expected returns enter GMV. Quarterly signals execute at the next session close and affect the following session return; holdings drift between trades. There is no extra concentration cap or leverage."
    )
    frontier_chart(ctx.release("frontier"))
    st.caption(
        "Frozen development frontier at 31 Dec 2014; extra return-floor constraints illustrate mean-variance trade-offs. These are estimated means, not future returns or the funding objective. Final results did not select this frontier."
    )


def _render_dynamic_weights_and_comparisons(ctx, daily_frame) -> None:
    bps = ctx.bps
    section("Historical dynamic weights")
    weight_cols = [col for col in daily_frame if col.startswith("end_weight_")]
    lines(daily_frame, "Date", weight_cols, True)
    table(
        ctx.release(f"sample_gmv_cost_{bps:g}bps_signals"),
        hide_index=True,
    )
    table(subset(ctx.data["deployment_optimization"], ctx.strategy), hide_index=True)
    st.caption(
        "Numerical feasibility/optimality certify the quadratic program, not economic suitability. Concentration can be a correct mathematical solution and an unsuitable institutional allocation."
    )
    with st.expander("Development sensitivity and mean uncertainty"):
        table(ctx.release("mean_sensitivity"), hide_index=True)
        table(ctx.release("mean_uncertainty"), hide_index=True)
    table(
        ctx.data["portfolio_metrics"].query("cost_bps == @bps")[SUMMARY_METRIC_COLUMNS],
        hide_index=True,
    )
    st.caption(
        "Original baselines remain comparisons. No final-period model reselection is allowed."
    )
    section("Original strategy comparison / matched cost case")
    comparison_paths(ctx)
    table(
        ctx.release("matched_baselines").query("cost_bps == @bps"),
        hide_index=True,
        width="stretch",
    )


def render(ctx) -> None:
    daily_frame = ctx.daily.reset_index()
    _render_target_allocation(ctx)
    _render_historical_performance_and_objective(ctx, daily_frame)
    _render_dynamic_weights_and_comparisons(ctx, daily_frame)
