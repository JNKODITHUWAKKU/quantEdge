"""Committee landing page, grounded in the fixed retrospective assessment."""

import numpy as np
import streamlit as st
from dashboard_ui.charts import bars, lines
from dashboard_ui.components import (
    MetricItem,
    fallback_notice,
    metric_strip,
    section,
    table,
    tone,
)
from dashboard_ui.data import subset
from dashboard_ui.disclosures import METHODS

OVERVIEW_CHART_HEIGHT = 220
OVERVIEW_COLUMN_RATIOS = (1, 1.5)


def _render_headline_metrics(ctx) -> None:
    metrics = ctx.metrics
    funding = ctx.funding
    metric_strip(
        [
            MetricItem(
                "Nominal terminal portfolio",
                f"${metrics.terminal_wealth_usd:,.0f}",
                "neutral",
                f"23 Oct 2026 · {ctx.bps:g} bp cost case",
            ),
            MetricItem(
                "Nominal net CAGR",
                f"{metrics.cagr:.2%}",
                tone(metrics.cagr),
                "Full retrospective period",
            ),
            MetricItem(
                "Real capital change",
                f"{funding.real_total_return:.2%}",
                tone(funding.real_total_return),
                "Through 30 Sep 2026 · before spending",
            ),
            MetricItem(
                "Maximum drawdown",
                f"{metrics.maximum_drawdown_loss:.2%}",
                "negative",
                "Full-period peak-to-trough loss",
            ),
        ]
    )
    st.caption(
        "USD 10m initial reference capital. Nominal gains coexisted with a loss of purchasing power. Low nominal variance does not establish suitability for the university's spending and capital-preservation needs."
    )


def _render_allocation_and_risk_columns(ctx) -> None:
    left_col, right_col = st.columns(OVERVIEW_COLUMN_RATIOS, gap="medium")
    with left_col:
        section("Indicative allocation / 23 Oct 2026")
        bars(
            ctx.weights,
            "asset",
            ["weight"],
            percent=True,
            height=OVERVIEW_CHART_HEIGHT,
        )
        table(
            ctx.weights[["asset", "weight", "reference_notional_usd"]], hide_index=True
        )
        st.caption(
            "Saved fresh-capital target; indicative and unexecuted. Historical held weights drift between trades. No allocation is approved."
        )
    with right_col:
        section("Historical net portfolio value / USD")
        lines(
            ctx.daily.reset_index(),
            "Date",
            ["nav_usd"],
            usd=True,
            height=OVERVIEW_CHART_HEIGHT,
        )
        section(f"One-session risk / {ctx.confidence:.0%} confidence")
        metric_strip(
            [
                MetricItem(
                    "VaR / loss threshold",
                    f"${ctx.chosen.var_usd:,.0f}",
                    "caution",
                    f"{ctx.chosen['var']:.2%} of USD 10m",
                ),
                MetricItem(
                    "ES / average tail loss",
                    f"${ctx.chosen.es_usd:,.0f}",
                    "negative",
                    f"{ctx.chosen.es:.2%} of USD 10m",
                ),
            ]
        )
        st.caption(
            f"Display method: {METHODS[ctx.method]}. Same allocation, valuation, confidence and one-session horizon. Static exposure excludes execution costs and has no attached future outcome. VaR is a threshold, not a maximum loss."
        )
        fallback_notice(ctx.chosen)


def _render_committee_risk_brief(ctx) -> None:
    section("Committee risk brief")
    stress = subset(ctx.data["historical_stress"], ctx.strategy)
    worst_replay = stress.loc[stress.maximum_drawdown_loss.idxmax()]
    hypothetical = subset(ctx.data["hypothetical_stress"], ctx.strategy)
    base_shock = hypothetical[np.isclose(hypothetical.scenario_scale, 1.0)].iloc[0]
    metric_strip(
        [
            MetricItem(
                "Real wealth / no spending",
                f"${ctx.funding.terminal_real_no_spending_usd:,.0f}",
                "negative",
                "Dec 2019 purchasing power · Sep 2026 endpoint",
            ),
            MetricItem(
                "Largest replay drawdown",
                f"{worst_replay.maximum_drawdown_loss:.2%}",
                "negative",
                str(worst_replay.scenario),
            ),
            MetricItem(
                "Base hypothetical loss",
                f"${base_shock.loss_usd:,.0f}",
                "negative",
                f"{base_shock.loss_fraction:.2%} · stipulated 20-session scenario",
            ),
        ]
    )
    st.warning(
        "Even zero spending fails the real-principal floor. Failed final GARCH-t calibration is a separate obstacle. A passing diagnostic or a more permissive loss tolerance cannot authorize deployment."
    )
    st.caption(
        f"Guarded filtered EVT-policy VaR {ctx.evt['var']:.2%} / ES {ctx.evt.es:.2%} (USD {ctx.evt.es_usd:,.0f}); actual model: {ctx.evt.actual_model_used}; status: {ctx.evt.fit_status}."
    )
    fallback_notice(ctx.evt)
    st.caption(
        "The USD10m long-term endowment supports scholarships, research, faculty development and capital projects. Human investment judgment is pending. Display controls do not change the frozen choice, assessment or final-period test family. Final evaluation is a disclosed post-opening recovery, not a pristine untouched holdout."
    )


def render(ctx) -> None:
    st.error(
        "HOLD / NO TRADE. Reject deployment of Sample GMV as the complete endowment allocation on current evidence. GARCH-t remains the selected research monitor with failed final calibration; shadow use only."
    )
    _render_headline_metrics(ctx)
    _render_allocation_and_risk_columns(ctx)
    _render_committee_risk_brief(ctx)
