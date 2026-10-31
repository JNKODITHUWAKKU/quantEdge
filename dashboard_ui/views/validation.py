"""Saved research presentation; numerical models remain in src."""

import pandas as pd
import streamlit as st
from src.risk_validation import evaluate_forecasts
from dashboard_ui.data import subset
from dashboard_ui.charts import forecast_chart
from dashboard_ui.components import MetricItem, metric_strip, section, table

EXCEPTION_TABLE_COLUMNS = [
    "forecast_origin",
    "realization_date",
    "var",
    "realized_hpl_loss",
]


def _render_diagnostic_backtest(ctx, filtered_forecasts: pd.DataFrame) -> None:
    section("Diagnostic backtest")
    summary_stats = evaluate_forecasts(filtered_forecasts)
    metric_strip(
        [
            MetricItem("Forecasts", f"{summary_stats['observations']:,}", "neutral"),
            MetricItem("Exceptions", str(summary_stats["exceptions"]), "negative"),
            MetricItem(
                "Expected exceptions",
                f"{summary_stats['expected_exceptions']:.1f}",
                "neutral",
            ),
            MetricItem(
                "Kupiec POF p-value",
                f"{summary_stats['kupiec_pvalue']:.4f}",
                "neutral",
            ),
        ]
    )
    st.info(
        "Statistical non-rejection is not proof of model accuracy. Sparse tail events, dependence and retrospective recovery limit inference. Full-period research assessment remains fixed."
    )
    zone = summary_stats["basel_zone"]
    indicator = {"GREEN": st.success, "AMBER": st.warning, "RED": st.error}.get(
        zone, st.info
    )
    indicator(f"Final-view Basel reference: {zone}")
    st.caption(
        f"{summary_stats['basel_note']} {summary_stats['independence_status']}"
    )
    christoffersen_pvalue = summary_stats["christoffersen_pvalue"]
    metric_strip(
        [
            MetricItem(
                "Exact coverage p-value",
                f"{summary_stats['exact_binomial_pvalue']:.4f}",
                "neutral",
            ),
            MetricItem(
                "Christoffersen independence p",
                "N/A"
                if christoffersen_pvalue is None
                else f"{christoffersen_pvalue:.4f}",
                "neutral",
            ),
        ]
    )
    forecast_chart(filtered_forecasts)
    exceptions = filtered_forecasts[
        filtered_forecasts.realized_hpl_loss > filtered_forecasts["var"]
    ]
    table(exceptions[EXCEPTION_TABLE_COLUMNS], hide_index=True)
    table(
        pd.DataFrame(
            {
                "Diagnostic": list(summary_stats),
                "Value": [str(val) for val in summary_stats.values()],
            }
        ),
        hide_index=True,
    )
    st.download_button(
        "Download filtered diagnostic forecasts",
        filtered_forecasts.to_csv(index=False).encode(),
        file_name="gmv_filtered_forecasts.csv",
        key="gmv_forecasts_download",
    )


def _render_full_period_assessment(ctx) -> None:
    section("Fixed full-period assessment · sample GMV / GARCH-t")
    table(ctx.release("monitor_assessment"), hide_index=True)
    table(ctx.data["final_calibration_family"], hide_index=True)
    table(ctx.data["final_score_comparisons"], hide_index=True)
    with st.expander("Basel history and pointwise ES diagnostics"):
        table(ctx.data["selected_monitor_basel_history"], hide_index=True)
        table(
            subset(ctx.data["es_calibration"], ctx.strategy, ctx.confidence),
            hide_index=True,
        )
        st.caption(
            "A latest green window does not erase full-period failures. ES moment bands are diagnostics, not calibration certification. Date slicing cannot reselect a model."
        )


def render(ctx) -> None:
    if ctx.date_error:
        st.info(ctx.date_error)
        section("Fixed full-period assessment · Sample GMV / GARCH-t")
        table(ctx.release("monitor_assessment"), hide_index=True)
        st.caption(
            "The date filter does not change the research decision or hide other workspaces."
        )
        return
    forecasts = subset(ctx.data["forecasts"], ctx.strategy, ctx.confidence)
    filtered_forecasts = forecasts[
        forecasts.method.eq(ctx.method)
        & forecasts.realization_date.between(ctx.start, ctx.end)
    ]
    _render_diagnostic_backtest(ctx, filtered_forecasts)
    _render_full_period_assessment(ctx)
