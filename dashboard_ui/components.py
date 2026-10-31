"""Shared, restrained presentation components."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
from dashboard_ui.data import FilterSelection
from dashboard_ui.disclosures import (
    CONFIDENCE_LEVELS,
    COST_SCENARIOS_BPS,
    METHODS,
    SELECTED_MONITOR,
)

DEFAULT_CONFIDENCE_INDEX = 2
DEFAULT_COST_INDEX = 1

PERCENT_COLUMNS = frozenset(
    {
        "var",
        "es",
        "weight",
        "historical_held_weight",
        "confidence",
        "cagr",
        "real_total_return",
        "annualized_volatility",
        "maximum_drawdown_loss",
        "terminal_loss_fraction",
        "loss_fraction",
        "var_component",
        "es_component",
        "portfolio_var",
        "portfolio_es",
        "annual_real_spending",
        "illustrative_loss_tolerance",
    }
)
WEIGHT_PREFIXES = ("end_weight_", "start_weight_")


@dataclass(frozen=True)
class MetricItem:
    """Single KPI tile specification for a horizontal metric strip."""

    label: str
    value: str
    tone: str = "neutral"
    note: str = ""


def _coerce_metric_item(item: MetricItem | tuple) -> MetricItem:
    if isinstance(item, MetricItem):
        return item
    label, value, *rest = item
    item_tone = rest[0] if rest else "neutral"
    note = rest[1] if len(rest) > 1 else ""
    return MetricItem(label=label, value=value, tone=item_tone, note=note)


def section(title: str) -> None:
    st.subheader(title)


def remembered_selectbox(label: str, options, *, key: str, **kwargs):
    """Retain page-specific controls when Streamlit cleans up inactive widgets."""
    saved_key = f"_saved_{key}"
    option_list = list(options)
    if saved_key in st.session_state and st.session_state[saved_key] in option_list:
        kwargs["index"] = option_list.index(st.session_state[saved_key])

    def remember():
        st.session_state[saved_key] = st.session_state[key]

    value = st.selectbox(label, option_list, key=key, on_change=remember, **kwargs)
    st.session_state[saved_key] = value
    return value


def metric_strip(items) -> None:
    for column, raw_item in zip(st.columns(len(items)), items):
        item = _coerce_metric_item(raw_item)
        with column:
            st.metric(item.label, item.value)
            if item.note:
                st.caption(item.note)


def _is_dollar_column(lower_name: str) -> bool:
    return (
        "usd" in lower_name
        or "dollars" in lower_name
        or lower_name == "market_pnl"
    )


def _is_percent_column(lower_name: str) -> bool:
    return lower_name in PERCENT_COLUMNS or lower_name.startswith(WEIGHT_PREFIXES)


def _infer_column_config(frame: pd.DataFrame) -> dict:
    config = {}
    for col in frame.select_dtypes(include="number"):
        lower_name = str(col).lower()
        if _is_dollar_column(lower_name):
            config[col] = st.column_config.NumberColumn(format="dollar")
        elif _is_percent_column(lower_name):
            config[col] = st.column_config.NumberColumn(format="percent")
    return config


def table(frame, **kwargs):
    """Format known units without altering data or exported CSV values."""
    config = _infer_column_config(frame) if isinstance(frame, pd.DataFrame) else {}
    config.update(kwargs.pop("column_config", {}))
    kwargs.setdefault("width", "stretch")
    return st.dataframe(frame, column_config=config, **kwargs)


def sidebar(data: dict, workspaces: list[str]) -> FilterSelection:
    st.sidebar.subheader("Advanced analysis")
    workspace = st.sidebar.radio(
        "Workspace",
        workspaces,
        key="workspace",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Display filters")
    confidence = st.sidebar.selectbox(
        "Confidence level",
        list(CONFIDENCE_LEVELS),
        index=DEFAULT_CONFIDENCE_INDEX,
        format_func=lambda x: f"{x:.0%}",
        key="final_confidence",
    )
    method_keys = list(METHODS)
    method = st.sidebar.selectbox(
        "VaR methodology",
        method_keys,
        index=method_keys.index(SELECTED_MONITOR),
        format_func=METHODS.get,
        key="final_method",
    )
    first_date = pd.Timestamp(data["summary"]["first_realization"]).date()
    last_date = pd.Timestamp(data["summary"]["last_realization"]).date()
    dates = st.sidebar.date_input(
        "Backtest date range",
        value=(first_date, last_date),
        min_value=first_date,
        max_value=last_date,
        key="final_dates",
    )
    st.sidebar.caption(
        "Dates filter diagnostic backtests only. Full-period performance, cutoff allocation and assessment remain fixed."
    )
    cost_bps = st.sidebar.selectbox(
        "Assumed trading cost (bp)",
        list(COST_SCENARIOS_BPS),
        index=DEFAULT_COST_INDEX,
        key="final_cost",
    )
    st.sidebar.caption(
        "Selects a saved cost case; execution costs are assumed. Spending and loss-tolerance illustrations are in Funding & Performance."
    )
    st.sidebar.caption(
        "Sample GMV · GARCH Student-t\n\nHOLD / NO TRADE · approval pending\n\nControls do not refit or reselect models."
    )
    return FilterSelection(
        workspace=workspace,
        cost_bps=cost_bps,
        confidence=confidence,
        method=method,
        dates=dates,
    )


def shell() -> None:
    st.caption(
        "HOLD / NO TRADE · Sample GMV portfolio · GARCH Student-t research monitor · Approval pending"
    )
    st.caption(
        "Retrospective evaluation: 2 Jan 2020–23 Oct 2026 · Real wealth through 30 Sep 2026 · Development through 2014 · Validation selection 2015–2019"
    )


def page_heading(workspace: str, ctx) -> None:
    st.title(workspace)
    st.caption(
        f"{ctx.confidence:.0%} confidence · {ctx.bps:g} bp assumed trading cost · Saved research results"
    )


def fallback_notice(row: pd.Series) -> None:
    if row.fit_status == "FALLBACK":
        st.warning(
            f"Requested {METHODS[row.method]} uses {row.actual_model_used}: {row.fallback_reason}. This is a fallback estimate."
        )


def tone(value: float) -> str:
    if not np.isfinite(value) or value == 0:
        return "neutral"
    return "positive" if value > 0 else "negative"
