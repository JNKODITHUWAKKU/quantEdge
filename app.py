"""Investor summary and advanced research views over verified local results."""

import logging
from pathlib import Path
import pandas as pd
import streamlit as st
from src.common import sha256
from dashboard_ui.results import find_dashboard_results as existing_final
from dashboard_ui.data import _tables, Context
from dashboard_ui.disclosures import (
    METHODS as METHODS,
    PRIMARY_CONFIDENCE,
    PRIMARY_COST_BPS,
    SELECTED_MONITOR,
    proposed_mandate as proposed_mandate,
    risk_budget_review as risk_budget_review,
)
from dashboard_ui.theme import apply_theme
from dashboard_ui.components import shell, sidebar, page_heading
from dashboard_ui.views import (
    evidence,
    executive,
    funding,
    investor,
    limitations,
    portfolio,
    risk,
    stress,
    validation,
)

LOGGER = logging.getLogger(__name__)

APP_TITLE = "Serendib University Endowment Fund"
HEADER_COLUMN_RATIOS = (4, 1)

WORKSPACE_RENDERERS = {
    "Executive Overview": executive,
    "Portfolio Construction": portfolio,
    "Risk Analytics": risk,
    "Backtesting & Model Validation": validation,
    "Stress Testing": stress,
    "Funding & Performance": funding,
    "Quantitative Evidence": evidence,
    "Assumptions & Limitations": limitations,
}
WORKSPACES = list(WORKSPACE_RENDERERS)

ADVANCED_KEYS = (
    "workspace",
    "final_cost",
    "final_confidence",
    "final_method",
    "final_dates",
)


def switch_mode(mode: str, workspace: str | None = None) -> None:
    if mode == "investor":
        st.session_state["_advanced_settings"] = {
            key: st.session_state[key]
            for key in ADVANCED_KEYS
            if key in st.session_state
        }
    else:
        for key, value in st.session_state.get("_advanced_settings", {}).items():
            st.session_state[key] = value
        if workspace is not None:
            st.session_state["workspace"] = workspace
    st.session_state["view_mode"] = mode


def header(is_advanced: bool) -> None:
    title_col, action_col = st.columns(
        HEADER_COLUMN_RATIOS, vertical_alignment="center"
    )
    with title_col:
        st.markdown(
            f'<div class="app-name">{APP_TITLE}</div>',
            unsafe_allow_html=True,
        )
    button_label, button_key, target_mode = (
        ("← Investor view", "open_investor", "investor")
        if is_advanced
        else ("Advanced mode →", "open_advanced", "advanced")
    )
    with action_col:
        st.button(
            button_label,
            key=button_key,
            on_click=switch_mode,
            args=(target_mode,),
            width="stretch",
        )


def _load_verified_results(project: Path) -> tuple[Path, dict] | None:
    try:
        path = existing_final(project)
        if path is None:
            st.error(
                "Final outputs are missing. Run python main.py from the project folder to generate the research results."
            )
            return None
        return path, _tables(str(path), sha256(path / "run_summary.csv"))
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        st.error(f"Final evidence cannot be used: {exc}")
        return None


def _render_investor_view(path: Path, data: dict) -> None:
    summary = data["summary"]
    dates = (
        pd.Timestamp(summary["first_realization"]).date(),
        pd.Timestamp(summary["last_realization"]).date(),
    )
    ctx = Context.create(
        path,
        data,
        PRIMARY_COST_BPS,
        PRIMARY_CONFIDENCE,
        SELECTED_MONITOR,
        dates,
    )
    investor.render(ctx, switch_mode)


def _render_advanced_view(path: Path, data: dict) -> None:
    selection = sidebar(data, WORKSPACES)
    ctx = Context.create(
        path,
        data,
        selection.cost_bps,
        selection.confidence,
        selection.method,
        selection.dates,
    )
    shell()
    page_heading(selection.workspace, ctx)
    WORKSPACE_RENDERERS[selection.workspace].render(ctx)
    st.caption(
        "Retrospective research. Human review and investment approval remain pending."
    )


def render_gmv(project: Path) -> None:
    loaded = _load_verified_results(project)
    if loaded is None:
        return
    path, data = loaded
    is_advanced = st.session_state.get("view_mode", "advanced") == "advanced"
    header(is_advanced)
    if not is_advanced:
        _render_investor_view(path, data)
        return
    _render_advanced_view(path, data)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    try:
        render_gmv(Path(__file__).resolve().parent)
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        pd.errors.ParserError,
    ) as exc:
        LOGGER.warning("Dashboard could not render: %s", exc)


if __name__ == "__main__":
    main()
