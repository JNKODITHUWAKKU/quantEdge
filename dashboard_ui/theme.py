"""Light, restrained styling for the investor and research views."""

from pathlib import Path
import altair as alt
import streamlit as st
from streamlit import config as streamlit_config

THEME_OPTIONS = {
    "theme.base": "light",
    "theme.primaryColor": "#2868a5",
    "theme.backgroundColor": "#ffffff",
    "theme.secondaryBackgroundColor": "#f3f5f7",
    "theme.textColor": "#263445",
}


@alt.theme.register("quantedge_research", enable=False)
def research_theme():
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": "#ffffff",
                "font": "Segoe UI",
                "view": {"stroke": None},
                "axis": {
                    "labelColor": "#66768a",
                    "titleColor": "#66768a",
                    "gridColor": "#e8edf2",
                    "domain": False,
                    "ticks": False,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "titleFontWeight": 400,
                    "labelPadding": 8,
                },
                "legend": {
                    "labelColor": "#526174",
                    "title": None,
                    "orient": "top",
                    "labelLimit": 280,
                    "symbolStrokeWidth": 3,
                },
                "title": {
                    "color": "#344357",
                    "anchor": "start",
                    "fontSize": 12,
                    "fontWeight": 500,
                    "offset": 16,
                },
                "range": {
                    "category": [
                        "#2868a5",
                        "#af7047",
                        "#398570",
                        "#8c77a3",
                        "#b84b50",
                        "#8c9aa8",
                        "#c89c4a",
                    ]
                },
            }
        }
    )


def apply_theme():
    # The pinned Streamlit version has no public per-app theme setter. Refresh
    # once to send the native theme to canvas tables as well as HTML widgets.
    changed = False
    for key, value in THEME_OPTIONS.items():
        if streamlit_config.get_option(key) != value:
            streamlit_config.set_option(key, value)
            changed = True
    if changed:
        st.rerun()
    alt.theme.enable("quantedge_research")
    st.markdown(
        "<style>"
        + Path(__file__).with_name("terminal.css").read_text(encoding="utf-8")
        + "</style>",
        unsafe_allow_html=True,
    )
