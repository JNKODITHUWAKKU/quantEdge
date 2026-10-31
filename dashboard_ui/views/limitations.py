"""Saved research presentation; numerical models remain in src."""

import pandas as pd
import streamlit as st
from dashboard_ui.disclosures import FUNDING, MANDATE, RECOVERY
from dashboard_ui.components import remembered_selectbox, section, table

MAX_PREVIEW_ROWS = 1000


def render(ctx) -> None:
    data = ctx.data
    section("Assumptions & Limitations")
    st.markdown(MANDATE)
    with st.expander("Fixed assessment and recovery", expanded=True):
        st.markdown(RECOVERY)
    with st.expander("Funding mathematics", expanded=True):
        st.markdown(FUNDING)
    st.warning(
        "Human source/model review, liability/spending decisions, investment approval and fresh second-machine review remain pending. No machine learning is included in this final submission."
    )
    section("Download research data")
    evidence_name = remembered_selectbox(
        "Frozen table",
        [key for key, val in data.items() if isinstance(val, pd.DataFrame)],
        key="gmv_download",
    )
    table(data[evidence_name].head(MAX_PREVIEW_ROWS), hide_index=True)
    st.download_button(
        "Download selected table",
        data[evidence_name].to_csv(index=False).encode(),
        file_name=f"{evidence_name}.csv",
        key="gmv_table_download",
    )
    st.caption(
        "Detailed original validation selection records and recovery documents remain in the separate full research release. CSV integrity verification certifies file consistency; it does not certify source accuracy, economic suitability or investment approval. Software checks do not replace independent human validation. The team remains responsible for source and formula review, institutional decisions and official submission."
    )
