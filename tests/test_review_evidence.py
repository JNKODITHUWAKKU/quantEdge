"""Acceptance checks for source auditing, historical evidence and decision display."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import proposed_mandate, risk_budget_review
from src.audit import local_dates
from src.common import session_calendar
from src.evidence import reconcile_source_issues, verify_evidence
from src.inputs import load_final, settings
from src.settings import SOURCE_SETTINGS
from src.source_quality import AUDIT_RULES, audit_equity, audit_local_sources

ROOT = Path(__file__).resolve().parents[1]


def test_every_original_warning_reproduces_from_raw_inputs():
    evidence = verify_evidence(ROOT)
    issues, inventory = audit_local_sources(ROOT)
    reconcile_source_issues(issues, evidence["original_source_issues"])
    assert issues.groupby("check").size().to_dict() == {
        "adjustment_jump": 12,
        "large_return": 51,
    }
    assert not issues.severity.eq("ERROR").any()
    assert len(inventory) == 7
    _, _, audit = load_final(ROOT, settings(ROOT))
    assert audit["warning_count"] == len(issues)


def test_selection_evidence_tampering_is_rejected(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    table = tmp_path / "evidence/validation_selection.csv"
    table.write_text(table.read_text().replace("garch_t", "ewma_fhs"))
    with pytest.raises(ValueError, match="evidence is missing or changed"):
        verify_evidence(tmp_path)


def test_issue_reconciliation_detects_same_count_different_event():
    archived = verify_evidence(ROOT)["original_source_issues"]
    changed = archived.copy()
    changed.loc[0, "date"] = "2001-01-01"
    with pytest.raises(ValueError, match="differ"):
        reconcile_source_issues(changed, archived)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("duplicate", "duplicate_session"),
        ("missing", "missing_session"),
        ("invalid", "invalid_price"),
        ("ohlc", "ohlc_order"),
    ],
)
def test_original_screens_reject_corrupt_market_records(mutation, expected):
    frame = pd.read_csv(ROOT / "data/SPY.csv")
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[50]]], ignore_index=True)
    elif mutation == "missing":
        frame = frame.drop(index=50)
    elif mutation == "invalid":
        frame.loc[50, "Adj Close"] = -1
    else:
        frame.loc[50, "High"] = 0.01
    cfg = {**SOURCE_SETTINGS, "audit": AUDIT_RULES}
    sessions = session_calendar(
        cfg["data"]["start"], cfg["data"]["cutoff"], cfg["data"]["calendar"]
    )
    _, issues, _ = audit_equity(frame, {"currency": "USD"}, "SPY", sessions, cfg)
    assert any(item.severity == "ERROR" and item.check == expected for item in issues)


def test_risk_budget_preserves_hedging_contributions_and_does_not_change_weights():
    # Asset C has a negative ES component (hedge); positive-share normalization must not alter signed Euler values.
    components = pd.DataFrame(
        {
            "asset": ["A", "B", "C"],
            "weight": [0.6, 0.3, 0.1],
            "marginal_var": [0.1, 0.2, -0.1],
            "var_component": [0.06, 0.06, -0.01],
            "es_component": [0.08, 0.04, -0.01],
        }
    )
    original = components.copy(deep=True)
    display = risk_budget_review(components)
    pd.testing.assert_frame_equal(components, original)
    assert display.es_component.iloc[-1] == -0.01
    np.testing.assert_allclose(display.positive_es_share, [2 / 3, 1 / 3, 0])
    assert display.sizing_review.iloc[0] == "REVIEW BEFORE ADDING EXPOSURE"
    zero = components.copy()
    zero.es_component = 0
    assert risk_budget_review(zero).sizing_review.eq("NOT ASSESSABLE").all()


def test_proposed_mandate_states_decision_and_evidence_boundary():
    text = proposed_mandate().to_string(index=False)
    assert "DO NOT DEPLOY" in text and "20-year" in text
    assert "Post-evaluation" in text and "3%" in text
    assert "no 20-year forecast" in text


def test_raw_date_convention_is_exchange_local():
    dates = local_dates(pd.Series(["2020-01-02 00:00:00-05:00"]))
    assert dates[0] == pd.Timestamp("2020-01-02")
