"""Verify archived selection evidence without selecting on final-period outcomes.

These are the original saved 2015–2019 comparison tables, not a fresh validation
fit. Rechecking saved gates does not reconstruct the original forecast ledger.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.selection import holm_adjust
from src.settings import VALIDATION_SETTINGS

EVIDENCE_DIGEST = "8950eb5b88453aa2d3848b1eb34ebdb42fc7e2c82546ba14e297df6cdb500027"
EXPECTED_CALIBRATION_FAMILY_SIZE = 80
CALIBRATION_KEY_COLUMNS = ["method", "strategy", "confidence", "test"]
ISSUE_RECONCILIATION_COLUMNS = ["severity", "symbol", "check", "date", "detail"]
HOLM_RTOL = 1e-10
HOLM_ATOL = 1e-12
CALIBRATION_ALPHA = VALIDATION_SETTINGS["risk_selection"]["calibration_family_alpha"]
VALIDATION_END_DATE = pd.Timestamp(VALIDATION_SETTINGS["run"]["end"])
EXPECTED_BLOCK_LENGTHS = set(VALIDATION_SETTINGS["risk_selection"]["block_lengths"])
SELECTED_MONITOR = "garch_t"
INELIGIBLE_DEFAULT_MONITOR = "ewma_fhs"
SELECTED_STRATEGY = "sample_gmv"


def verify_evidence(root: Path) -> dict[str, pd.DataFrame]:
    folder = root / "evidence"
    if _compute_evidence_digest(folder) != EVIDENCE_DIGEST:
        raise ValueError("Supporting selection/source evidence is missing or changed.")
    tables = {path.stem: pd.read_csv(path) for path in sorted(folder.glob("*.csv"))}
    _verify_calibration_family(tables["validation_calibration_family"])
    _verify_selection_tables(tables)
    return tables


def _compute_evidence_digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        if not path.is_file():
            raise ValueError("Unexpected folder in supporting research evidence.")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _verify_calibration_family(family: pd.DataFrame) -> None:
    if (
        len(family) != EXPECTED_CALIBRATION_FAMILY_SIZE
        or family.duplicated(CALIBRATION_KEY_COLUMNS).any()
    ):
        raise ValueError("Original validation calibration family is incomplete.")
    adjusted = holm_adjust(family.family_input_pvalue.to_numpy())
    if not np.allclose(adjusted, family.holm_pvalue, rtol=HOLM_RTOL, atol=HOLM_ATOL):
        raise ValueError("Archived Holm-adjusted validation screens do not reconcile.")
    if not np.array_equal(adjusted < CALIBRATION_ALPHA, family.reject):
        raise ValueError("Archived validation rejection decisions do not reconcile.")
    rejects = family.groupby("method").reject.sum()
    # EWMA-FHS failed one calibration test in validation; GARCH-t passed all and became the selected monitor.
    if rejects[INELIGIBLE_DEFAULT_MONITOR] != 1 or rejects[SELECTED_MONITOR] != 0:
        raise ValueError("Original eligibility evidence differs from the recorded choice.")


def _verify_selection_tables(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["validation_portfolio_metrics"]
    if not pd.to_datetime(metrics.end_date).eq(VALIDATION_END_DATE).all():
        raise ValueError("Final-period data entered the archived selection comparison.")
    intervals = tables["validation_portfolio_variance_comparison"]
    if set(intervals.block_length) != EXPECTED_BLOCK_LENGTHS or intervals.ci95_high.lt(0).all():
        raise ValueError("Original evidence for retaining Sample GMV changed.")
    selection = tables["validation_selection"].set_index("field").value
    if (
        selection["risk.selected_model"] != SELECTED_MONITOR
        or selection["portfolio.selected_strategy"] != SELECTED_STRATEGY
    ):
        raise ValueError("Archived selection is inconsistent with the fixed procedure.")


def reconcile_source_issues(observed: pd.DataFrame, archived: pd.DataFrame) -> None:
    """Compare every warning, not only the total, with the original source audit."""
    left = (
        observed[ISSUE_RECONCILIATION_COLUMNS]
        .fillna("")
        .sort_values(ISSUE_RECONCILIATION_COLUMNS)
        .reset_index(drop=True)
    )
    right = (
        archived[ISSUE_RECONCILIATION_COLUMNS]
        .fillna("")
        .sort_values(ISSUE_RECONCILIATION_COLUMNS)
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise ValueError("Recomputed source issues differ from the original audit evidence.")
