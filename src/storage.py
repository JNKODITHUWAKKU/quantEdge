"""Atomic CSV exports and compact integrity checks for research results."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

import pandas as pd

from src.common import sha256
from src.settings import FINAL_SETTINGS, INPUT_DIGEST

SUMMARY_LISTS = {
    "strategies": str,
    "methods": str,
    "confidence_levels": float,
    "cost_scenarios_bps": float,
    "selection_changes": str,
}
MONITOR_LISTS = {"review_reasons": str, "robust_adverse_score_references": str}
DIGEST_BLOCK_SIZE = 1024 * 1024
CSV_FLOAT_FORMAT = "%.17g"
REQUIRED_INPUT_FILES = {
    "SPY.csv",
    "EFA.csv",
    "EEM.csv",
    "AGG.csv",
    "GLD.csv",
    "XLE.csv",
    "XLK.csv",
    "DGS3MO.csv",
    "CPIAUCNS.csv",
}
FORBIDDEN_OUTPUT_SUFFIXES = {".json", ".jsonl", ".toml"}
PENDING_FILE_PATTERN = re.compile(r"\.(.+\.csv)\.[0-9a-f]{32}\.pending")
SUMMARY_FILENAME = "run_summary.csv"
MONITOR_ASSESSMENT_FILENAME = "monitor_assessment.csv"
COMMITTEE_ASSESSMENT_FILENAME = "committee_assessment.csv"


def csv_digest(
    folder: Path, *, recursive: bool = False, exclude: str | None = None
) -> str:
    """Fingerprint CSV contents and relative names without creating an inventory file."""
    digest = hashlib.sha256()
    paths = folder.rglob("*.csv") if recursive else folder.glob("*.csv")
    for path in sorted(paths):
        if path.name == exclude:
            continue
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(DIGEST_BLOCK_SIZE), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_inputs(root: Path) -> None:
    """Confirm that the required local CSV snapshot is unchanged."""
    actual = {path.name for path in (root / "data").glob("*.csv")}
    if actual != REQUIRED_INPUT_FILES or csv_digest(root / "data") != INPUT_DIGEST:
        raise ValueError(
            "Required market CSV inputs are missing or differ from the frozen snapshot."
        )


def csv(frame: pd.DataFrame, path: Path) -> None:
    """Round-trip-check each export before replacing the completed CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    _write_and_sync_csv(frame, temporary)
    _verify_roundtrip_shape(frame, temporary, path.name)
    expected_digest = sha256(temporary)
    try:
        temporary.replace(path)
        if sha256(path) != expected_digest:
            raise RuntimeError(
                f"Published CSV bytes differ from the validated export: {path.name}"
            )
    finally:
        # This unique temporary name belongs to this writer invocation. It may
        # be removed even if a nonstandard filesystem leaves a stale alias.
        temporary.unlink(missing_ok=True)
    if temporary.exists():
        raise RuntimeError(
            f"Temporary CSV export could not be removed: {temporary.name}"
        )


def _write_and_sync_csv(frame: pd.DataFrame, destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, float_format=CSV_FLOAT_FORMAT)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_roundtrip_shape(
    expected: pd.DataFrame, temporary: Path, display_name: str
) -> None:
    saved = pd.read_csv(temporary, low_memory=False)
    if saved.shape != expected.shape or list(saved.columns) != list(expected.columns):
        raise RuntimeError(f"Incomplete CSV export: {display_name}")


def _write_row(values: dict, path: Path) -> None:
    # Semicolon-encode list fields so single-row metadata tables stay plain flat CSVs.
    row = {
        key: ";".join(str(item) for item in value) if isinstance(value, list) else value
        for key, value in values.items()
    }
    if any(isinstance(value, dict) for value in row.values()):
        raise ValueError("Nested records are not financial table columns.")
    csv(pd.DataFrame([row]), path)


def _read_row(path: Path, lists: dict | None = None) -> dict:
    table = pd.read_csv(path, keep_default_na=False)
    if len(table) != 1:
        raise ValueError(f"Expected one summary row: {path.name}")
    row = table.iloc[0].to_dict()
    for key, converter in (lists or {}).items():
        if key not in row:
            continue
        row[key] = (
            [converter(value) for value in str(row[key]).split(";")]
            if row[key] != ""
            else []
        )
    return row


def write_assessment(output: Path, monitor: dict, committee: dict) -> None:
    """Save explicit, readable monitoring and committee assessment tables."""
    _write_row(monitor, output / MONITOR_ASSESSMENT_FILENAME)
    _write_row(committee, output / COMMITTEE_ASSESSMENT_FILENAME)


def read_assessment(output: Path) -> dict:
    monitor = _read_row(output / MONITOR_ASSESSMENT_FILENAME, MONITOR_LISTS)
    committee = _read_row(output / COMMITTEE_ASSESSMENT_FILENAME)
    if committee["approved_allocation"] == "":
        committee["approved_allocation"] = None
    return {
        "monitor": monitor,
        "committee": committee,
        "evidence_class": FINAL_SETTINGS["run"]["evidence_class"],
        "final_period_evaluated": True,
        "selection_changes": [],
    }


def write_summary(output: Path, summary: dict) -> None:
    """Save a single ordinary run-summary table; settings remain in Python source."""
    _write_row(summary, output / SUMMARY_FILENAME)


def read_summary(output: Path) -> dict:
    return _read_row(output / SUMMARY_FILENAME, SUMMARY_LISTS)


def finish_run(output: Path) -> None:
    """Publish a compact result fingerprint after all financial exports finish."""
    _cleanup_completed_pending_files(output)
    if any(path.suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES for path in output.rglob("*")):
        raise RuntimeError("Unexpected non-CSV research output.")
    summary = read_summary(output)
    summary["result_digest"] = csv_digest(
        output, recursive=True, exclude=SUMMARY_FILENAME
    )
    summary["csv_table_count"] = len(list(output.rglob("*.csv")))
    write_summary(output, summary)


def _cleanup_completed_pending_files(output: Path) -> None:
    for temporary in sorted(output.rglob("*.pending")):
        match = PENDING_FILE_PATTERN.fullmatch(temporary.name)
        completed = (
            temporary.with_name(match.group(1))
            if match
            else temporary.with_suffix("")
        )
        if not completed.is_file() or sha256(temporary) != sha256(completed):
            raise RuntimeError(
                f"Unfinished result export: {temporary.relative_to(output)}"
            )
        temporary.unlink()


def verify_run(output: Path) -> dict:
    """Detect incomplete or modified generated results without sidecar files."""
    summary = read_summary(output)
    if summary.get("result_digest") != csv_digest(
        output, recursive=True, exclude=SUMMARY_FILENAME
    ):
        raise ValueError("Generated research CSV contents are missing or changed.")
    if int(summary["csv_table_count"]) != len(list(output.rglob("*.csv"))):
        raise ValueError("The research result table count changed.")
    if any(output.rglob("*.pending")):
        raise ValueError("The research result contains unfinished exports.")
    return summary
