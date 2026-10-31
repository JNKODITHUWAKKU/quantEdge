"""Run and atomically publish the fixed research using CSV outputs only."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from src.inputs import research_identity
from src.settings import PORTFOLIO_SETTINGS
from src.storage import finish_run, verify_run

LOGGER = logging.getLogger(__name__)
SUMMARY_FILENAME = "run_summary.csv"
LOCK_FILENAME = ".research.lock"
STAGING_PREFIX = ".research_"
LEGACY_RUN_PATTERN = re.compile(r"research_[0-9a-f]{16}")


def current_result(root: Path) -> Path | None:
    output = root / "results"
    if not (output / SUMMARY_FILENAME).is_file():
        return None
    identity = research_identity(root)
    summary = verify_run(output)
    if not _matches_identity(summary, identity):
        return None
    return output


def _matches_identity(summary: dict | None, identity: dict) -> bool:
    if summary is None:
        return False
    published_id = summary.get("publication_code_id", summary.get("run_id"))
    return published_id == identity["lock_id"]


def _publish_flat_results(working: Path, results: Path) -> None:
    """Replace the direct results CSVs, publishing the integrity summary last."""
    files = sorted(working.glob("*.csv"))
    if not files or any(path.parent != working for path in working.rglob("*.csv")):
        raise RuntimeError("Research outputs must be flat CSV files.")
    names = {path.name for path in files}
    if SUMMARY_FILENAME not in names:
        raise RuntimeError("Research run summary is missing.")

    summary = working / SUMMARY_FILENAME
    for path in files:
        if path != summary:
            os.replace(path, results / path.name)
    for old in results.glob("*.csv"):
        if old.name not in names:
            old.unlink()
    # Move run_summary.csv last so readers never see a new summary pointing to old tables.
    os.replace(summary, results / summary.name)

    # Remove only legacy run directories created by earlier versions of this
    # runner before checking the new flat result fingerprint.
    for old in results.iterdir():
        if old.is_dir() and LEGACY_RUN_PATTERN.fullmatch(old.name):
            shutil.rmtree(old)
    verify_run(results)


@contextmanager
def research_lock(folder: Path):
    path = folder / LOCK_FILENAME
    try:
        handle = path.open("x")
    except FileExistsError as exc:
        raise RuntimeError(
            "A research process is already running. After an interruption, confirm it "
            "has stopped before removing results/.research.lock."
        ) from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def run_research(root: Path) -> Path:
    from src.inputs import settings

    root = root.resolve()
    cfg = settings(root)
    identity = research_identity(root)
    results = root / "results"
    results.mkdir(exist_ok=True)
    with research_lock(results):
        if _has_verified_current_results(results, identity):
            LOGGER.info("Reusing verified flat results in results/.")
            return results
        _execute_and_publish(root, cfg, identity, results)
        return results


def _has_verified_current_results(results: Path, identity: dict) -> bool:
    try:
        summary = verify_run(results)
    except (OSError, ValueError, KeyError):
        return False
    return _matches_identity(summary, identity)


def _execute_and_publish(
    root: Path, cfg: dict, identity: dict, results: Path
) -> None:
    from src.development import build_development
    from src.inputs import load_final
    from src.pipeline import build_final
    from src.reporting import build_display

    working = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=root))
    published = False
    try:
        LOGGER.info("Recomputing the original selected procedure from local CSV inputs.")
        build_final(root, working)
        prices, _, _ = load_final(root, cfg)
        build_development(prices, deepcopy(PORTFOLIO_SETTINGS), working)
        build_display(working, prices.pct_change(fill_method=None).iloc[1:])
        if research_identity(root) != identity:
            raise ValueError("Scientific source or inputs changed during execution.")
        finish_run(working)
        verify_run(working)
        _publish_flat_results(working, results)
        published = True
    except Exception:
        LOGGER.error(
            "Incomplete run retained for diagnosis at %s; not published as results.",
            working,
        )
        raise
    finally:
        if published:
            shutil.rmtree(working, ignore_errors=True)
