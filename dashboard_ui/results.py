"""Locate verified dashboard outputs without changing research publication."""

from pathlib import Path

from src.inputs import research_identity
from src.runner import current_result
from src.storage import verify_run


def find_dashboard_results(project: Path) -> Path | None:
    """Read a relocated flat result set, or the original generated run folder.

    Flat CSVs take precedence. An incomplete or invalid flat set fails explicitly
    instead of silently showing a different nested run.
    """
    output = project / "results"
    if (output / "run_summary.csv").is_file() or any(output.glob("*.csv")):
        summary = verify_run(output)
        if summary["run_id"] != research_identity(project)["lock_id"]:
            raise ValueError("Results belong to a different research implementation.")
        return output
    return current_result(project)
