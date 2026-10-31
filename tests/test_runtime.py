"""Runtime integrity and entry-point behavior without expensive research fits."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import src.runner
from src.inputs import load_cpi, load_final, settings
from src.runner import _publish_flat_results, current_result, research_lock
from src.storage import (
    csv,
    finish_run,
    read_assessment,
    verify_inputs,
    verify_run,
    write_assessment,
    write_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_local_inputs_and_boundaries():
    verify_inputs(ROOT)
    cfg = settings(ROOT)
    prices, rates, _ = load_final(ROOT, cfg)
    assert prices.index[-1] == pd.Timestamp("2026-10-23")
    assert not prices.isna().any().any() and not rates.isna().any()
    assert list(prices.columns) == ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
    cpi = load_cpi(ROOT, cfg)
    assert cpi[cpi.isna()].index.astype(str).tolist() == ["2025-10"]


def test_modified_input_is_rejected(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    path = tmp_path / "data/SPY.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen snapshot"):
        verify_inputs(tmp_path)


def test_research_lock_is_exclusive_and_released(tmp_path):
    with research_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with research_lock(tmp_path):
                pass
    assert not (tmp_path / ".research.lock").exists()


def test_failed_operation_releases_lock(tmp_path):
    with pytest.raises(ValueError):
        with research_lock(tmp_path):
            raise ValueError("Synthetic failure")
    assert not (tmp_path / ".research.lock").exists()


def test_result_fingerprint_rejects_modified_table(tmp_path):
    write_summary(tmp_path, {"run_id": "synthetic"})
    csv(pd.DataFrame({"value": [1.0, 2.0]}), tmp_path / "data.csv")
    finish_run(tmp_path)
    assert verify_run(tmp_path)["run_id"] == "synthetic"
    (tmp_path / "data.csv").write_text("value\n999\n")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_run(tmp_path)


def test_no_active_result_is_not_fabricated(tmp_path):
    assert current_result(tmp_path) is None


def test_results_publish_flat_and_remove_legacy_run_folder(tmp_path):
    staged = tmp_path / ".staged"
    staged.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    legacy = results / "research_0123456789abcdef"
    legacy.mkdir()
    (legacy / "old.csv").write_text("old\n1\n")
    csv(pd.DataFrame({"value": [1.0]}), staged / "headline.csv")
    write_summary(staged, {"run_id": "flat-run"})
    finish_run(staged)

    _publish_flat_results(staged, results)

    assert verify_run(results)["run_id"] == "flat-run"
    assert {path.name for path in results.iterdir()} == {
        "headline.csv",
        "run_summary.csv",
    }


def test_cli_help_does_not_run_research():
    process = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0 and "--verify" in process.stdout


def test_app_missing_results_has_clear_guidance(monkeypatch):
    import app as dashboard
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(dashboard, "existing_final", lambda _: None)
    app = AppTest.from_string("import app\napp.main()").run()
    assert not app.exception
    assert any("Run python main.py" in item.value for item in app.error)


def test_publication_removes_only_identical_completed_export_residue(tmp_path):
    write_summary(tmp_path, {"run_id": "synthetic"})
    csv(pd.DataFrame({"value": [1.0, 2.0]}), tmp_path / "data.csv")
    shutil.copy2(tmp_path / "data.csv", tmp_path / "data.csv.pending")
    finish_run(tmp_path)
    assert not (tmp_path / "data.csv.pending").exists()
    assert verify_run(tmp_path)["run_id"] == "synthetic"


def test_publication_rejects_an_unfinished_export(tmp_path):
    write_summary(tmp_path, {"run_id": "synthetic"})
    (tmp_path / "data.csv").write_text("value\n1\n")
    (tmp_path / "data.csv.pending").write_text("value\n9\n")
    with pytest.raises(RuntimeError, match="Unfinished result export"):
        finish_run(tmp_path)


def test_assessment_csv_roundtrip_preserves_domain_values(tmp_path):
    monitor = {
        "review_reasons": ["COVERAGE_FAILURE"],
        "robust_adverse_score_references": [],
        "holm_rejections": 3,
        "missing_tests": 0,
        "replacement_selected": False,
    }
    committee = {
        "approved_allocation": None,
        "human_decision": "PENDING",
        "zero_spend_real_principal_floor_met": False,
    }
    write_assessment(tmp_path, monitor, committee)
    actual = read_assessment(tmp_path)
    assert actual["monitor"] == monitor
    assert actual["committee"] == committee
    assert all(path.suffix == ".csv" for path in tmp_path.iterdir())


def test_no_sidecar_files_are_shipped():
    forbidden = {".json", ".jsonl", ".toml"}
    ignored = {
        "results",
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".agents",
        "skills-lock.json",
    }
    assert not any(
        path.suffix.lower() in forbidden
        for path in ROOT.rglob("*")
        if not ignored.intersection(path.relative_to(ROOT).parts)
    )
    assert {
        path.suffix
        for path in (ROOT / "data").iterdir()
        if not path.name.startswith(".")
    } == {".csv"}


def test_successful_csv_export_cleans_a_stale_temporary_copy(tmp_path, monkeypatch):
    expected = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    def copy_then_leave_stale_name(temporary, destination):
        shutil.copyfile(temporary, destination)
        temporary.write_text("value\n1\n")
        return destination

    monkeypatch.setattr(Path, "replace", copy_then_leave_stale_name)
    csv(expected, tmp_path / "data.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "data.csv"), expected, check_dtype=False
    )
    assert not (tmp_path / "data.csv.pending").exists()


def test_research_package_contains_no_interface_modules():
    # Enforce strict separation: src/ must remain pure numerical/research code with no UI dependencies.
    forbidden = {"streamlit", "altair", "app"}
    assert not any(
        (ROOT / "src" / name).exists()
        for name in ["dashboard.py", "labels.py", "methodology.py"]
    )
    for path in (ROOT / "src").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name.split(".")[0] in forbidden for alias in node.names
                ), path.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, path.name


def test_csv_publication_detects_corrupt_destination_and_cleans_owned_temporary(
    tmp_path, monkeypatch
):
    def corrupt_after_replace(temporary, destination):
        shutil.copyfile(temporary, destination)
        destination.write_text("value\n999\n")
        return destination

    monkeypatch.setattr(Path, "replace", corrupt_after_replace)
    with pytest.raises(RuntimeError, match="Published CSV bytes differ"):
        csv(pd.DataFrame({"value": [1.0, 2.0]}), tmp_path / "data.csv")
    assert not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("identical", [True, False])
def test_unique_temporary_alias_cleanup_requires_exact_bytes(tmp_path, identical):
    write_summary(tmp_path, {"run_id": "synthetic"})
    csv(pd.DataFrame({"value": [1.0, 2.0]}), tmp_path / "data.csv")
    alias = tmp_path / (".data.csv." + "a" * 32 + ".pending")
    shutil.copyfile(tmp_path / "data.csv", alias)
    if not identical:
        alias.write_text("value\n999\n")
        with pytest.raises(RuntimeError, match="Unfinished result export"):
            finish_run(tmp_path)
        return
    finish_run(tmp_path)
    assert not alias.exists()
    verify_run(tmp_path)


def test_publication_identity_preserves_original_calculation_id(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    folder.mkdir()
    write_summary(
        folder,
        {"run_id": "calculation-original", "publication_code_id": "publisher-current"},
    )
    csv(pd.DataFrame({"value": [1.0]}), folder / "data.csv")
    finish_run(folder)
    monkeypatch.setattr(
        src.runner, "research_identity", lambda _: {"lock_id": "publisher-current"}
    )
    assert src.runner.current_result(tmp_path) == folder
    assert verify_run(folder)["run_id"] == "calculation-original"
