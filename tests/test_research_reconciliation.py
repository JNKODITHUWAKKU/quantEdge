"""Independent final artifact reconciliation and recovery conformance."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.final_assessment import assess_monitor, committee_assessment
from src.inputs import load_cpi, settings
from src.inputs import research_identity as verify_lock
from src.risk_validation import evaluate_forecasts
from src.runner import current_result as existing_final
from src.selection import holm_adjust
from src.storage import read_assessment, verify_run

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FORECAST_GROUPS = 150
TRAINING_LOOKBACK = 1260
REFERENCE_CAPITAL_USD = 1e7
PRIMARY_COST_BPS = 5
BPS_DIVISOR = 10_000
EXPECTED_CHECK_COUNT = 65

# Regression references from the original opened/recovered research record.
# They are test expectations, never optimizer inputs or displayed results.
PRIOR_REFERENCES = {
    "gmv_cagr_rounded": 0.016185,
    "gmv_drawdown_rounded": 0.177985,
    "gmv_real_total_return_rounded": -0.170661,
    "primary_holm_rejections": 3,
    "garch_t_95_exceptions": 108,
    "garch_t_99_exceptions": 37,
    "monte_carlo_agreement_flag_rows": 5,
}

pytestmark = pytest.mark.skipif(
    not (ROOT / "results/run_summary.csv").is_file(),
    reason="Run python main.py before scientific integration tests.",
)


def _check_accounting_and_hpl_invariants(
    path: Path, cfg: dict, evaluated: pd.DataFrame, checks: dict
) -> None:
    max_accounting_error = max_hpl_error = 0.0
    for cost in cfg["run"]["cost_scenarios_bps"]:
        for name in cfg["run"]["strategies"]:
            daily = pd.read_csv(
                path / f"{name}_cost_{cost:g}bps_daily.csv",
                index_col="Date",
                parse_dates=True,
            )
            max_accounting_error = max(
                max_accounting_error,
                float(
                    (
                        daily.nav_usd
                        - (
                            daily.nav_start_reported
                            + daily.market_pnl
                            - daily.trading_cost_dollars
                        )
                    )
                    .abs()
                    .max()
                ),
                float(daily.accounting_residual.abs().max()),
            )
            checks[f"initial_capital_{name}_{cost:g}"] = (
                abs(daily.nav_start_reported.iloc[0] - REFERENCE_CAPITAL_USD) < 1e-6
            )
            if cost == PRIMARY_COST_BPS:
                hpl = (
                    evaluated[evaluated.strategy.eq(name)]
                    .groupby("realization_date")
                    .realized_hpl_loss.first()
                )
                max_hpl_error = max(
                    max_hpl_error, float((hpl + daily.gross_return).abs().max())
                )
    checks["accounting_reconciles"] = max_accounting_error < 1e-6
    checks["hpl_matches_holdings"] = max_hpl_error < 1e-12


def _check_stress_attribution_invariants(path: Path, checks: dict) -> None:
    weights = pd.read_csv(path / "deployment_weights.csv")
    checks["feasible_weights"] = (
        np.allclose(weights.groupby("strategy").weight.sum(), 1, atol=1e-12)
        and weights.weight.between(0, 1).all()
    )
    stress = pd.read_csv(path / "historical_stress.csv").set_index(
        ["strategy", "scenario"]
    )
    stress_parts = (
        pd.read_csv(path / "historical_stress_components.csv")
        .groupby(["strategy", "scenario"])
        .terminal_loss_usd.sum()
    )
    checks["historical_attribution"] = np.allclose(
        stress_parts,
        stress.loc[stress_parts.index].terminal_loss_usd,
        atol=1e-6,
        rtol=1e-12,
    )
    checks["mandatory_crises"] = {
        "GFC_2008",
        "COVID_March_2020",
        "Rates_2022",
    }.issubset(stress.index.get_level_values("scenario"))
    hypo = pd.read_csv(path / "hypothetical_stress.csv").set_index(
        ["strategy", "scenario_scale"]
    )
    hypo_parts = (
        pd.read_csv(path / "hypothetical_components.csv")
        .groupby(["strategy", "scenario_scale"])
        .loss_usd.sum()
    )
    checks["hypothetical_attribution"] = np.allclose(
        hypo_parts, hypo.loc[hypo_parts.index].loss_usd, atol=1e-6, rtol=1e-12
    )


def _check_funding_invariants(
    cfg: dict, summary: dict, fund: pd.DataFrame, paths: pd.DataFrame, checks: dict
) -> None:
    cpi = load_cpi(ROOT, cfg)
    checks["cpi_gap_retained"] = (
        cpi[cpi.isna()].index.astype(str).tolist() == ["2025-10"]
        and paths.loc[paths.reference_month.eq("2025-10"), "real_no_spending_base_usd"]
        .isna()
        .all()
    )
    checks["real_nominal_endpoints_distinct"] = (
        fund.last_valuation_date.eq("2026-09-30").all()
        and summary["last_realization"] == "2026-10-23"
    )
    max_funding_error = 0.0
    for _, row in fund.iterrows():
        wealth_path = paths[
            paths.strategy.eq(row.strategy) & paths.cost_bps.eq(row.cost_bps)
        ]
        initial = float(wealth_path.nominal_no_spending_usd.iloc[0])
        real = float(
            wealth_path.nominal_no_spending_usd.iloc[-1] / wealth_path.cpi_ratio.iloc[-1]
        )
        max_funding_error = max(
            max_funding_error, abs(real - row.terminal_real_no_spending_usd)
        )
        budget = row.feasible_nonnegative_annual_real_budget_usd
        if pd.notna(budget):
            wealth = initial
            for idx in range(1, len(wealth_path)):
                wealth *= (
                    wealth_path.nominal_no_spending_usd.iloc[idx]
                    / wealth_path.nominal_no_spending_usd.iloc[idx - 1]
                )
                if wealth_path.year_end_payment.iloc[idx]:
                    wealth -= (
                        budget
                        * wealth_path.cpi_ratio.iloc[idx]
                        / (1 - row.cost_bps / BPS_DIVISOR)
                    )
                if wealth <= 0:
                    raise RuntimeError("Capacity path insolvent.")
            max_funding_error = max(
                max_funding_error, abs(wealth - initial * wealth_path.cpi_ratio.iloc[-1])
            )
        else:
            checks[f"unavailable_budget_explained_{row.strategy}_{row.cost_bps:g}"] = (
                real < initial
            )
    checks["funding_recursion_reconciles"] = max_funding_error < 1e-5


def _check_regression_references(
    path: Path,
    evaluated: pd.DataFrame,
    fund: pd.DataFrame,
    recalculated: dict,
    checks: dict,
) -> None:
    metrics = pd.read_csv(path / "portfolio_metrics.csv")
    gmv_metric = metrics[
        metrics.strategy.eq("sample_gmv") & metrics.cost_bps.eq(PRIMARY_COST_BPS)
    ].iloc[0]
    gmv_fund = fund[
        fund.strategy.eq("sample_gmv") & fund.cost_bps.eq(PRIMARY_COST_BPS)
    ].iloc[0]
    checks["observed_original_summary_reproduced"] = (
        abs(gmv_metric.cagr - PRIOR_REFERENCES["gmv_cagr_rounded"]) < 5e-7
        and abs(
            gmv_metric.maximum_drawdown_loss - PRIOR_REFERENCES["gmv_drawdown_rounded"]
        )
        < 5e-7
        and abs(
            gmv_fund.real_total_return
            - PRIOR_REFERENCES["gmv_real_total_return_rounded"]
        )
        < 5e-7
        and recalculated["holm_rejections"]
        == PRIOR_REFERENCES["primary_holm_rejections"]
    )
    for confidence, count in [
        (0.95, PRIOR_REFERENCES["garch_t_95_exceptions"]),
        (0.99, PRIOR_REFERENCES["garch_t_99_exceptions"]),
    ]:
        subset = evaluated[
            evaluated.strategy.eq("sample_gmv")
            & evaluated.method.eq("garch_t")
            & np.isclose(evaluated.confidence, confidence)
        ]
        eval_diag = evaluate_forecasts(subset)
        checks[f"selected_exceptions_match_original_{confidence}"] = (
            eval_diag["exceptions"] == count
        )
    mc = evaluated[evaluated.method.eq("monte_carlo")]
    checks["original_mc_flags_preserved"] = (
        int(mc.mc_agreement_with_analytic.eq(False).sum())
        == PRIOR_REFERENCES["monte_carlo_agreement_flag_rows"]
    )


def test_complete_scientific_evidence_reconciles():
    path = existing_final(ROOT)
    if path is None:
        raise RuntimeError("Final outputs missing.")
    summary = verify_run(path)
    lock = verify_lock(ROOT)
    cfg = settings(ROOT)
    dates = ["forecast_origin", "realization_date", "training_end"]
    issued = pd.read_csv(
        path / "forecasts_issued.csv", parse_dates=dates, low_memory=False
    )
    evaluated = pd.read_csv(
        path / "forecasts_evaluated.csv", parse_dates=dates, low_memory=False
    )
    diagnostics = pd.read_csv(path / "backtest_diagnostics.csv")
    parts = pd.read_csv(path / "es_contributions.csv")
    deploy = pd.read_csv(path / "deployment_risk.csv")
    deploy_parts = pd.read_csv(path / "deployment_es_contributions.csv")
    keys = ["strategy", "method", "forecast_origin", "confidence"]
    tot = parts.groupby(keys).agg(
        a=("es_component", "sum"), b=("portfolio_es", "first")
    )
    dtot = deploy_parts.groupby(["strategy", "method", "confidence"]).agg(
        a=("es_component", "sum"), b=("portfolio_es", "first")
    )
    contrib_error = float((tot.a - tot.b).abs().max())
    deploy_contrib_error = float((dtot.a - dtot.b).abs().max())
    obs_per_group = summary["observations_per_group"]
    assessment = read_assessment(path)
    family = pd.read_csv(path / "final_calibration_family.csv")
    opening = pd.Timestamp(summary["opened_at_utc"])
    checks = {
        "original_recovery_contract_precedes_reexecution": pd.Timestamp(
            lock["source_recovery_contract_at_utc"]
        )
        < opening,
        "frozen_inputs_and_selected_procedure_verified": True,
        "full_grid": len(evaluated)
        == len(issued)
        == EXPECTED_FORECAST_GROUPS * obs_per_group
        and len(diagnostics) == EXPECTED_FORECAST_GROUPS,
        "matched_groups": diagnostics.observations.eq(obs_per_group).all()
        and diagnostics.forecast_gaps.eq(0).all(),
        "unique_forecasts": not evaluated.duplicated(keys).any(),
        "final_dates_only": evaluated.realization_date.between(
            "2020-01-01", "2026-10-23"
        ).all(),
        "origin_before_outcome": (
            evaluated.forecast_origin < evaluated.realization_date
        ).all(),
        "past_only_training": evaluated.training_end.eq(evaluated.forecast_origin).all()
        and evaluated.training_observations.eq(TRAINING_LOOKBACK).all(),
        "issued_evaluated_identity": evaluated[issued.columns].equals(issued),
        "issued_without_outcomes": not any(
            c.startswith("realized_") or c == "exception" for c in issued
        ),
        "finite_forecasts": np.isfinite(evaluated[["var", "es", "realized_hpl_loss"]])
        .all()
        .all(),
        "consistent_es": (evaluated.es >= evaluated["var"] - 1e-12).all(),
        "components_reconcile": contrib_error < 1e-12,
        "deployment_components_reconcile": deploy_contrib_error < 1e-12,
        "dollar_var_es_reconcile": np.allclose(
            evaluated.var_usd,
            evaluated["var"] * evaluated.origin_value_usd,
            atol=1e-6,
            rtol=1e-12,
        )
        and np.allclose(
            evaluated.es_usd,
            evaluated.es * evaluated.origin_value_usd,
            atol=1e-6,
            rtol=1e-12,
        ),
        "fallbacks_disclosed": evaluated.loc[
            evaluated.fit_status.eq("FALLBACK"), "fallback_reason"
        ]
        .notna()
        .all(),
        "four_primary_tests": len(family) == 4,
        "holm_recomputes": np.allclose(
            family.holm_pvalue, holm_adjust(family.family_input_pvalue.to_numpy())
        ),
        "selection_unchanged": assessment["selection_changes"] == []
        and not assessment["monitor"]["replacement_selected"],
        "human_approval_pending": assessment["committee"]["human_decision"] == "PENDING"
        and assessment["committee"]["approved_allocation"] is None,
        "deployment_no_fabricated_outcome": deploy.realization_date.isna().all()
        and not any(c.startswith("realized_") for c in deploy),
        "deployment_capital_cutoff_grid": len(deploy) == EXPECTED_FORECAST_GROUPS
        and deploy.origin_value_usd.eq(REFERENCE_CAPITAL_USD).all()
        and pd.to_datetime(deploy.forecast_origin).eq(pd.Timestamp("2026-10-23")).all(),
        "no_temporary_dependencies": not any(path.rglob("*.pending")),
    }
    for field in ["var", "es"]:
        pivot = evaluated.pivot(
            index=["strategy", "method", "forecast_origin"],
            columns="confidence",
            values=field,
        )
        checks[field + "_monotonic"] = (
            (pivot.diff(axis=1).iloc[:, 1:] >= -1e-12).all().all()
        )
    _check_accounting_and_hpl_invariants(path, cfg, evaluated, checks)
    _check_stress_attribution_invariants(path, checks)
    fund = pd.read_csv(path / "funding_diagnostics.csv")
    paths = pd.read_csv(path / "real_wealth_paths.csv")
    _check_funding_invariants(cfg, summary, fund, paths, checks)
    recalculated, tables = assess_monitor(evaluated, diagnostics, cfg)
    checks["assessment_recomputes"] = recalculated == assessment["monitor"]
    original = pd.read_csv(path / "final_score_comparisons.csv")
    checks["score_intervals_recompute"] = np.allclose(
        original[["mean_difference", "ci_low", "ci_high"]],
        tables["final_score_comparisons"][["mean_difference", "ci_low", "ci_high"]],
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True,
    )
    checks["committee_assessment_recomputes"] = (
        committee_assessment(recalculated, fund, cfg) == assessment["committee"]
    )
    _check_regression_references(path, evaluated, fund, recalculated, checks)
    checks = {k: bool(v) for k, v in checks.items()}
    assert all(checks.values()), "Failed scientific checks: " + str(
        [key for key, passed in checks.items() if not passed]
    )
    assert len(checks) == EXPECTED_CHECK_COUNT
