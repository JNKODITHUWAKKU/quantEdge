"""Presentation derivatives from frozen research, with no optimization or fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

ORDER = ["sample_gmv", "shrinkage_gmv", "policy_40_60", "equal_weight", "agg_only"]
LABELS = {
    "sample_gmv": "Sample GMV (selected)",
    "shrinkage_gmv": "Shrinkage GMV",
    "policy_40_60": "40% SPY / 60% AGG",
    "equal_weight": "Equal weight",
    "agg_only": "AGG only",
}
SELECTED_STRATEGY = "sample_gmv"
EXTENDED_GFC_SCENARIO = "GFC_extended"
COST_SCENARIOS_BPS = (0.0, 5.0, 10.0)
EXPECTED_STRATEGY_COUNT = len(ORDER)
EXPECTED_STRATEGY_COST_ROWS = EXPECTED_STRATEGY_COUNT * len(COST_SCENARIOS_BPS)
SPENDING_RATES = (0.0, 0.02, 0.03, 0.04)
LOSS_TOLERANCES = (0.15, 0.20, 0.25)
REFERENCE_CAPITAL_USD = 1e7


def _validate_baseline_inputs(
    metrics: pd.DataFrame, funding: pd.DataFrame, gfc_stress: pd.DataFrame
) -> None:
    if (
        len(metrics) != EXPECTED_STRATEGY_COST_ROWS
        or set(metrics.strategy) != set(ORDER)
        or set(metrics.cost_bps) != set(COST_SCENARIOS_BPS)
        or metrics.duplicated(["strategy", "cost_bps"]).any()
        or not metrics.groupby("cost_bps").strategy.nunique().eq(EXPECTED_STRATEGY_COUNT).all()
    ):
        raise ValueError("Expected complete unique five-strategy/cost records")
    for key in ["n_returns", "start_valuation_date", "first_return_date", "end_date"]:
        if metrics[key].isna().any() or metrics[key].nunique(dropna=False) != 1:
            raise ValueError("Baseline comparison has unmatched " + key)
    if funding.duplicated(["strategy", "cost_bps"]).any():
        raise ValueError("Duplicate funding records")
    if set(zip(funding.strategy, funding.cost_bps)) != set(
        zip(metrics.strategy, metrics.cost_bps)
    ):
        raise ValueError("Incomplete funding comparison records")
    if funding.last_valuation_date.nunique() != 1:
        raise ValueError("Unmatched real endpoints")
    if len(gfc_stress) != EXPECTED_STRATEGY_COUNT or gfc_stress.strategy.duplicated().any():
        raise ValueError("Incomplete extended-GFC comparison")
    for key in [
        "requested_start",
        "requested_end",
        "base_date",
        "first_return_date",
        "last_date",
        "weights_origin",
        "replay_type",
    ]:
        if gfc_stress[key].nunique(dropna=False) != 1:
            raise ValueError("Unmatched stress " + key)


def matched_baselines(metrics, funding, stress):
    metrics_copy = metrics.copy()
    gfc_stress = stress[stress.scenario.eq(EXTENDED_GFC_SCENARIO)].copy()
    _validate_baseline_inputs(metrics_copy, funding, gfc_stress)
    cols = [
        "strategy",
        "cost_bps",
        "n_returns",
        "first_return_date",
        "end_date",
        "cagr",
        "annualized_volatility",
        "maximum_drawdown_loss",
        "sharpe",
        "total_trading_cost_dollars",
    ]
    out = metrics_copy[cols].merge(
        funding[["strategy", "cost_bps", "real_total_return", "last_valuation_date"]],
        on=["strategy", "cost_bps"],
        validate="one_to_one",
    )
    out = out.merge(
        gfc_stress[["strategy", "maximum_drawdown_loss", "weights_origin"]].rename(
            columns={"maximum_drawdown_loss": "gfc_extended_replay_drawdown"}
        ),
        on="strategy",
        validate="many_to_one",
    )
    out["portfolio"] = out.strategy.map(LABELS)
    out["research_role"] = np.where(
        out.strategy.eq(SELECTED_STRATEGY),
        "Selected procedure",
        "Original research baseline",
    )
    out["evidence_class"] = "Frozen retrospective comparison; no reselection"
    out["strategy_order"] = out.strategy.map({v: i for i, v in enumerate(ORDER)})
    return (
        out.sort_values(["cost_bps", "strategy_order"])
        .drop(columns="strategy_order")
        .reset_index(drop=True)
    )


def funding_map(paths, metrics, stress, hypothetical, assessment):
    rows = []
    calibration_ok = (
        int(assessment["monitor"]["holm_rejections"]) == 0
        and int(assessment["monitor"]["missing_tests"]) == 0
    )
    replay_drawdown = float(
        stress[stress.strategy.eq(SELECTED_STRATEGY)].maximum_drawdown_loss.max()
    )
    shock_loss = float(
        hypothetical[hypothetical.strategy.eq(SELECTED_STRATEGY)].loss_fraction.max()
    )
    for cost in COST_SCENARIOS_BPS:
        wealth_path = paths[
            paths.strategy.eq(SELECTED_STRATEGY) & paths.cost_bps.eq(cost)
        ].copy()
        wealth_path["valuation_date"] = pd.to_datetime(wealth_path.valuation_date)
        wealth_path = wealth_path.sort_values("valuation_date")
        if wealth_path.empty or wealth_path.valuation_date.duplicated().any():
            raise ValueError("Incomplete/duplicate real wealth path")
        last = wealth_path.iloc[-1]
        metric_row = metrics[metrics.strategy.eq(SELECTED_STRATEGY) & metrics.cost_bps.eq(cost)]
        if (
            len(metric_row) != 1
            or not np.isfinite(
                [
                    last.cpi_ratio,
                    last.nominal_no_spending_usd,
                    last.withdrawal_terminal_coefficient,
                ]
            ).all()
            or last.cpi_ratio <= 0
        ):
            raise ValueError("Missing or invalid final metric or CPI endpoint")
        observed = float(metric_row.maximum_drawdown_loss.iloc[0])
        for spend in SPENDING_RATES:
            # Terminal real wealth is linear in fixed real spending via the precomputed withdrawal coefficient.
            terminal = float(
                (
                    last.nominal_no_spending_usd
                    - spend * REFERENCE_CAPITAL_USD * last.withdrawal_terminal_coefficient
                )
                / last.cpi_ratio
            )
            for tolerance in LOSS_TOLERANCES:
                checks = {
                    "real_principal_floor_met": terminal >= REFERENCE_CAPITAL_USD,
                    "observed_drawdown_met": observed <= tolerance,
                    "largest_replay_met": replay_drawdown <= tolerance,
                    "largest_hypothetical_met": shock_loss <= tolerance,
                    "calibration_screen_met": calibration_ok,
                }
                failed = [k for k, v in checks.items() if not v]
                rows.append(
                    {
                        "cost_bps": cost,
                        "annual_real_spending": spend,
                        "illustrative_loss_tolerance": tolerance,
                        "terminal_real_usd": terminal,
                        "real_endpoint": str(last.valuation_date.date()),
                        "completed_december_payments": int(wealth_path.year_end_payment.sum()),
                        "observed_drawdown": observed,
                        "largest_replay_drawdown": replay_drawdown,
                        "largest_hypothetical_loss": shock_loss,
                        **checks,
                        "all_numeric_screens_met": all(checks.values()),
                        "investment_approved": False,
                        "decision": "HOLD / NO TRADE",
                        "failed_screens": "; ".join(failed),
                        "assumption_status": "Post-evaluation managerial illustration; not adopted client terms",
                    }
                )
    return pd.DataFrame(rows)


def committee_actions():
    return pd.DataFrame(
        [
            {
                "Action": "Withhold complete-allocation deployment",
                "Proposed owner": "Investment committee",
                "Required evidence": "Named, dated human decision and authorized mandate",
                "Status": "Investment approval pending",
            },
            {
                "Action": "Keep GARCH-t in research/shadow use; no automated trades",
                "Proposed owner": "Risk lead",
                "Required evidence": "Acknowledged calibration failure and human escalation process",
                "Status": "Proposed control; human owner pending",
            },
            {
                "Action": "Define currency, liability basket, payout, fees and liquidity needs",
                "Proposed owner": "Investment / finance lead",
                "Required evidence": "Adopted mandate, loss limits and feasible liquidity plan within allowed instruments",
                "Status": "Mandate and reserve not established",
            },
            {
                "Action": "Independently review material source and mathematical checks",
                "Proposed owner": "Independent human reviewers",
                "Required evidence": "Signed formula/source review with unresolved findings retained",
                "Status": "Human review pending",
            },
            {
                "Action": "Complete graphical browser and separate-machine acceptance",
                "Proposed owner": "Release / QA lead",
                "Required evidence": "Dated control/download checks and environment/test records",
                "Status": "External acceptance pending",
            },
            {
                "Action": "Rehearse and approve final research presentation",
                "Proposed owner": "Team / presentation lead",
                "Required evidence": "Observed 10–12 minute rehearsal and signed research review",
                "Status": "Human rehearsal/sign-off pending",
            },
        ]
    )


def architecture():
    return pd.DataFrame(
        [
            {
                "Stage": "Allocation",
                "Frequency": "Quarterly signals",
                "Input": "Trailing 1,260 ETF simple returns known at signal close",
                "Rule": "Sample covariance; long-only fully invested minimum variance",
                "Output": "Target executes next session close; following return uses new holdings",
            },
            {
                "Stage": "Risk training",
                "Frequency": "Each forecast origin",
                "Input": "Origin exposure w_t and last 1,260 asset returns R_s",
                "Rule": "Recast r_s = w_tᵀ R_s, then subtract its window mean",
                "Output": "One centered portfolio-exposure series, not seven separate GARCH fits",
            },
            {
                "Stage": "Risk issuance",
                "Frequency": "Each origin; one-session horizon",
                "Input": "Centered exposure series; known origin data only",
                "Rule": "Refit GARCH(1,1)-t; standardize t to unit variance",
                "Output": "VaR/ES at 90/95/99%; outcome joined afterwards",
            },
        ]
    )
