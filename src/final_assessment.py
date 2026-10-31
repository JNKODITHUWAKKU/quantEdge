"""Assess fixed roles; never select a replacement using final outcomes."""

import numpy as np
import pandas as pd

from src.risk import seeded_identifier
from src.risk_validation import joint_fz0
from src.selection import holm_adjust, simultaneous_mean_intervals

PRIMARY_CONFIDENCES = (0.95, 0.99)
BASEL_CONFIDENCE = 0.99
SUPPORT_TOLERANCE = 1e-12
CALIBRATION_TESTS = (
    ("exact_coverage", "exact_binomial_pvalue"),
    ("christoffersen", "christoffersen_pvalue"),
)


def _build_calibration_family(
    diagnostics: pd.DataFrame, name: str, method: str, calibration_alpha: float
) -> pd.DataFrame:
    rows = []
    for confidence in PRIMARY_CONFIDENCES:
        match = diagnostics[
            diagnostics.strategy.eq(name)
            & diagnostics.method.eq(method)
            & np.isclose(diagnostics.confidence, confidence)
        ]
        if len(match) != 1:
            raise ValueError("Incomplete primary diagnostic family.")
        diag_row = match.iloc[0]
        for test_name, pvalue_key in CALIBRATION_TESTS:
            available = bool(pd.notna(diag_row[pvalue_key]))
            # Missing transition tests enter the 4-test Holm family at p=1.0 so family size stays fixed.
            rows.append(
                {
                    "strategy": name,
                    "method": method,
                    "confidence": confidence,
                    "test": test_name,
                    "raw_pvalue": float(diag_row[pvalue_key]) if available else np.nan,
                    "available": available,
                    "family_input_pvalue": float(diag_row[pvalue_key]) if available else 1.0,
                    "independence_status": diag_row.independence_status,
                }
            )
    family = pd.DataFrame(rows)
    family["holm_pvalue"] = holm_adjust(family.family_input_pvalue.to_numpy())
    family["reject"] = family.available & family.holm_pvalue.lt(calibration_alpha)
    return family


def _compute_paired_score_intervals(
    forecasts: pd.DataFrame, name: str, method: str, spec: dict, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    refs = spec["score_references"]
    methods = [method, *refs]
    confidence = spec["score_confidence"]
    scores = forecasts[
        forecasts.strategy.eq(name)
        & forecasts.method.isin(methods)
        & np.isclose(forecasts.confidence, confidence)
    ].copy()
    scores["joint_score"] = joint_fz0(
        scores.realized_hpl_loss.to_numpy(),
        scores["var"].to_numpy(),
        scores.es.to_numpy(),
        confidence,
    )
    if scores.duplicated(["method", "realization_date"]).any():
        raise ValueError("Duplicate paired score.")
    matrix = scores.pivot(
        index="realization_date", columns="method", values="joint_score"
    ).sort_index()
    if len(scores) != len(matrix) * len(methods):
        raise ValueError("Incomplete paired scores.")
    delta = np.column_stack([(matrix[method] - matrix[ref]).to_numpy() for ref in refs])
    interval_rows = []
    for length in spec["block_lengths"]:
        estimates = simultaneous_mean_intervals(
            delta,
            length,
            spec["bootstrap_replicates"],
            seeded_identifier(seed, "final_score", length),
            spec["confidence_interval"],
        )
        interval_rows.extend(
            {
                **estimate,
                "strategy": name,
                "selected_monitor": method,
                "reference": ref,
                "confidence": confidence,
            }
            for estimate, ref in zip(estimates, refs)
        )
    intervals = pd.DataFrame(interval_rows)
    adverse = [
        ref
        for ref in refs
        if intervals[intervals.reference.eq(ref)].status.eq("OK").all()
        and intervals[intervals.reference.eq(ref)].ci_low.gt(0).all()
    ]
    return scores, intervals, adverse


def _is_valid_forecast_support(chosen: pd.DataFrame) -> bool:
    return bool(
        np.isfinite(chosen[["var", "es"]]).all().all()
        and chosen.es.gt(0).all()
        and (chosen.es >= chosen["var"] - SUPPORT_TOLERANCE).all()
        and chosen.es.le(1).all()
        and chosen["var"].le(1).all()
    )


def _collect_review_reasons(
    family: pd.DataFrame,
    support: bool,
    basel_zone: str,
    adverse: list[str],
    intervals: pd.DataFrame,
) -> list[str]:
    reasons = []
    if family.reject.any():
        reasons.append("PREDECLARED_CALIBRATION_REJECTION")
    if not family.available.all():
        reasons.append("SELECTED_DIAGNOSTIC_UNAVAILABLE")
    if not support:
        reasons.append("INVALID_FORECAST_DOMAIN_OR_SUPPORT")
    if basel_zone == "RED":
        reasons.append("LATEST_250_BASEL_RED")
    if adverse:
        reasons.append("ROBUST_ADVERSE_PAIRED_SCORE")
    if not intervals.status.eq("OK").all():
        reasons.append("SCORE_UNCERTAINTY_NOT_ESTIMABLE")
    return reasons


def assess_monitor(forecasts, diagnostics, cfg):
    spec = cfg["assessment"]
    name = spec["portfolio"]
    method = spec["monitor"]
    if (
        not pd.to_datetime(forecasts.realization_date)
        .between(cfg["run"]["start"], cfg["run"]["end"])
        .all()
    ):
        raise ValueError("Dates outside final block.")
    chosen = forecasts[forecasts.strategy.eq(name) & forecasts.method.eq(method)]
    if chosen.empty:
        raise ValueError("Selected procedure missing.")
    family = _build_calibration_family(diagnostics, name, method, spec["calibration_alpha"])
    scores, intervals, adverse = _compute_paired_score_intervals(
        forecasts, name, method, spec, cfg["run"]["seed"]
    )
    diag_99 = diagnostics[
        diagnostics.strategy.eq(name)
        & diagnostics.method.eq(method)
        & np.isclose(diagnostics.confidence, BASEL_CONFIDENCE)
    ].iloc[0]
    support = _is_valid_forecast_support(chosen)
    reasons = _collect_review_reasons(family, support, str(diag_99.basel_zone), adverse, intervals)
    result = {
        "selected_portfolio": name,
        "selected_monitor": method,
        "status": "MONITOR_REQUIRES_REVIEW"
        if reasons
        else "NO_PREDECLARED_REJECTION_WITH_LIMITATIONS",
        "review_reasons": reasons,
        "holm_rejections": int(family.reject.sum()),
        "missing_tests": int((~family.available).sum()),
        "domain_and_support_valid": support,
        "latest_250_basel99": str(diag_99.basel_zone),
        "robust_adverse_score_references": adverse,
        "fallback_forecasts": int(chosen.fit_status.eq("FALLBACK").sum()),
        "replacement_selected": False,
        "investment_approval": "PENDING",
        "interpretation": "Non-rejection is not proof; sparse transitions, dependence and retrospective recovery limit inference.",
    }
    return result, {
        "final_calibration_family": family,
        "final_score_comparisons": intervals,
        "final_score_observations": scores[
            [
                "forecast_origin",
                "realization_date",
                "strategy",
                "method",
                "confidence",
                "joint_score",
            ]
        ],
    }


def committee_assessment(monitor, funding, cfg):
    row = funding[
        funding.strategy.eq(cfg["assessment"]["portfolio"])
        & funding.cost_bps.eq(cfg["run"]["primary_cost_bps"])
    ]
    if len(row) != 1:
        raise ValueError("Selected funding row missing.")
    floor = bool(row.iloc[0].preserved_real_principal_before_spending)
    return {
        "proposal": "sample_gmv as the complete USD endowment allocation",
        "research_assessment": "ADJUST_OR_REJECT_COMPLETE_ENDOWMENT_PROPOSAL"
        if not floor
        else "CONDITIONAL_REVIEW_REQUIRED",
        "zero_spend_real_principal_floor_met": floor,
        "monitor_status": monitor["status"],
        "human_decision": "PENDING",
        "human_review": "PENDING",
        "approved_allocation": None,
        "no_test_set_reselection": True,
        "action": "Resolve spending, liability currency, loss tolerance and human review before an investment decision.",
        "limitation": "Nominal minimum variance does not optimize real funding survival; an endpoint pass would not prove suitability.",
    }
