"""Original numerical research settings, consolidated into Python.

These values are copied from the established research configuration. They are
not fitted or selected using final-period outcomes. Change control belongs to
the research protocol, not to interactive dashboard selectors.
"""

from copy import deepcopy

PORTFOLIO_SETTINGS = {
    "run": {
        "snapshot_id": "20261024T074456Z_518bd91b",
        "evidence_class": "exploratory_development_only",
        "end": "2014-12-31",
        "lookback_returns": 1260,
        "rebalance_frequency": "quarterly",
        "initial_value_usd": 10000000,
        "annualization": 252,
        "hac_lags": 5,
    },
    "costs": {
        "basis_points_per_dollar_traded": [0.0, 5.0, 10.0],
        "display_scenario_basis_points": 5.0,
        "charge_initial_purchase": True,
        "terminal_liquidation": False,
        "fund_expenses": "already embedded in ETF returns; no second deduction",
        "taxes": "not modelled; no assertion of tax exemption",
    },
    "portfolio": {
        "long_only": True,
        "fully_invested": True,
        "position_cap": 1.0,
        "policy_benchmark": "40% SPY / 60% AGG; proposed managerial comparator",
        "strategies": ["policy_40_60", "equal_weight", "agg_only", "sample_gmv", "shrinkage_gmv"],
        "market_proxy": "SPY",
        "sortino_target": "lagged DGS3MO cash proxy; not a spending target",
    },
    "frontier": {
        "points": 11,
        "mean_sensitivity_target_fraction": 0.5,
        "mean_sensitivity_shock": "plus/minus one HAC standard error for one asset at a time",
        "target_interpretation": "diagnostic frontier location only; not client return objective",
    },
    "controls": {
        "feasibility_tolerance": 1e-08,
        "optimality_gap_tolerance": 1e-06,
        "source_human_review": "PENDING",
        "model_human_review": "PENDING",
        "prior_holdout_exposure": "UNDECLARED",
        "recommendation_status": "NO_RECOMMENDATION",
    },
}

RISK_SETTINGS = {
    "run": {
        "end": "2014-12-31",
        "evidence_class": "exploratory_development_only",
        "lookback_returns": 1260,
        "confidence_levels": [0.9, 0.95, 0.99],
        "methods": ["historical", "gaussian", "monte_carlo"],
        "holdings_cost_bps": 5.0,
        "horizon_sessions": 1,
        "pnl_type": "frozen_exposure_market_loss",
        "mean_estimator": "daily_sample_mean",
        "covariance_estimator": "daily_sample_covariance_ddof1",
        "seed": 20261025,
    },
    "monte_carlo": {
        "batches": 16,
        "initial_draws_per_batch": 16384,
        "maximum_draws_per_batch": 524288,
        "normal_design_variance_buffer": 1.5,
        "target_ci95_halfwidth_return": 0.0001,
        "agreement_standard_errors": 6.0,
        "asset_level_check_draws_per_batch": 65536,
        "interval_note": "Approximate conditional MC interval; adaptive precision "
        "stopping; not parameter/model uncertainty",
    },
    "bootstrap": {
        "replicates": 500,
        "block_lengths": [10, 20, 60],
        "scope": "last development forecast origin only",
        "interval": "95% percentile sensitivity, conditional on holdings and historical window",
    },
    "validation": {
        "exception_rule": "realized_loss_strictly_greater_than_var",
        "test_size": 0.05,
        "basel_confidence": 0.99,
        "basel_observations": 250,
        "score": "quantile_loss_and_FZ0_positive_ES_domain",
        "model_selection": "NONE; development diagnostics only",
    },
}

CONDITIONAL_SETTINGS = {
    "run": {
        "end": "2014-12-31",
        "evidence_class": "exploratory_development_only",
        "lookback_returns": 1260,
        "confidence_levels": [0.9, 0.95, 0.99],
        "seed": 20261026,
        "mean": "same_window_sample_mean",
    },
    "filter": {
        "ewma_lambda": 0.94,
        "initialization_sessions": 60,
        "garch": "GARCH(1,1); externally sample-centred returns; zero-mean fit; standardized "
        "Student-t",
        "garch_max_iterations": 500,
        "garch_tolerance": 1e-08,
        "garch_near_integration": 0.995,
        "failure_fallback": "EWMA filtered historical simulation; retain requested model and "
        "reason",
    },
    "evt": {
        "threshold_probabilities": [0.8, 0.85, 0.9, 0.925, 0.95],
        "minimum_exceedances": 50,
        "minimum_stable_thresholds": 3,
        "maximum_shape_span": 0.2,
        "maximum_modified_scale_span_over_median_scale": 0.25,
        "maximum_q99_relative_span": 0.2,
        "shape_lower_numeric_bound": -0.95,
        "shape_upper_numeric_bound": 2.0,
        "rule": "longest contiguous stable region; ties prefer lower region; lower central "
        "threshold",
        "unstable_fallback": "matching historical/FHS distribution; never silently drop a forecast",
        "reference_threshold_for_unstable_diagnostics": 0.9,
        "bootstrap_replicates": 200,
        "bootstrap_block_length": 20,
        "bootstrap_scope": "last development origin; refit filter and GPD; refilter observed "
        "origin history with bootstrap parameters; percentile fixed; selection "
        "uncertainty excluded",
        "gof_replicates": 199,
        "diagnostic_origins": ["2009-12-30", "2011-12-29", "2013-12-30", "2014-12-30"],
    },
    "stress": {
        "value_usd": 10000000,
        "hypothetical_horizon_sessions": 20,
        "hypothetical_returns": [-0.2, -0.22, -0.28, -0.09, -0.07, 0.05, -0.3],
        "assets": ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"],
        "hypothetical_scale_factors": [0.5, 1.0, 1.5],
        "correlation_mixture": [0.0, 0.5, 1.0],
        "volatility_multipliers": [1.0, 1.5, 2.0],
        "final_period_stresses": "WITHHELD_UNTIL_SELECTION_FREEZE",
    },
    "portfolio_research": {
        "windows": [756, 1260],
        "mean_retention": [1.0, 0.5, 0.0],
        "mean_target": "common cross-asset sample mean; diagnostic bias/variance "
        "experiment, not economic prior",
        "covariance_diagonal_shrinkage": [0.0, 0.25, 0.5],
        "agg_cap_sensitivity": [1.0, 0.8, 0.6, 0.4],
        "status": "snapshot robustness only; no cap, target or allocation selected",
    },
}

VALIDATION_SETTINGS = {
    "run": {
        "start": "2015-01-01",
        "end": "2019-12-31",
        "final_start": "2020-01-01",
        "evidence_class": "model_selection_validation",
        "lookback_returns": 1260,
        "confidence_levels": [0.9, 0.95, 0.99],
        "strategies": ["policy_40_60", "equal_weight", "agg_only", "sample_gmv", "shrinkage_gmv"],
        "cost_scenarios_bps": [0.0, 5.0, 10.0],
        "primary_cost_bps": 5.0,
        "reference_capital_usd": 10000000,
        "seed": 20261027,
        "annualization": 252,
    },
    "risk_selection": {
        "candidate_order": ["ewma_fhs", "garch_t", "garch_fhs", "garch_pot_guarded"],
        "default": "ewma_fhs",
        "primary_confidence": 0.95,
        "safeguard_confidence": 0.99,
        "primary_score": "FZ0; daily equal-weight average over all five prescribed "
        "portfolio candidates",
        "confidence_interval": 0.95,
        "bootstrap_replicates": 4999,
        "block_lengths": [10, 20, 40],
        "primary_block_length": 20,
        "family": "all six ordered complexity-pair contrasts; simultaneous "
        "max-absolute-standardized-bootstrap intervals",
        "calibration_family_alpha": 0.05,
        "calibration_family": "80 predeclared tests: four candidates x five portfolios "
        "x two confidences x exact coverage/independence",
        "minimum_improved_calendar_years": 3,
        "state_volatility_window": 21,
        "state_median_window": 1260,
        "require_both_volatility_states_nonworse": True,
        "evt_must_also_beat": ["garch_t", "garch_fhs"],
        "no_eligible_model_action": "NO_MODEL_APPROVED; display EWMA-FHS only as a "
        "flagged operational reference",
        "adoption_note": "Research choice only; calibration non-rejection and point "
        "safeguards are not guarantees.",
    },
    "portfolio_selection": {
        "candidates": ["sample_gmv", "shrinkage_gmv"],
        "default": "sample_gmv",
        "objective": "matched validation net-return variance at five bp; "
        "shrinkage must demonstrate improvement",
        "require_variance_ci_upper_below_zero_all_blocks": True,
        "require_variance_nonworse_all_costs": True,
        "require_empirical_es95_nonworse_all_costs": True,
        "require_drawdown_nonworse_all_costs": True,
        "require_worst_hypothetical_loss_nonworse": True,
        "constraints": "long-only; fully invested; no leverage; quarterly; no new "
        "cap or return target",
        "status": "Capital-preservation research candidate; "
        "real-growth/income/suitability not approved",
    },
    "calibration_diagnostics": {
        "es_moment_bootstrap_replicates": 1999,
        "es_moment_block_length": 20,
        "confidence_interval": 0.95,
        "scope": "pointwise normalized ES identification-moment intervals; "
        "diagnostic only; no model selection",
    },
}

FINAL_SETTINGS = {
    "run": {
        "start": "2020-01-01",
        "end": "2026-10-23",
        "evidence_class": "locked_final_retrospective_recovery",
        "lookback_returns": 1260,
        "confidence_levels": [0.9, 0.95, 0.99],
        "strategies": ["policy_40_60", "equal_weight", "agg_only", "sample_gmv", "shrinkage_gmv"],
        "cost_scenarios_bps": [0.0, 5.0, 10.0],
        "primary_cost_bps": 5.0,
        "reference_capital_usd": 10000000,
        "annualization": 252,
        "seed": 20261027,
    },
    "assessment": {
        "portfolio": "sample_gmv",
        "monitor": "garch_t",
        "calibration_alpha": 0.05,
        "score_confidence": 0.95,
        "score_references": ["historical", "ewma_fhs"],
        "bootstrap_replicates": 4999,
        "block_lengths": [10, 20, 40],
        "confidence_interval": 0.95,
    },
    "calibration_diagnostics": {
        "es_moment_bootstrap_replicates": 1999,
        "es_moment_block_length": 20,
        "confidence_interval": 0.95,
    },
    "funding": {"cpi_file": "data/CPIAUCNS.csv", "base_month": "2019-12", "last_month": "2026-09"},
}

SOURCE_SETTINGS = {
    "data": {
        "tickers": ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"],
        "start": "2000-01-01",
        "cutoff": "2026-10-23",
        "calendar": "XNYS",
        "currency": "USD",
        "fred_series": ["DGS3MO"],
        "timeout_seconds": 25,
        "auto_adjust": False,
        "repair": False,
        "keepna": True,
    },
    "conventions": {
        "portfolio_value_usd": 10000000,
        "trading_days_per_year": 252,
        "strategic_rebalance": "quarterly, subject to validation",
        "loss_sign": "positive numbers denote losses",
        "price_return": "simple adjusted-price return for wealth and portfolio accounting",
        "diagnostic_return": "log adjusted-price return for ADF and appropriate features",
        "rate_series": "DGS3MO",
        "rate_use": "approximate cash-rate benchmark; not a tradable total-return index",
        "rate_assumed_availability_lag_sessions": 2,
        "rate_max_age_calendar_days": 10,
        "rate_day_count": "ACT/365 simple accrual approximation",
    },
}

# The digest identifies the original CSV snapshot; no metadata sidecars are needed.
INPUT_DIGEST = "3d1066053afc7e4f8865ccf75c8b1fc3dbfeeb7c96bb2bf05a42fdf7984d8e8c"
ORIGINAL_FINAL_LOCK = "stage6_recovery_568acecc83ad264b"
ORIGINAL_SELECTION_LOCK = "stage5_da46c31f31fd32c7"
ORIGINAL_RECOVERY_DATE = "2026-10-27T13:54:14.854535+00:00"


def final_settings() -> dict:
    """Return an isolated copy of the fixed final-evaluation settings."""
    return deepcopy(FINAL_SETTINGS)
