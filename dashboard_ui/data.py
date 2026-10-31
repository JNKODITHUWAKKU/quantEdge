"""Read-only access to verified financial CSVs; no fitting or market access."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from dashboard_ui.disclosures import (
    CONFIDENCE_LEVELS,
    COST_SCENARIOS_BPS,
    EXPECTED_ASSETS,
    GUARDED_EVT_METHOD,
    METHODS,
    SELECTED_MONITOR,
    SELECTED_STRATEGY,
)
from src.storage import verify_run, read_assessment, read_summary

MIN_BACKTEST_SESSIONS = 20

TABLE_NAMES = (
    "portfolio_metrics",
    "funding_diagnostics",
    "real_wealth_paths",
    "deployment_weights",
    "latest_held_weights",
    "deployment_risk",
    "deployment_es_contributions",
    "deployment_threshold_diagnostics",
    "deployment_evt_qqpp",
    "deployment_evt_gof",
    "deployment_evt_bootstrap",
    "deployment_filter_diagnostics",
    "historical_stress",
    "historical_stress_components",
    "hypothetical_stress",
    "hypothetical_components",
    "correlation_stress",
    "causal_strategy_crises",
    "backtest_diagnostics",
    "final_calibration_family",
    "final_score_comparisons",
    "es_calibration",
    "stationarity_diagnostics",
    "selected_monitor_basel_history",
    "deployment_optimization",
    "monte_carlo_agreement_flags",
)

FORECAST_COLUMNS = [
    "forecast_origin",
    "realization_date",
    "strategy",
    "method",
    "confidence",
    "var",
    "es",
    "realized_hpl_loss",
    "pnl_type",
    "fit_status",
    "actual_model_used",
]

REQUIRED_TABLE_COLUMNS = {
    "portfolio_metrics": (
        "cost_bps",
        "cagr",
        "maximum_drawdown_loss",
        "terminal_wealth_usd",
    ),
    "funding_diagnostics": (
        "cost_bps",
        "real_total_return",
        "terminal_real_no_spending_usd",
    ),
    "deployment_weights": ("asset", "weight", "reference_notional_usd"),
    "deployment_risk": (
        "method",
        "confidence",
        "var",
        "es",
        "var_usd",
        "es_usd",
        "fit_status",
        "actual_model_used",
    ),
}

RISK_VALUE_COLUMNS = ["var", "es", "var_usd", "es_usd"]


@dataclass(frozen=True)
class FilterSelection:
    """User-selected display filters in the sidebar."""

    workspace: str
    cost_bps: float
    confidence: float
    method: str
    dates: tuple

    def __iter__(self):
        return iter(
            (self.workspace, self.cost_bps, self.confidence, self.method, self.dates)
        )


@dataclass(frozen=True)
class DateWindow:
    """Validated backtest date slice over the daily accounting series."""

    shown: pd.DataFrame
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    error: str | None


@st.cache_data(show_spinner=False)
def _tables(path_string: str, results_digest: str) -> dict:
    del results_digest
    path = Path(path_string)
    verify_run(path)
    data = {name: pd.read_csv(path / f"{name}.csv") for name in TABLE_NAMES}
    data["forecasts"] = pd.read_csv(
        path / "forecasts_evaluated.csv",
        usecols=FORECAST_COLUMNS,
        parse_dates=["forecast_origin", "realization_date"],
    )
    data["assessment"] = read_assessment(path)
    data["summary"] = read_summary(path)
    validate_tables(data)
    return data


def subset(
    frame: pd.DataFrame, strategy: str, confidence: float | None = None
) -> pd.DataFrame:
    if "strategy" not in frame:
        return frame.iloc[:0]
    result = frame[frame.strategy.eq(strategy)].copy()
    if confidence is None:
        return result
    return result[np.isclose(result.confidence, confidence)]


def _validate_required_columns(data: dict) -> None:
    for name, columns in REQUIRED_TABLE_COLUMNS.items():
        frame = subset(data[name], SELECTED_STRATEGY)
        if frame.empty or not set(columns).issubset(frame.columns):
            raise ValueError(f"Required selected research evidence is missing: {name}")


def _validate_risk_grid(risk: pd.DataFrame) -> None:
    for method in METHODS:
        for confidence in CONFIDENCE_LEVELS:
            row = risk[
                risk.method.eq(method) & np.isclose(risk.confidence, confidence)
            ]
            if (
                len(row) != 1
                or not np.isfinite(row[RISK_VALUE_COLUMNS].to_numpy(float)).all()
            ):
                raise ValueError(
                    f"Missing, duplicate or nonfinite risk result: {method}/{confidence}"
                )


def _validate_cost_scenarios(data: dict) -> None:
    for name in ("portfolio_metrics", "funding_diagnostics"):
        frame = subset(data[name], SELECTED_STRATEGY)
        for cost in COST_SCENARIOS_BPS:
            if len(frame[np.isclose(frame.cost_bps, cost)]) != 1:
                raise ValueError(f"Missing or duplicate cost scenario: {name}/{cost}")


def _validate_weights(weights: pd.DataFrame) -> None:
    if set(weights.asset) != EXPECTED_ASSETS or len(weights) != len(EXPECTED_ASSETS):
        raise ValueError("The seven-asset selected allocation is incomplete.")
    if (
        not np.isfinite(weights.weight).all()
        or (weights.weight < 0).any()
        or not np.isclose(weights.weight.sum(), 1.0)
    ):
        raise ValueError("Selected allocation weights are invalid.")


def _validate_selected_procedure(summary: dict) -> None:
    if (
        summary["selected_research_portfolio"] != SELECTED_STRATEGY
        or summary["selected_research_monitor"] != SELECTED_MONITOR
    ):
        raise ValueError(
            "Saved selected procedure differs from the frozen research choice."
        )


def validate_tables(data: dict) -> None:
    """Reject an incomplete selected-procedure grid before presenting any estimates."""
    _validate_required_columns(data)
    _validate_risk_grid(subset(data["deployment_risk"], SELECTED_STRATEGY))
    _validate_cost_scenarios(data)
    _validate_weights(subset(data["deployment_weights"], SELECTED_STRATEGY))
    _validate_selected_procedure(data["summary"])


@st.cache_data(show_spinner=False)
def read_csv(path_string: str, digest: str, relative_name: str) -> pd.DataFrame:
    """Cache by verified run fingerprint, including the published release tables."""
    del digest
    return pd.read_csv(Path(path_string) / relative_name, low_memory=False)


def _load_daily_accounting(path: Path, digest: str, cost_bps: float) -> pd.DataFrame:
    daily = read_csv(
        str(path),
        digest,
        f"{SELECTED_STRATEGY}_cost_{cost_bps:g}bps_daily.csv",
    ).set_index("Date")
    daily.index = pd.to_datetime(daily.index)
    if (
        daily.empty
        or daily.index.has_duplicates
        or not daily.index.is_monotonic_increasing
        or not np.isfinite(daily.nav_usd).all()
    ):
        raise ValueError("The selected daily accounting output is invalid.")
    return daily


def _resolve_date_window(daily: pd.DataFrame, dates: tuple) -> DateWindow:
    if len(dates) != 2:
        return DateWindow(daily.iloc[:0], None, None, "Choose both backtest dates.")
    start, end = map(pd.Timestamp, dates)
    shown = daily.loc[start:end]
    error = (
        f"Choose at least {MIN_BACKTEST_SESSIONS} final-period trading sessions."
        if len(shown) < MIN_BACKTEST_SESSIONS
        else None
    )
    return DateWindow(shown, start, end, error)


@dataclass
class Context:
    path: Path
    data: dict
    bps: float
    c: float
    method: str
    daily: pd.DataFrame
    shown: pd.DataFrame
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    date_error: str | None
    m: pd.Series
    funding: pd.Series
    weights: pd.DataFrame
    risk: pd.DataFrame
    chosen: pd.Series
    evt: pd.Series
    strategy: str = SELECTED_STRATEGY

    @property
    def confidence(self) -> float:
        return self.c

    @property
    def metrics(self) -> pd.Series:
        return self.m

    @classmethod
    def create(
        cls,
        path: Path,
        data: dict,
        bps: float,
        confidence: float,
        method: str,
        dates: tuple,
    ) -> "Context":
        daily = _load_daily_accounting(
            path, data["summary"]["result_digest"], bps
        )
        window = _resolve_date_window(daily, dates)
        metrics_table = subset(data["portfolio_metrics"], SELECTED_STRATEGY)
        funding_table = subset(data["funding_diagnostics"], SELECTED_STRATEGY)
        risk_table = subset(data["deployment_risk"], SELECTED_STRATEGY, confidence)
        return cls(
            path=path,
            data=data,
            bps=bps,
            c=confidence,
            method=method,
            daily=daily,
            shown=window.shown,
            start=window.start,
            end=window.end,
            date_error=window.error,
            m=metrics_table[metrics_table.cost_bps.eq(bps)].iloc[0],
            funding=funding_table[funding_table.cost_bps.eq(bps)].iloc[0],
            weights=subset(data["deployment_weights"], SELECTED_STRATEGY),
            risk=risk_table,
            chosen=risk_table[risk_table.method.eq(method)].iloc[0],
            evt=risk_table[risk_table.method.eq(GUARDED_EVT_METHOD)].iloc[0],
        )

    def release(self, name: str) -> pd.DataFrame:
        return read_csv(
            str(self.path), self.data["summary"]["result_digest"], f"{name}.csv"
        )
