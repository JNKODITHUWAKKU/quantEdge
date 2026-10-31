"""Recovery of the fixed final retrospective assessment; no model reselection."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.accounting import run_portfolio
from src.common import utc_now
from src.conditional import METHODS, forecast_portfolio
from src.final_assessment import assess_monitor, committee_assessment
from src.final_snapshot import deployment_snapshot
from src.forecasting import attach_outcomes, forecast_cross_section, origin_book
from src.funding import real_funding
from src.inputs import load_cpi, load_final, research_identity, settings
from src.metrics import performance_metrics
from src.risk import empirical_tail
from src.risk_validation import evaluate_forecasts
from src.selection import causal_states, es_calibration
from src.settings import (
    CONDITIONAL_SETTINGS,
    PORTFOLIO_SETTINGS,
    RISK_SETTINGS,
    VALIDATION_SETTINGS,
)
from src.storage import csv, write_assessment, write_summary
from src.stress import WINDOWS

EVIDENCE_CLASS = "locked_final_retrospective_recovery"
SELECTED_STRATEGY = "sample_gmv"
SELECTED_MONITOR = "garch_t"
BENCHMARK_STRATEGY = "policy_40_60"
REBALANCE_FREQUENCY = "quarterly"
BASE_RISK_METHODS = ["historical", "gaussian", "monte_carlo"]
EMPIRICAL_ES_CONFIDENCE = 0.95
BASEL_CONFIDENCE = 0.99
BASEL_WINDOW = 250
BASEL_GREEN_MAX = 4
BASEL_AMBER_MAX = 9
PROGRESS_LOG_INTERVAL = 50
REFERENCE_CAPITAL_USD = 1e7
PRIMARY_CRISIS_COST_BPS = 5.0
EQUIVALENCE_RTOL = 1e-10
EQUIVALENCE_ATOL = 1e-12
DOLLAR_FIELDS = [
    "nav_start_reported",
    "invested_start",
    "market_pnl",
    "trading_cost_dollars",
    "nav_usd",
    "accounting_residual",
]


@dataclass(frozen=True)
class PipelineContext:
    root: Path
    output: Path
    cfg: dict
    lock: dict
    selection_cfg: dict
    baseline_cfg: dict
    base_risk_cfg: dict
    conditional_cfg: dict
    prices: pd.DataFrame
    returns: pd.DataFrame
    rate: pd.Series
    audit: dict
    assets: list[str]
    strategies: list[str]
    confidence_levels: list[float]
    lookback: int
    expected_dates: pd.DatetimeIndex
    first_origin: pd.Timestamp


@dataclass(frozen=True)
class BookExposureCheck:
    strategy: str
    weights: np.ndarray


@dataclass(frozen=True)
class StrategyCostRun:
    daily: pd.DataFrame
    book: pd.DataFrame
    metrics: dict


@dataclass(frozen=True)
class AccountingOutputs:
    dailies: dict[str, pd.DataFrame]
    books: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    first_book: pd.DataFrame


@dataclass(frozen=True)
class ForecastLedger:
    issued: pd.DataFrame
    issued_at: str
    last_origin: pd.Timestamp
    last_training: pd.DataFrame
    last_exposures: dict[str, dict]


@dataclass(frozen=True)
class AssessmentOutputs:
    evaluated: pd.DataFrame
    diagnostics: pd.DataFrame
    monitor: dict


def build_final(root: Path, output: Path) -> Path:
    started = time.perf_counter()
    cfg = settings(root)
    lock = research_identity(root)
    output.mkdir(parents=True, exist_ok=True)
    phase = "portfolio_accounting"
    origin: pd.Timestamp | None = None
    opened_at = utc_now()

    def track_origin(current_origin: pd.Timestamp) -> None:
        nonlocal origin
        origin = current_origin

    try:
        ctx = _create_context(root, output, cfg, lock)
        accounting = _run_portfolio_accounting(ctx)
        phase = "past_only_forecasts"
        ledger = _issue_past_only_forecasts(ctx, accounting, track_origin)
        phase = "outcome_join_and_assessment"
        assessment = _evaluate_and_assess_outcomes(ctx, accounting, ledger)
        phase = "funding_and_cutoff_snapshot"
        committee = _run_funding_and_snapshot(ctx, accounting, ledger, assessment)
        summary = _build_run_summary(
            ctx=ctx,
            accounting=accounting,
            ledger=ledger,
            assessment=assessment,
            committee=committee,
            opened_at=opened_at,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_summary(output, summary)
        print(
            f"Completed {output.name}: {len(ledger.issued)} final forecasts; no reselection.",
            flush=True,
        )
        return output
    except Exception:
        logging.getLogger(__name__).exception(
            "Research failed during %s at origin %s", phase, origin
        )
        raise


def _create_context(root: Path, output: Path, cfg: dict, lock: dict) -> PipelineContext:
    prices, rate, audit = load_final(root, cfg)
    csv(audit["issues"], output / "source_issues.csv")
    csv(audit["inventory"], output / "source_inventory.csv")
    selection_cfg = deepcopy(VALIDATION_SETTINGS)
    baseline_cfg = deepcopy(PORTFOLIO_SETTINGS)
    base_risk_cfg = deepcopy(RISK_SETTINGS)
    conditional_cfg = deepcopy(CONDITIONAL_SETTINGS)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    assets = list(returns.columns)
    if assets != conditional_cfg["stress"]["assets"]:
        raise ValueError("Shock vector order differs from the asset panel.")
    expected_dates = prices.index[
        (prices.index >= cfg["run"]["start"]) & (prices.index <= cfg["run"]["end"])
    ]
    first_origin = prices.index[prices.index.get_loc(expected_dates[0]) - 1]
    return PipelineContext(
        root=root,
        output=output,
        cfg=cfg,
        lock=lock,
        selection_cfg=selection_cfg,
        baseline_cfg=baseline_cfg,
        base_risk_cfg=base_risk_cfg,
        conditional_cfg=conditional_cfg,
        prices=prices,
        returns=returns,
        rate=rate,
        audit=audit,
        assets=assets,
        strategies=cfg["run"]["strategies"],
        confidence_levels=cfg["run"]["confidence_levels"],
        lookback=cfg["run"]["lookback_returns"],
        expected_dates=expected_dates,
        first_origin=first_origin,
    )


def _run_portfolio_accounting(ctx: PipelineContext) -> AccountingOutputs:
    validation_prices = ctx.prices.loc[: ctx.selection_cfg["run"]["end"]]
    weight_cols = [f"weight_{asset}" for asset in ctx.assets]
    dailies: dict[str, pd.DataFrame] = {}
    books: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict] = []
    book_checks: list[BookExposureCheck] = []
    latest_weights: list[dict] = []

    for cost in ctx.cfg["run"]["cost_scenarios_bps"]:
        results = {
            strategy: run_portfolio(
                ctx.prices,
                strategy,
                ctx.lookback,
                REBALANCE_FREQUENCY,
                cost,
                ctx.cfg["run"]["reference_capital_usd"],
            )
            for strategy in ctx.strategies
        }
        benchmark = results[BENCHMARK_STRATEGY].daily.net_return.loc[ctx.expected_dates]
        for strategy in ctx.strategies:
            cost_run = _evaluate_strategy_at_cost(
                ctx=ctx,
                strategy=strategy,
                cost=cost,
                result=results[strategy],
                validation_prices=validation_prices,
                benchmark=benchmark,
            )
            metric_rows.append(cost_run.metrics)
            if cost == ctx.cfg["run"]["primary_cost_bps"]:
                dailies[strategy] = cost_run.daily
                books[strategy] = cost_run.book
                latest_weights.extend(_extract_latest_weights(strategy, cost_run.daily, ctx.assets))
            else:
                # The same execution rule should make weights independent of proportional cost.
                if cost == ctx.cfg["run"]["cost_scenarios_bps"][0]:
                    book_checks.append(
                        BookExposureCheck(strategy, cost_run.book[weight_cols].to_numpy())
                    )
                elif strategy in books and not np.allclose(
                    cost_run.book[weight_cols],
                    books[strategy][weight_cols],
                    atol=EQUIVALENCE_ATOL,
                    rtol=0,
                ):
                    raise ValueError("Cost scenarios changed risk exposures.")

    _verify_exposure_invariance(book_checks, books, weight_cols)
    metrics_df = pd.DataFrame(metric_rows)
    csv(metrics_df, ctx.output / "portfolio_metrics.csv")
    csv(pd.DataFrame(latest_weights), ctx.output / "latest_held_weights.csv")

    first_book = books[ctx.strategies[0]]
    _verify_aligned_forecast_dates(books, first_book)
    states = causal_states(ctx.returns, first_book, ctx.selection_cfg)
    csv(states, ctx.output / "volatility_states.csv")
    return AccountingOutputs(
        dailies=dailies,
        books=books,
        metrics=metrics_df,
        first_book=first_book,
    )


def _evaluate_strategy_at_cost(
    ctx: PipelineContext,
    strategy: str,
    cost: float,
    result: object,
    validation_prices: pd.DataFrame,
    benchmark: pd.Series,
) -> StrategyCostRun:
    full = result.daily
    prior = run_portfolio(
        validation_prices,
        strategy,
        ctx.lookback,
        REBALANCE_FREQUENCY,
        cost,
        ctx.cfg["run"]["reference_capital_usd"],
    ).daily
    matched = full.loc[prior.index]
    fields = ["net_return", "gross_return"] + [f"end_weight_{asset}" for asset in ctx.assets]
    if not np.allclose(matched[fields], prior[fields], rtol=EQUIVALENCE_RTOL, atol=EQUIVALENCE_ATOL):
        raise ValueError("Extending the history altered prior validation returns/holdings.")

    book = origin_book(full, result.signals, result.trades, ctx.prices.index, ctx.assets)
    book = book[pd.to_datetime(book.realization_date).isin(ctx.expected_dates)].reset_index(
        drop=True
    )
    if not pd.DatetimeIndex(book.realization_date).equals(ctx.expected_dates):
        raise ValueError("Validation book misses eligible sessions.")

    reference_capital = ctx.cfg["run"]["reference_capital_usd"]
    # Rebase the full-history backtest NAV so the final evaluation window starts at USD 10m on 2019-12-31.
    scale = reference_capital / float(full.loc[ctx.first_origin, "nav_usd"])
    daily = full.loc[ctx.expected_dates].copy()
    daily[DOLLAR_FIELDS] *= scale
    book["origin_value_usd"] *= scale
    daily["drawdown"] = (
        daily.nav_usd / np.maximum.accumulate(np.r_[reference_capital, daily.nav_usd])[1:] - 1
    )
    daily["risk_free_return"] = ctx.rate.reindex(daily.index)
    daily["benchmark_return"] = benchmark
    daily["market_return"] = ctx.returns.SPY.reindex(daily.index)

    prefix = f"{strategy}_cost_{cost:g}bps"
    csv(daily.reset_index(), ctx.output / f"{prefix}_daily.csv")
    signals = result.signals[
        pd.to_datetime(result.signals.execution_date).isin(ctx.expected_dates)
    ].copy()
    trades = result.trades[pd.to_datetime(result.trades.date).isin(ctx.expected_dates)].copy()
    trades[["cost_dollars", "gross_traded_dollars"]] *= scale
    csv(signals, ctx.output / f"{prefix}_signals.csv")
    csv(trades, ctx.output / f"{prefix}_trades.csv")

    metrics = performance_metrics(
        daily.net_return,
        daily.risk_free_return,
        benchmark,
        daily.market_return,
        ctx.first_origin,
        ctx.cfg["run"]["annualization"],
        ctx.baseline_cfg["run"]["hac_lags"],
    )
    metrics.update(
        strategy=strategy,
        cost_bps=cost,
        empirical_es95=empirical_tail(-daily.net_return.to_numpy(), EMPIRICAL_ES_CONFIDENCE)["es"],
        observations=len(daily),
        start_valuation_date=str(ctx.first_origin.date()),
        first_return_date=str(daily.index[0].date()),
        end_date=str(daily.index[-1].date()),
        total_trading_cost_dollars=float(daily.trading_cost_dollars.sum()),
        terminal_wealth_usd=float(daily.nav_usd.iloc[-1]),
        max_held_weight=float(daily.filter(like="end_weight_").to_numpy().max()),
        evidence_class=EVIDENCE_CLASS,
    )
    return StrategyCostRun(daily=daily, book=book, metrics=metrics)


def _extract_latest_weights(strategy: str, daily: pd.DataFrame, assets: list[str]) -> list[dict]:
    last_date = daily.index[-1]
    last_row = daily.iloc[-1]
    return [
        {
            "strategy": strategy,
            "date": last_date,
            "asset": asset,
            "weight": last_row[f"end_weight_{asset}"],
        }
        for asset in assets
    ]


def _verify_exposure_invariance(
    book_checks: list[BookExposureCheck],
    books: dict[str, pd.DataFrame],
    weight_cols: list[str],
) -> None:
    for check in book_checks:
        if not np.allclose(
            check.weights,
            books[check.strategy][weight_cols],
            atol=EQUIVALENCE_ATOL,
            rtol=0,
        ):
            raise ValueError("Cost-zero exposure differs from primary holdings.")


def _verify_aligned_forecast_dates(
    books: dict[str, pd.DataFrame], first_book: pd.DataFrame
) -> None:
    date_cols = ["forecast_origin", "realization_date"]
    reference_dates = first_book[date_cols]
    if any(not book[date_cols].equals(reference_dates) for book in books.values()):
        raise ValueError("Portfolio forecast dates differ.")


def _diagnostic_origin_dates(first_book: pd.DataFrame) -> set[pd.Timestamp]:
    origin_index = pd.DatetimeIndex(first_book.forecast_origin)
    origin_series = origin_index.to_series()
    return set(origin_series.groupby(origin_index.year).max())


def _build_origin_exposures(
    books: dict[str, pd.DataFrame],
    strategies: list[str],
    weight_cols: list[str],
    row_index: int,
    realization_date: pd.Timestamp,
) -> dict[str, dict]:
    return {
        strategy: {
            "weights": books[strategy].iloc[row_index][weight_cols].to_numpy(float),
            "origin_value_usd": float(books[strategy].iloc[row_index].origin_value_usd),
            "realization_date": realization_date,
        }
        for strategy in strategies
    }


def _issue_past_only_forecasts(
    ctx: PipelineContext,
    accounting: AccountingOutputs,
    on_origin: Callable[[pd.Timestamp], None],
) -> ForecastLedger:
    all_forecasts: list[dict] = []
    all_components: list[dict] = []
    grids: list[dict] = []
    filter_rows: list[dict] = []
    extremograms: list[dict] = []
    weight_cols = [f"weight_{asset}" for asset in ctx.assets]
    diagnostic_dates = _diagnostic_origin_dates(accounting.first_book)
    total_origins = len(accounting.first_book)
    origin = pd.Timestamp(accounting.first_book.iloc[0].forecast_origin)
    training = pd.DataFrame()
    exposures: dict[str, dict] = {}

    for row_index, entry in accounting.first_book.iterrows():
        origin = pd.Timestamp(entry.forecast_origin)
        on_origin(origin)
        training = ctx.returns.loc[:origin].tail(ctx.lookback)
        realization_date = pd.Timestamp(entry.realization_date)
        exposures = _build_origin_exposures(
            accounting.books, ctx.strategies, weight_cols, row_index, realization_date
        )
        base_forecasts, base_parts = forecast_cross_section(
            ctx.returns,
            origin,
            exposures,
            ctx.lookback,
            ctx.confidence_levels,
            ctx.base_risk_cfg["monte_carlo"],
            ctx.base_risk_cfg["run"]["seed"],
        )
        all_forecasts.extend(base_forecasts.to_dict("records"))
        all_components.extend(base_parts.to_dict("records"))
        for strategy in ctx.strategies:
            cond_forecasts, cond_parts, tail_grid, filters, extreme, _ = forecast_portfolio(
                training,
                origin,
                exposures[strategy],
                strategy,
                ctx.conditional_cfg,
                diagnostics=origin in diagnostic_dates,
            )
            all_forecasts.extend(cond_forecasts)
            grids.extend(tail_grid)
            filter_rows.extend(filters)
            extremograms.extend(extreme)
            origin_value = exposures[strategy]["origin_value_usd"]
            for component_row in cond_parts:
                component_row["es_component_usd"] = component_row["es_component"] * origin_value
            all_components.extend(cond_parts)
        if row_index % PROGRESS_LOG_INTERVAL == 0 or row_index == total_origins - 1:
            print(
                f"Recovery forecasts {row_index + 1}/{total_origins}: {origin.date()}; no reselection",
                flush=True,
            )

    issued = pd.DataFrame(all_forecasts)
    issued["evidence_class"] = EVIDENCE_CLASS
    issued["risk_run_id"] = ctx.lock["lock_id"]
    issued["source_snapshot"] = ctx.baseline_cfg["run"]["snapshot_id"]
    issued["final_lock_id"] = ctx.lock["original_final_lock_id"]
    issued["selection_lock_id"] = ctx.lock["identity"]["selection_lock_id"]
    expected_rows = (
        total_origins
        * len(ctx.strategies)
        * (len(BASE_RISK_METHODS) + len(METHODS))
        * len(ctx.confidence_levels)
    )
    if len(issued) != expected_rows:
        raise RuntimeError("Forecast grid is incomplete.")

    # Persist all forecast values before adding outcomes; this is reconstructed historical forecasting.
    csv(issued, ctx.output / "forecasts_issued.csv")
    issued_at = utc_now()
    csv(pd.DataFrame(all_components), ctx.output / "es_contributions.csv")
    csv(pd.DataFrame(grids), ctx.output / "threshold_diagnostics.csv")
    csv(pd.DataFrame(filter_rows), ctx.output / "filter_diagnostics.csv")
    csv(pd.DataFrame(extremograms), ctx.output / "extremograms.csv")
    return ForecastLedger(
        issued=issued,
        issued_at=issued_at,
        last_origin=origin,
        last_training=training,
        last_exposures=exposures,
    )


def _evaluate_and_assess_outcomes(
    ctx: PipelineContext,
    accounting: AccountingOutputs,
    ledger: ForecastLedger,
) -> AssessmentOutputs:
    evaluated = attach_outcomes(ledger.issued, ctx.returns, accounting.dailies, ctx.assets)
    csv(evaluated, ctx.output / "forecasts_evaluated.csv")
    diagnostics = _export_backtest_diagnostics(evaluated, ctx.output)
    monitor, tables = assess_monitor(evaluated, diagnostics, ctx.cfg)
    for title, frame in tables.items():
        csv(frame, ctx.output / f"{title}.csv")
    es_cfg = {
        **ctx.selection_cfg,
        "run": ctx.cfg["run"],
        "calibration_diagnostics": ctx.cfg["calibration_diagnostics"],
    }
    csv(es_calibration(evaluated, es_cfg), ctx.output / "es_calibration.csv")
    _export_annual_backtests(evaluated, ctx.output)
    _export_basel_rolling_history(evaluated, ctx.output)
    shocks = np.asarray(ctx.conditional_cfg["stress"]["hypothetical_returns"])
    _export_hypothetical_sensitivity(accounting.books, ctx.assets, shocks, ctx.output)
    _export_stationarity_diagnostics(
        training=ledger.last_training,
        exposures=ledger.last_exposures,
        assets=ctx.assets,
        strategies=ctx.strategies,
        origin=ledger.last_origin,
        output=ctx.output,
    )
    csv(
        ledger.issued[ledger.issued.forecast_origin.eq(ledger.last_origin)],
        ctx.output / "last_evaluated_origin_risk.csv",
    )
    return AssessmentOutputs(evaluated=evaluated, diagnostics=diagnostics, monitor=monitor)


def _export_backtest_diagnostics(evaluated: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for (strategy, method, _), group in evaluated.groupby(
        ["strategy", "method", "confidence"], sort=False
    ):
        is_fallback = group.fit_status.eq("FALLBACK")
        rows.append(
            {
                "strategy": strategy,
                "method": method,
                **evaluate_forecasts(group),
                "fallback_count": int(is_fallback.sum()),
                "fallback_fraction": float(is_fallback.mean()),
                "evidence_class": EVIDENCE_CLASS,
                "interpretation": (
                    "Locked retrospective recovery; no reselection; final results previously seen."
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    csv(diagnostics, output / "backtest_diagnostics.csv")
    return diagnostics


def _export_annual_backtests(evaluated: pd.DataFrame, output: Path) -> None:
    yearly = [
        {
            "strategy": strategy,
            "method": method,
            "confidence": confidence_level,
            "calendar_year": int(year),
            **evaluate_forecasts(group),
            "scope": "descriptive; no adoption",
        }
        for (strategy, method, confidence_level, year), group in evaluated.groupby(
            ["strategy", "method", "confidence", evaluated.realization_date.dt.year]
        )
    ]
    csv(pd.DataFrame(yearly), output / "annual_backtests.csv")


def _export_basel_rolling_history(evaluated: pd.DataFrame, output: Path) -> None:
    selected99 = evaluated[
        evaluated.strategy.eq(SELECTED_STRATEGY)
        & evaluated.method.eq(SELECTED_MONITOR)
        & np.isclose(evaluated.confidence, BASEL_CONFIDENCE)
    ].sort_values("realization_date")
    counts = selected99.exception.astype(int).rolling(BASEL_WINDOW).sum()
    rolling = pd.DataFrame(
        {
            "realization_date": selected99.realization_date,
            "exceptions_last_250": counts,
        }
    )
    rolling["zone"] = np.where(
        counts.isna(),
        "INSUFFICIENT",
        np.where(
            counts <= BASEL_GREEN_MAX,
            "GREEN",
            np.where(counts <= BASEL_AMBER_MAX, "AMBER", "RED"),
        ),
    )
    csv(rolling, output / "selected_monitor_basel_history.csv")


def _export_hypothetical_sensitivity(
    books: dict[str, pd.DataFrame],
    assets: list[str],
    shocks: np.ndarray,
    output: Path,
) -> None:
    weight_cols = [f"weight_{asset}" for asset in assets]
    scenario_rows = []
    for strategy, book in books.items():
        losses = -book[weight_cols].to_numpy() @ shocks
        scenario_rows.extend(
            {
                "strategy": strategy,
                "forecast_origin": forecast_origin,
                "hypothetical_loss_fraction": float(loss_fraction),
                "reference_loss_usd": float(loss_fraction * REFERENCE_CAPITAL_USD),
                "probability": "NOT_ASSIGNED",
            }
            for forecast_origin, loss_fraction in zip(book.forecast_origin, losses)
        )
    csv(pd.DataFrame(scenario_rows), output / "hypothetical_exposure_sensitivity.csv")


def _export_stationarity_diagnostics(
    training: pd.DataFrame,
    exposures: dict[str, dict],
    assets: list[str],
    strategies: list[str],
    origin: pd.Timestamp,
    output: Path,
) -> None:
    model_series = {asset: training[asset].to_numpy() for asset in assets}
    training_matrix = training.to_numpy()
    model_series.update(
        {strategy: training_matrix @ exposures[strategy]["weights"] for strategy in strategies}
    )
    stationarity_rows = []
    for series_name, values in model_series.items():
        for convention, transformed in [("simple", values), ("log", np.log1p(values))]:
            adf_result = adfuller(transformed, regression="c", autolag="AIC")
            stationarity_rows.append(
                {
                    "series": series_name,
                    "return_convention": convention,
                    "origin": origin,
                    "training_start": training.index[0],
                    "adf_statistic": adf_result[0],
                    "pvalue": adf_result[1],
                    "selected_lags": adf_result[2],
                    "observations": adf_result[3],
                    "null": "unit root",
                    "interpretation": "Rejection does not establish iid or risk-model adequacy.",
                }
            )
    csv(pd.DataFrame(stationarity_rows), output / "stationarity_diagnostics.csv")


def _run_funding_and_snapshot(
    ctx: PipelineContext,
    accounting: AccountingOutputs,
    ledger: ForecastLedger,
    assessment: AssessmentOutputs,
) -> dict:
    cpi = load_cpi(ctx.root, ctx.cfg)
    funding_rows = []
    real_paths = []
    for cost in ctx.cfg["run"]["cost_scenarios_bps"]:
        for strategy in ctx.strategies:
            daily = pd.read_csv(
                ctx.output / f"{strategy}_cost_{cost:g}bps_daily.csv",
                index_col="Date",
                parse_dates=True,
            )
            result, path = real_funding(
                daily,
                cpi,
                REFERENCE_CAPITAL_USD,
                ctx.cfg["funding"]["base_month"],
                ctx.cfg["funding"]["last_month"],
                cost,
            )
            funding_rows.append({"strategy": strategy, **result})
            real_paths.append(path.assign(strategy=strategy, cost_bps=cost))
    funding_df = pd.DataFrame(funding_rows)
    csv(funding_df, ctx.output / "funding_diagnostics.csv")
    csv(pd.concat(real_paths, ignore_index=True), ctx.output / "real_wealth_paths.csv")
    deployment_snapshot(
        ctx.prices,
        ctx.returns,
        ctx.cfg,
        ctx.base_risk_cfg,
        ctx.conditional_cfg,
        ctx.output,
    )
    _export_causal_strategy_crises(accounting.dailies, ctx.output)
    committee = committee_assessment(assessment.monitor, funding_df, ctx.cfg)
    write_assessment(ctx.output, assessment.monitor, committee)
    # Preserve the five observed MC flags; never change seeds to manufacture agreement.
    csv(
        assessment.evaluated[
            assessment.evaluated.method.eq("monte_carlo")
            & assessment.evaluated.mc_agreement_with_analytic.eq(False)
        ],
        ctx.output / "monte_carlo_agreement_flags.csv",
    )
    return committee


def _export_causal_strategy_crises(dailies: dict[str, pd.DataFrame], output: Path) -> None:
    crisis_rows = []
    for definition in WINDOWS:
        for strategy, daily in dailies.items():
            section = daily.loc[definition["start"] : definition["end"]]
            if section.empty:
                crisis_rows.append(
                    {
                        "strategy": strategy,
                        "scenario": definition["scenario"],
                        "status": "OUTSIDE_FINAL_STRATEGY_LEDGER",
                    }
                )
                continue
            growth_path = (1 + section.net_return).cumprod().to_numpy()
            anchored_growth = np.r_[1.0, growth_path]
            drawdowns = 1 - anchored_growth / np.maximum.accumulate(anchored_growth)
            crisis_rows.append(
                {
                    "strategy": strategy,
                    "scenario": definition["scenario"],
                    "status": "CAUSAL_REBALANCED_STRATEGY",
                    "first_return_date": section.index[0],
                    "last_return_date": section.index[-1],
                    "terminal_loss_fraction": float(1 - growth_path[-1]),
                    "maximum_drawdown_loss": float(drawdowns.max()),
                    "cost_bps": PRIMARY_CRISIS_COST_BPS,
                }
            )
    csv(pd.DataFrame(crisis_rows), output / "causal_strategy_crises.csv")


def _build_run_summary(
    ctx: PipelineContext,
    accounting: AccountingOutputs,
    ledger: ForecastLedger,
    assessment: AssessmentOutputs,
    committee: dict,
    opened_at: str,
    elapsed_seconds: float,
) -> dict:
    mc = ledger.issued[ledger.issued.method.eq("monte_carlo")]
    return {
        "run_id": ctx.lock["lock_id"],
        "created_at_utc": utc_now(),
        "final_lock_id": ctx.lock["original_final_lock_id"],
        "recovery_contract_recorded_at_utc": ctx.lock["source_recovery_contract_at_utc"],
        "status": "FINAL_RECOVERY_COMPLETE",
        "source_snapshot": ctx.baseline_cfg["run"]["snapshot_id"],
        "opened_at_utc": opened_at,
        "evidence_class": ctx.cfg["run"]["evidence_class"],
        "human_review": "PENDING",
        "human_investment_approval": "PENDING",
        "strategies": ctx.strategies,
        "methods": BASE_RISK_METHODS + METHODS,
        "confidence_levels": ctx.confidence_levels,
        "cost_scenarios_bps": ctx.cfg["run"]["cost_scenarios_bps"],
        "forecast_rows": len(ledger.issued),
        "observations_per_group": len(accounting.first_book),
        "first_origin": str(accounting.first_book.iloc[0].forecast_origin.date()),
        "last_origin": str(ledger.last_origin.date()),
        "first_realization": str(ctx.expected_dates[0].date()),
        "last_realization": str(ctx.expected_dates[-1].date()),
        "forecast_written_before_outcome_join_at_utc": ledger.issued_at,
        "fallback_rows": int(ledger.issued.fit_status.eq("FALLBACK").sum()),
        "mc_precision_limit_rows": int(mc.mc_precision_reached.eq(False).sum()),
        "mc_agreement_flag_rows": int(mc.mc_agreement_with_analytic.eq(False).sum()),
        "source_warning_count": ctx.audit["warning_count"],
        "validation_prefix_equivalence": True,
        "selected_research_portfolio": SELECTED_STRATEGY,
        "selected_research_monitor": SELECTED_MONITOR,
        "monitor_assessment": assessment.monitor["status"],
        "committee_research_assessment": committee["research_assessment"],
        "final_period_evaluated": True,
        "selection_changes": [],
        "optional_ml": "DEFERRED_UNTESTED",
        "deployment_valuation_date": str(ctx.prices.index[-1].date()),
        "real_wealth_endpoint": "2026-09",
        "elapsed_seconds": elapsed_seconds,
        "delivery": "ZIP_ONLY_NO_REMOTE_PUSH",
        "recovery_note": (
            "Final outputs were observed before this recovery; original pre-opening scratch "
            "lock/commit bytes were lost. The disclosed recovery history is documented in "
            "README.md and the dashboard."
        ),
    }
