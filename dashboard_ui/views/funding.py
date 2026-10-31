"""Saved research presentation; numerical models remain in src."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
from dashboard_ui.data import subset
from dashboard_ui.charts import annual_returns, lines
from dashboard_ui.components import (
    MetricItem,
    metric_strip,
    remembered_selectbox,
    section,
    table,
    tone,
)
from dashboard_ui.disclosures import (
    INCEPTION_DATE,
    INITIAL_CAPITAL_USD,
    LOSS_TOLERANCES,
    SPENDING_RATES,
)

DEFAULT_SPENDING_INDEX = 2
DEFAULT_TOLERANCE_INDEX = 1
PERCENT_SCALE = 100.0

USD_KPIS = frozenset({"total_trading_cost_dollars", "terminal_wealth_usd"})
PERCENT_KPIS = frozenset(
    {
        "cagr",
        "annualized_volatility",
        "jensen_alpha_annual_arithmetic",
        "alpha_ci95_low_annual",
        "alpha_ci95_high_annual",
        "tracking_error_annual",
        "maximum_drawdown_loss",
    }
)

KPI_KEYS = (
    "cagr",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "beta",
    "jensen_alpha_annual_arithmetic",
    "alpha_ci95_low_annual",
    "alpha_ci95_high_annual",
    "alpha_hac_pvalue",
    "treynor",
    "information_ratio",
    "tracking_error_annual",
    "maximum_drawdown_loss",
    "calmar",
    "total_trading_cost_dollars",
    "terminal_wealth_usd",
)

BASELINE_COLUMN_LABELS = {
    "portfolio": "Portfolio",
    "research_role": "Research role",
    "cagr": "Net CAGR (%)",
    "annualized_volatility": "Annual volatility (%)",
    "maximum_drawdown_loss": "Maximum drawdown (%)",
    "sharpe": "Sharpe",
    "real_total_return": "Real change (%)",
    "gfc_extended_replay_drawdown": "GFC replay drawdown (%)",
}


@dataclass(frozen=True)
class DrawdownEpisode:
    """Dates and recovery requirement for the worst full-period drawdown."""

    peak_date: str
    trough_date: str
    recovery_label: str
    gain_needed_after_trough: float


def _format_kpi(name: str, value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if name in USD_KPIS:
        return f"USD {value:,.0f}"
    if name in PERCENT_KPIS:
        return f"{value:.2%}"
    return f"{value:.4f}"


def _compute_spending_screens(
    ctx, wealth_path: pd.DataFrame, spending_rate: float, loss_tolerance: float
) -> pd.DataFrame:
    annual_budget_usd = spending_rate * INITIAL_CAPITAL_USD
    nominal_after_spending = (
        wealth_path.nominal_no_spending_usd
        - annual_budget_usd * wealth_path.withdrawal_terminal_coefficient
    )
    real_terminal_usd = float(
        nominal_after_spending.iloc[-1] / wealth_path.cpi_ratio.iloc[-1]
    )
    max_replay_drawdown = subset(
        ctx.data["historical_stress"], ctx.strategy
    ).maximum_drawdown_loss.max()
    max_hypothetical_loss = subset(
        ctx.data["hypothetical_stress"], ctx.strategy
    ).loss_fraction.max()
    return pd.DataFrame(
        [
            {
                "Screen": "Real principal after spending",
                "Value": real_terminal_usd,
                "Pass": real_terminal_usd >= INITIAL_CAPITAL_USD,
            },
            {
                "Screen": "Observed drawdown",
                "Value": ctx.metrics.maximum_drawdown_loss,
                "Pass": ctx.metrics.maximum_drawdown_loss <= loss_tolerance,
            },
            {
                "Screen": "Largest replay drawdown",
                "Value": max_replay_drawdown,
                "Pass": max_replay_drawdown <= loss_tolerance,
            },
            {
                "Screen": "Largest stipulated terminal shock",
                "Value": max_hypothetical_loss,
                "Pass": max_hypothetical_loss <= loss_tolerance,
            },
        ]
    )


def _compute_drawdown_episode(daily: pd.DataFrame) -> DrawdownEpisode:
    wealth_values = np.r_[INITIAL_CAPITAL_USD, daily.nav_usd.to_numpy()]
    drawdowns = wealth_values / np.maximum.accumulate(wealth_values) - 1
    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(wealth_values[: trough_idx + 1]))
    recovery_indices = np.flatnonzero(
        wealth_values[trough_idx + 1 :] >= wealth_values[peak_idx]
    )
    dates = pd.DatetimeIndex([pd.Timestamp(INCEPTION_DATE), *daily.index])
    recovery_label = (
        str(dates[trough_idx + 1 + int(recovery_indices[0])].date())
        if len(recovery_indices)
        else "Not recovered by cutoff"
    )
    gain_needed = 1.0 / (1.0 + drawdowns[trough_idx]) - 1.0
    return DrawdownEpisode(
        peak_date=str(dates[peak_idx].date()),
        trough_date=str(dates[trough_idx].date()),
        recovery_label=recovery_label,
        gain_needed_after_trough=gain_needed,
    )


def _render_purchasing_power_section(ctx) -> pd.DataFrame:
    metrics = ctx.metrics
    funding = ctx.funding
    metric_strip(
        [
            MetricItem("Sharpe", f"{metrics.sharpe:.3f}", tone(metrics.sharpe)),
            MetricItem("Sortino", f"{metrics.sortino:.3f}", tone(metrics.sortino)),
            MetricItem("Calmar", f"{metrics.calmar:.3f}", tone(metrics.calmar)),
            MetricItem("CAPM beta / SPY", f"{metrics.beta:.3f}", "neutral"),
        ]
    )
    section("Purchasing power and path-dependent spending capacity")
    st.write(
        f"USD10m becomes **USD {funding.terminal_real_no_spending_usd:,.0f} in December2019 purchasing power** at {funding.last_valuation_date}, before spending."
    )
    if pd.isna(funding.feasible_nonnegative_annual_real_budget_usd):
        st.warning(
            "No nonnegative constant annual real withdrawal preserves initial real principal along this observed path."
        )
    else:
        st.write(
            f"Retrospective annual real budget capacity: USD {funding.feasible_nonnegative_annual_real_budget_usd:,.0f}, over {int(funding.completed_annual_payments)} completed December payments."
        )
    st.caption(
        "Path capacity is not a forecast or recommended payout. Pro-rata sales include costs; reinvested dividends are not added again."
    )
    wealth_path = subset(ctx.data["real_wealth_paths"], ctx.strategy)
    wealth_path = wealth_path[wealth_path.cost_bps.eq(ctx.bps)]
    lines(
        wealth_path,
        "valuation_date",
        ["nominal_no_spending_usd", "real_no_spending_base_usd"],
    )
    st.caption(
        "October2025 CPI is missing; the real line has a gap. US CPI is a provisional basket, not the university's verified liabilities."
    )
    table(ctx.data["portfolio_metrics"], hide_index=True, width="stretch")
    table(ctx.data["funding_diagnostics"], hide_index=True, width="stretch")
    return wealth_path


def _render_spending_screens_section(ctx, wealth_path: pd.DataFrame) -> None:
    section("Illustrative spending and loss-tolerance screens")
    spending_rate = remembered_selectbox(
        "Annual real spending (% initial USD10m)",
        list(SPENDING_RATES),
        index=DEFAULT_SPENDING_INDEX,
        format_func=lambda x: f"{x:.0%}",
        key="gmv_spend",
    )
    loss_tolerance = remembered_selectbox(
        "Illustrative loss tolerance",
        list(LOSS_TOLERANCES),
        index=DEFAULT_TOLERANCE_INDEX,
        format_func=lambda x: f"{x:.0%}",
        key="gmv_loss",
    )
    screens = _compute_spending_screens(
        ctx, wealth_path, spending_rate, loss_tolerance
    )
    table(screens, hide_index=True)
    st.caption(
        "Six completed December real payments, with pro-rata sales and costs. First value is base USD, others are loss fractions. Scenarios are post-evaluation managerial illustrations, not actual client terms or safe payout estimates. Passing a numeric screen does not override failed calibration or authorize investment."
    )
    st.warning(
        "HOLD / NO TRADE remains. Reconsider only after human-owned liabilities, payout, liquidity and loss limits; independent source/model review; and a newly declared protocol with genuinely new validation evidence."
    )


def _render_matched_baselines_section(ctx) -> None:
    section("Matched original research baselines")
    baselines = ctx.release("matched_baselines")
    baselines = baselines[np.isclose(baselines.cost_bps, ctx.bps)].copy()
    display_frame = baselines[list(BASELINE_COLUMN_LABELS)].rename(
        columns=BASELINE_COLUMN_LABELS
    )
    for col in [v for v in BASELINE_COLUMN_LABELS.values() if "(%)" in v]:
        display_frame[col] *= PERCENT_SCALE
    table(
        display_frame,
        hide_index=True,
        width="stretch",
        column_config={
            col: st.column_config.NumberColumn(format="%.2f")
            for col in display_frame
            if "(%)" in col or col == "Sharpe"
        },
    )
    st.caption(
        f"All five original strategies: 1,712 outcomes, 2 Jan 2020–23 Oct 2026, same {ctx.bps:g} bp cost case. Real changes end 30 Sep 2026. GFC column is a separate 2007–2009 fixed-exposure replay using each 23 Oct 2026 target, without trading/costs. Rows keep research order, not performance rank. No final-period reselection."
    )
    st.download_button(
        "Download matched baseline evidence",
        baselines.to_csv(index=False).encode(),
        file_name="matched_baselines.csv",
        key="gmv_baseline_download",
    )


def _render_net_performance_section(ctx) -> None:
    daily = ctx.daily
    metrics = ctx.metrics
    section("Net performance and drawdown")
    frame = daily.reset_index()
    base_nav = float(frame.nav_start_reported.iloc[0])
    frame["Full-period cumulative return"] = frame.nav_usd / base_nav - 1
    frame["Full-period drawdown"] = (
        frame.nav_usd / np.maximum.accumulate(np.r_[base_nav, frame.nav_usd])[1:] - 1
    )
    lines(
        frame,
        "Date",
        ["Full-period cumulative return", "Full-period drawdown"],
        True,
    )
    st.caption(
        "Performance charts and KPIs retain the full 2020–2026 period and selected cost case. Sidebar dates filter diagnostic backtests only."
    )
    table(
        pd.DataFrame(
            {
                "Metric": [key.replace("_", " ").capitalize() for key in KPI_KEYS],
                "Value": [_format_kpi(key, metrics[key]) for key in KPI_KEYS],
            }
        ),
        hide_index=True,
    )
    st.latex(r"r_{p,t}-r_{f,t}=\alpha+\beta(r_{SPY,t}-r_{f,t})+\epsilon_t")
    st.write(
        "SPY is the market proxy; the same-cost quarterly 40% SPY / 60% AGG portfolio is the information-ratio benchmark. Lagged FRED DGS3MO is the excess-return and Sortino target. Alpha is annualized daily OLS with HAC(5) uncertainty. Its interval includes zero. Undefined ratios remain missing and are displayed as N/A. CAGR uses calendar years; volatility uses 252 sessions. The KPI table formats returns as percentages; raw CSV quantities retain their original fractional units. Trading costs and wealth are USD."
    )
    episode = _compute_drawdown_episode(daily)
    metric_strip(
        [
            MetricItem("Full-period peak", episode.peak_date, "neutral"),
            MetricItem("Trough", episode.trough_date, "neutral"),
            MetricItem("Recovery", episode.recovery_label, "neutral"),
            MetricItem(
                "Gain needed after trough",
                f"{episode.gain_needed_after_trough:.2%}",
                "neutral",
            ),
        ]
    )
    yearly_returns = daily.net_return.groupby(daily.index.year).apply(
        lambda s: (1 + s).prod() - 1
    )
    annual_returns(yearly_returns)
    st.caption("Calendar-year simple returns; 2026 is partial through 23 October.")
    table(
        ctx.release(f"sample_gmv_cost_{ctx.bps:g}bps_trades"),
        hide_index=True,
    )


def render(ctx) -> None:
    wealth_path = _render_purchasing_power_section(ctx)
    _render_spending_screens_section(ctx, wealth_path)
    _render_matched_baselines_section(ctx)
    _render_net_performance_section(ctx)
