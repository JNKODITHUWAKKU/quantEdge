"""Explicit KPI definitions; undefined denominators produce NaN, never infinity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_ANNUALIZATION = 252
DEFAULT_HAC_LAGS = 5
MIN_SAMPLE_SIZE = 20
MIN_MARKET_VARIANCE = 1e-16
DENOMINATOR_TOLERANCE = 1e-12
DAYS_PER_YEAR = 365.25
CONFIDENCE_ALPHA = 0.05
MARKET_PROXY_SYMBOL = "SPY"


def _aligned(*series: pd.Series) -> None:
    if any(not item.index.equals(series[0].index) for item in series[1:]):
        raise ValueError("Metric series must have exactly matching dates; no silent join/drop.")
    if series[0].index.has_duplicates or not series[0].index.is_monotonic_increasing:
        raise ValueError("Metric dates must be unique and increasing.")
    if any(not np.isfinite(item.to_numpy()).all() for item in series):
        raise ValueError("Metric input contains missing/non-finite data.")


def capm(
    portfolio: pd.Series,
    market: pd.Series,
    risk_free: pd.Series,
    annualization: int = DEFAULT_ANNUALIZATION,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> dict:
    _aligned(portfolio, market, risk_free)
    excess_portfolio = (portfolio - risk_free).to_numpy()
    excess_market = (market - risk_free).to_numpy()
    if (
        len(excess_market) < max(MIN_SAMPLE_SIZE, hac_lags + 3)
        or np.var(excess_market) <= MIN_MARKET_VARIANCE
    ):
        raise ValueError("CAPM needs an adequate sample and a nonconstant market excess return.")
    fit = sm.OLS(
        excess_portfolio, sm.add_constant(excess_market, has_constant="add")
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags, "use_correction": True},
        use_t=False,
    )
    confidence_interval = np.asarray(fit.conf_int(alpha=CONFIDENCE_ALPHA))
    return {
        "alpha_daily": float(fit.params[0]),
        "jensen_alpha_annual_arithmetic": float(fit.params[0] * annualization),
        "alpha_ci95_low_annual": float(confidence_interval[0, 0] * annualization),
        "alpha_ci95_high_annual": float(confidence_interval[0, 1] * annualization),
        "alpha_hac_pvalue": float(fit.pvalues[0]),
        "beta": float(fit.params[1]),
        "beta_ci95_low": float(confidence_interval[1, 0]),
        "beta_ci95_high": float(confidence_interval[1, 1]),
        "r_squared": float(fit.rsquared),
        "hac_lags": hac_lags,
        "market_proxy": MARKET_PROXY_SYMBOL,
        "interpretation": "historical single-factor attribution; not proof of skill",
    }


def performance_metrics(
    returns: pd.Series,
    risk_free: pd.Series,
    benchmark: pd.Series,
    market: pd.Series,
    start_date: pd.Timestamp,
    annualization: int = DEFAULT_ANNUALIZATION,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> dict:
    _aligned(returns, risk_free, benchmark, market)
    if (returns <= -1).any() or len(returns) < MIN_SAMPLE_SIZE:
        raise ValueError("Metrics need at least 20 valid returns greater than -100%.")
    days = (returns.index[-1] - pd.Timestamp(start_date)).days
    if days <= 0 or pd.Timestamp(start_date) >= returns.index[0]:
        raise ValueError("A pre-return valuation date is required.")
    growth = np.cumprod(1 + returns.to_numpy())
    peaks = np.maximum.accumulate(np.r_[1.0, growth])[1:]
    max_drawdown = float(-(growth / peaks - 1).min())
    excess, active = returns - risk_free, returns - benchmark
    sd_excess = float(excess.std(ddof=1))
    # Divide by full sample size N (not just negative days) for proper lower partial moment LPM_2.
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0) ** 2)))
    tracking = float(active.std(ddof=1))
    regression = capm(returns, market, risk_free, annualization, hac_lags)
    cagr = float(growth[-1] ** (DAYS_PER_YEAR / days) - 1)
    beta = regression["beta"]
    # Treynor is only meaningful when beta is positive and statistically bounded above zero.
    beta_usable = beta > 0 and regression["beta_ci95_low"] > 0
    annual_scale = np.sqrt(annualization)
    return {
        "n_returns": len(returns),
        "total_return": float(growth[-1] - 1),
        "cagr": cagr,
        "annualized_volatility": float(returns.std(ddof=1) * annual_scale),
        "annualized_mean_excess_return": float(excess.mean() * annualization),
        "sharpe": float(excess.mean() / sd_excess * annual_scale)
        if sd_excess > DENOMINATOR_TOLERANCE
        else np.nan,
        "sortino": float(excess.mean() / downside * annual_scale)
        if downside > DENOMINATOR_TOLERANCE
        else np.nan,
        "downside_deviation_annual": downside * annual_scale,
        "information_ratio": float(active.mean() / tracking * annual_scale)
        if tracking > DENOMINATOR_TOLERANCE
        else np.nan,
        "tracking_error_annual": tracking * annual_scale,
        "maximum_drawdown_loss": max_drawdown,
        "calmar": cagr / max_drawdown if max_drawdown > DENOMINATOR_TOLERANCE else np.nan,
        "treynor": float(excess.mean() * annualization / beta) if beta_usable else np.nan,
        "treynor_status": "defined"
        if beta_usable
        else "not_ranked_nonpositive_or_uncertain_beta",
        "lag1_excess_autocorrelation": float(excess.autocorr(1)),
        **regression,
    }


def mean_uncertainty(
    returns: pd.DataFrame,
    annualization: int = DEFAULT_ANNUALIZATION,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> pd.DataFrame:
    rows = []
    for symbol in returns:
        series = returns[symbol].to_numpy()
        if not np.isfinite(series).all() or len(series) < MIN_SAMPLE_SIZE:
            raise ValueError("Invalid input for mean uncertainty.")
        fit = sm.OLS(series, np.ones((len(series), 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags, "use_correction": True},
            use_t=False,
        )
        confidence_interval = np.asarray(fit.conf_int())[0] * annualization
        rows.append(
            {
                "asset": symbol,
                "mean_annual_arithmetic": float(fit.params[0] * annualization),
                "mean_hac_standard_error_annual": float(fit.bse[0] * annualization),
                "mean_ci95_low_annual": float(confidence_interval[0]),
                "mean_ci95_high_annual": float(confidence_interval[1]),
                "n": len(series),
                "hac_lags": hac_lags,
            }
        )
    return pd.DataFrame(rows)
