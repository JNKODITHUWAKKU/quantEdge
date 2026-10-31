"""Hand-solvable finance checks and chronology/optimization invariants."""

import numpy as np
import pandas as pd
import pytest
from src.accounting import rebalance_cost, run_portfolio
from src.metrics import capm, performance_metrics
from src.portfolio import estimate_moments, solve_markowitz


def test_initial_purchase_pays_its_own_cost():
    result = rebalance_cost(np.zeros(2), np.array([0.4, 0.6]), 0.001)
    assert result["cost_fraction"] == pytest.approx(0.001 / 1.001)
    assert result["gross_traded_fraction"] == pytest.approx(1 / 1.001)


def test_full_rotation_pays_both_sale_and_purchase():
    result = rebalance_cost(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.001)
    assert result["cost_fraction"] == pytest.approx(0.002 / 1.001)
    assert np.sum(result["trade_fractions"]) == pytest.approx(-result["cost_fraction"])


def test_no_trade_means_no_cost():
    result = rebalance_cost(np.array([0.4, 0.6]), np.array([0.4, 0.6]), 0.001)
    assert result["cost_fraction"] == 0


@pytest.mark.parametrize("rate", [-0.01, 1.0, np.nan])
def test_invalid_cost_fails(rate):
    with pytest.raises(ValueError):
        rebalance_cost(np.array([0.5, 0.5]), np.array([0.4, 0.6]), rate)


def drift_prices():
    return pd.DataFrame(
        {
            "SPY": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0],
            "AGG": [100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 99.0],
        },
        index=pd.bdate_range("2010-01-04", periods=7),
    )


def test_weight_drift_matches_independent_holdings_arithmetic():
    result = run_portfolio(
        drift_prices(), "equal_weight", lookback=3, cost_bps=0, initial_value=10000
    )
    assert result.daily.iloc[0]["net_return"] == pytest.approx(0)
    assert result.daily.iloc[0]["end_weight_SPY"] == pytest.approx(0.55)
    assert result.daily.iloc[1]["net_return"] == pytest.approx(0.045)
    assert result.daily.iloc[-1]["nav_usd"] == pytest.approx(10450)


def test_wealth_cost_and_return_products_reconcile():
    result = run_portfolio(
        drift_prices(), "equal_weight", lookback=3, cost_bps=5, initial_value=10000
    )
    assert np.prod(1 + result.daily["net_return"]) == pytest.approx(
        result.daily["nav_usd"].iloc[-1] / 10000
    )
    assert result.daily["trading_cost_dollars"].sum() == pytest.approx(
        result.trades["cost_dollars"].sum()
    )
    assert np.abs(result.daily["accounting_residual"]).max() < 1e-8


def test_ordering_has_no_same_close_execution():
    result = run_portfolio(drift_prices(), "equal_weight", lookback=3, cost_bps=0)
    assert (result.signals["training_end"] < result.signals["execution_date"]).all()
    assert (result.signals["execution_date"] < result.signals["first_affected_return"]).all()


def test_later_prices_cannot_change_earlier_decisions():
    rng = np.random.default_rng(71031)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (220, 2)), axis=0),
        index=pd.bdate_range("2010-01-04", periods=220),
        columns=["SPY", "AGG"],
    )
    changed = prices.copy()
    cutoff = prices.index[150]
    changed.loc[cutoff:, "SPY"] *= 1.5
    original = run_portfolio(prices, "sample_gmv", lookback=40, cost_bps=5)
    altered = run_portfolio(changed, "sample_gmv", lookback=40, cost_bps=5)
    pd.testing.assert_frame_equal(
        original.signals[original.signals.training_end < cutoff],
        altered.signals[altered.signals.training_end < cutoff],
    )
    pd.testing.assert_frame_equal(
        original.daily.loc[: prices.index[149]], altered.daily.loc[: prices.index[149]]
    )


def test_diagonal_minimum_variance_has_known_solution():
    # For diagonal cov(0.04, 0.09), GMV weights are proportional to inverse variances (1/0.04 : 1/0.09 = 9:4).
    result = solve_markowitz(np.array([0.05, 0.1]), np.diag([0.04, 0.09]))
    np.testing.assert_allclose(result["weights"], [9 / 13, 4 / 13], atol=1e-7)
    assert result["scaled_optimality_gap"] < 1e-6


def test_return_constraint_has_known_two_asset_solution():
    result = solve_markowitz(np.array([0.05, 0.1]), np.diag([0.04, 0.09]), target=0.09)
    np.testing.assert_allclose(result["weights"], [0.2, 0.8], atol=1e-7)


def test_unattainable_target_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        solve_markowitz(np.array([0.05, 0.1]), np.diag([0.04, 0.09]), target=0.11)


def test_indefinite_covariance_rejected():
    with pytest.raises(ValueError, match="positive definite"):
        solve_markowitz(np.array([0.05, 0.1]), np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_shrinkage_matches_disclosed_scaled_identity_formula():
    rng = np.random.default_rng(10)
    returns = pd.DataFrame(rng.normal(size=(80, 3)), index=pd.bdate_range("2010-01-04", periods=80))
    sample = estimate_moments(returns, "sample")
    shrunk = estimate_moments(returns, "shrinkage")
    delta = shrunk["shrinkage_intensity"]
    s = sample["covariance"]
    expected = (1 - delta) * s + delta * np.trace(s) / 3 * np.eye(3)
    np.testing.assert_allclose(shrunk["covariance"], expected, atol=1e-12)


def synthetic_metric_series():
    index = pd.bdate_range("2010-01-04", periods=60)
    market = pd.Series(np.tile([-0.02, 0.01, 0.025, -0.005, 0.003], 12), index=index)
    rate = pd.Series(0.0001, index=index)
    portfolio = rate + 0.0002 + 1.5 * (market - rate)
    return portfolio, market, rate


def test_capm_recovers_known_alpha_and_beta():
    portfolio, market, rate = synthetic_metric_series()
    result = capm(portfolio, market, rate)
    assert result["beta"] == pytest.approx(1.5)
    assert result["alpha_daily"] == pytest.approx(0.0002)
    assert result["jensen_alpha_annual_arithmetic"] == pytest.approx(0.0504)


def test_sortino_uses_all_observations_in_denominator():
    portfolio, market, rate = synthetic_metric_series()
    result = performance_metrics(
        portfolio, rate, market, market, portfolio.index[0] - pd.Timedelta(days=1)
    )
    excess = (portfolio - rate).to_numpy()
    expected = (
        np.mean(excess) * np.sqrt(252) / np.sqrt(np.sum(np.minimum(excess, 0) ** 2) / len(excess))
    )
    assert result["sortino"] == pytest.approx(expected)


def test_benchmark_information_ratio_is_undefined():
    portfolio, market, rate = synthetic_metric_series()
    result = performance_metrics(
        portfolio, rate, portfolio, market, portfolio.index[0] - pd.Timedelta(days=1)
    )
    assert np.isnan(result["information_ratio"])


def test_initial_loss_counts_in_drawdown():
    _, market, rate = synthetic_metric_series()
    r = pd.Series(0.0, index=market.index)
    r.iloc[0] = -0.1
    result = performance_metrics(r, rate, market, market, r.index[0] - pd.Timedelta(days=1))
    assert result["maximum_drawdown_loss"] == pytest.approx(0.1)


def test_misaligned_metric_dates_rejected():
    portfolio, market, rate = synthetic_metric_series()
    with pytest.raises(ValueError, match="matching dates"):
        capm(portfolio, market.iloc[1:], rate)
