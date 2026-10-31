"""Financial reference cases, not assertions that reproduce implementation loops."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from src.risk import (
    circular_block_indices,
    empirical_tail,
    gaussian_risk,
    historical_risk,
    multivariate_gaussian_check,
    normal_mc_coefficients,
    projected_mc,
    simulation_summary,
)
from src.risk_validation import (
    basel_traffic_light,
    christoffersen,
    evaluate_forecasts,
    joint_fz0,
    kupiec,
    quantile_loss,
)


def test_fractional_tail_mass_known_answer():
    losses = np.r_[np.zeros(247), 0.08, 0.12, 0.20]
    result = empirical_tail(losses, 0.99)
    assert result["var"] == pytest.approx(0.08)
    assert result["tail_mass_observations"] == pytest.approx(2.5)
    assert result["es"] == pytest.approx((0.20 + 0.12 + 0.5 * 0.08) / 2.5)


def test_tied_boundary_mass_is_shared_and_reconciles():
    x = np.array([0, 0, 0.1, 0.1, 0.2])
    result = empirical_tail(x, 0.7)
    assert result["var"] == pytest.approx(0.1)
    assert result["es"] == pytest.approx((0.2 + 0.5 * 0.1) / 1.5)
    np.testing.assert_allclose(result["tail_weights"], [0, 0, 1 / 6, 1 / 6, 2 / 3])


def test_empirical_integer_boundary_is_inverse_cdf_not_linear_interpolation():
    result = empirical_tail(np.arange(1.0, 101.0), 0.95)
    assert result["var"] == 95
    assert result["es"] == 98
    assert result["strict_exceedances_in_sample"] == 5


def test_negative_var_is_not_clamped():
    result = empirical_tail(np.array([-0.05, -0.04, -0.03, -0.02]), 0.5)
    assert result["var"] == -0.04
    assert result["es"] == pytest.approx(-0.025)


@pytest.mark.parametrize("bad", [[0, np.nan], [0, np.inf], []])
def test_missing_or_nonfinite_losses_fail(bad):
    with pytest.raises(ValueError):
        empirical_tail(np.array(bad), 0.99)


@pytest.mark.parametrize("bad", [0, 1, -0.1, np.nan])
def test_invalid_confidence_fails(bad):
    with pytest.raises(ValueError):
        empirical_tail(np.array([0, 0.1]), bad)


def test_historical_components_use_same_portfolio_tail_and_keep_hedge():
    r = np.array([[-0.10, 0.02], [-0.04, 0.01], [0.01, -0.01], [0.02, -0.02]])
    result = historical_risk(r, np.array([0.5, 0.5]), 0.75)
    np.testing.assert_allclose(result["es_components"], [0.05, -0.01])
    assert result["es"] == pytest.approx(0.04)
    assert result["contribution_residual"] == pytest.approx(0, abs=1e-14)


def test_empirical_es_is_monotone_and_scale_equivariant():
    x = np.array([-0.07, 0.01, 0.02, 0.09, 0.15])
    answers = [empirical_tail(x, c) for c in [0.5, 0.8, 0.95]]
    assert np.all(np.diff([r["var"] for r in answers]) >= 0)
    assert np.all(np.diff([r["es"] for r in answers]) >= 0)
    assert empirical_tail(x * 10, 0.8)["es"] == pytest.approx(10 * answers[1]["es"])


def test_es_subadditivity_reference_counterexample_to_var():
    a = np.r_[np.ones(4), np.zeros(96)]
    b = np.r_[np.zeros(4), np.ones(4), np.zeros(92)]
    ra, rb, rab = [empirical_tail(x, 0.95) for x in [a, b, a + b]]
    assert rab["var"] > ra["var"] + rb["var"]
    assert rab["es"] <= ra["es"] + rb["es"] + 1e-12


def test_gaussian_known_one_asset_and_zero_variance():
    r = gaussian_risk(np.array([0.001]), np.array([[0.0004]]), np.ones(1), 0.95)
    assert r["var"] == pytest.approx(-0.001 + 1.6448536269514722 * 0.02)
    assert r["es"] == pytest.approx(-0.001 + 2.0627128075074257 * 0.02)
    z = gaussian_risk(np.array([0.001]), np.zeros((1, 1)), np.ones(1), 0.99)
    assert z["var"] == z["es"] == -0.001


def test_gaussian_euler_components_include_mean_and_hedge():
    mu = np.array([0.001, 0.0002])
    cov = np.array([[0.04, -0.018], [-0.018, 0.01]])
    w = np.array([0.8, 0.2])
    r = gaussian_risk(mu, cov, w, 0.99)
    assert r["es_components"][1] < 0
    assert r["es_components"].sum() == pytest.approx(r["es"])
    assert r["var_components"].sum() == pytest.approx(r["var"])


def test_indefinite_covariance_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        gaussian_risk(np.zeros(2), np.array([[1, 2], [2, 1]]), np.array([0.5, 0.5]), 0.99)


def test_normal_mc_and_correlated_asset_mc_match_analytical_within_error():
    levels = [0.90, 0.95, 0.99]
    mu = np.array([0.0003, 0.0001])
    cov = np.array([[0.0001, 0.000015], [0.000015, 0.000025]])
    w = np.array([0.4, 0.6])
    coeff = normal_mc_coefficients(levels, 91, 0.01, 16, 8192, 32768, 0.001)
    asset = multivariate_gaussian_check(mu, cov, w, levels, 32, 16, 8192)
    for c in levels:
        a = gaussian_risk(mu, cov, w, c)
        projected = projected_mc(coeff[c], a["mean_return"], a["standard_deviation"])
        for result in [projected, asset[c]]:
            for field in ["var", "es"]:
                assert abs(result[field] - a[field]) < 6 * result[field + "_mc_se"]


def test_pooled_mc_empirical_estimates_match_hand_check():
    draws = np.tile(np.arange(1.0, 101.0), (4, 1))
    result = simulation_summary(draws, [0.95])[0.95]
    assert result["var"] == 95
    assert result["es"] == 98
    assert result["var_mc_se"] == 0


def test_common_row_blocks_preserve_internal_sequence():
    rng = np.random.default_rng(9)
    indices = circular_block_indices(20, 5, rng).reshape(-1, 5)
    np.testing.assert_array_equal(np.diff(indices, axis=1) % 20, np.ones((4, 4), int))


def test_kupiec_zero_exceptions_has_finite_exact_limit():
    r = kupiec(np.zeros(250), 0.99)
    assert r["kupiec_lr"] == pytest.approx(-2 * 250 * np.log(0.99))
    assert np.isfinite(r["kupiec_pvalue"])
    assert r["exception_rate_ci95_low"] == 0
    assert 0 < r["exception_rate_ci95_high"] < 0.02


def test_kupiec_all_exceptions_and_exact_nominal_count():
    assert kupiec(np.ones(10), 0.95)["kupiec_lr"] == pytest.approx(-2 * 10 * np.log(0.05))
    r = kupiec(np.r_[np.ones(10), np.zeros(990)], 0.99)
    assert r["kupiec_lr"] == pytest.approx(0, abs=1e-12)
    assert r["kupiec_pvalue"] == pytest.approx(1)


def test_independence_hand_transition_table_and_constant_uninformative():
    r = christoffersen(np.array([0, 0, 1, 1, 0, 1, 0]))
    assert [r[k] for k in ["n00", "n01", "n10", "n11"]] == [1, 2, 2, 1]
    expected = 2 * (2 * np.log(1 / 3) + 4 * np.log(2 / 3) - 6 * np.log(0.5))
    assert r["christoffersen_lr"] == pytest.approx(expected)
    r = christoffersen(np.zeros(250))
    assert r["christoffersen_pvalue"] is None
    assert "INSUFFICIENT" in r["independence_status"]


def test_independence_detects_clustering_at_identical_coverage():
    clustered = np.r_[np.ones(20), np.zeros(980)]
    spaced = np.zeros(1000)
    spaced[np.arange(0, 1000, 50)] = 1
    assert kupiec(clustered, 0.98)["kupiec_lr"] == pytest.approx(kupiec(spaced, 0.98)["kupiec_lr"])
    assert (
        christoffersen(clustered)["christoffersen_lr"] > christoffersen(spaced)["christoffersen_lr"]
    )
    assert christoffersen(clustered)["christoffersen_pvalue"] < 0.001


@pytest.mark.parametrize(
    "count,zone", [(0, "GREEN"), (4, "GREEN"), (5, "AMBER"), (9, "AMBER"), (10, "RED")]
)
def test_basel_exact_boundaries(count, zone):
    assert (
        basel_traffic_light(np.r_[np.ones(count), np.zeros(250 - count)], 0.99)["basel_zone"]
        == zone
    )


def test_basel_other_confidence_or_length_is_not_classic():
    assert basel_traffic_light(np.zeros(250), 0.95)["basel_zone"] == "NOT_APPLICABLE"
    assert basel_traffic_light(np.zeros(249), 0.99)["basel_zone"] == "INSUFFICIENT_OR_NONCONTIGUOUS"


def test_fz0_sign_translation_and_invalid_domain():
    result = joint_fz0(np.array([0.04, 0.01]), np.array([0.02, 0.02]), np.array([0.03, 0.03]), 0.95)
    np.testing.assert_allclose(
        result,
        [
            (0.02 + 0.02 / 0.05) / 0.03 + np.log(0.03) - 1,
            0.02 / 0.03 + np.log(0.03) - 1,
        ],
    )
    assert np.isnan(joint_fz0(np.array([0.0]), np.array([-0.01]), np.array([0.0]), 0.95)[0])
    assert quantile_loss(np.array([0.04]), np.array([0.02]), 0.95)[0] == pytest.approx(0.019)


def test_joint_score_expectation_minimized_at_true_normal_pair():
    c = 0.95
    q0 = float(norm.ppf(c))
    e0 = float(norm.pdf(q0) / (1 - c))

    # Integrate the normal positive-part expectation analytically, separately
    # from joint_fz0's observation-level formula.
    def expectation(q, e):
        positive_part = norm.pdf(q) - q * norm.sf(q)
        return (q + positive_part / (1 - c)) / e + np.log(e) - 1

    truth = expectation(q0, e0)
    for dq, de in [(0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2), (0.1, 0.1)]:
        assert expectation(q0 + dq, e0 + de) > truth


def test_gap_is_not_a_consecutive_exception_transition():
    dates = pd.to_datetime(["2010-01-04", "2010-01-05", "2010-01-08"])
    origins = pd.to_datetime(["2010-01-01", "2010-01-04", "2010-01-07"])
    data = pd.DataFrame(
        {
            "realization_date": dates,
            "forecast_origin": origins,
            "confidence": 0.95,
            "var": 0.02,
            "es": 0.03,
            "realized_hpl_loss": [0.01, 0.04, 0.05],
            "pnl_type": "frozen_exposure_market_loss",
        }
    )
    r = evaluate_forecasts(data)
    assert r["forecast_gaps"] == 1
    assert r["eligible_transitions"] == 1


def test_missing_forecast_is_not_silently_dropped():
    data = pd.DataFrame(
        {
            "realization_date": ["2010-01-05"],
            "forecast_origin": ["2010-01-04"],
            "confidence": [0.95],
            "var": [np.nan],
            "es": [0.03],
            "realized_hpl_loss": [0.01],
            "pnl_type": ["frozen_exposure_market_loss"],
        }
    )
    with pytest.raises(ValueError, match="Missing or failed"):
        evaluate_forecasts(data)
