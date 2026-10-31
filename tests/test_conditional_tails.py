"""Unit tests for conditional volatility filters, Student-t scaling, and GPD tail splicing."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import genpareto, t
import src.conditional as conditional_module
from src.conditional import (
    ewma_filter,
    fhs_risk,
    fixed_garch_forecast,
    forecast_portfolio,
    garch_filter,
    student_risk,
)
from src.risk import empirical_tail
from src.settings import CONDITIONAL_SETTINGS
from src.storage import csv
from src.tails import (
    choose_threshold,
    fit_gpd,
    gpd_objective,
    hybrid_risk,
    tail_quantile,
)

CFG = deepcopy(CONDITIONAL_SETTINGS)


def test_ewma_variance_is_before_the_current_innovation():
    f = ewma_filter(np.array([1.0, 1.0, 2.0, 3.0]), 0.5, 2)
    np.testing.assert_allclose(f["conditional_sigma"][2:], [1.0, np.sqrt(2.5)])
    assert f["next_sigma"] == pytest.approx(np.sqrt(5.75))
    assert f["residuals"][0] == 2


@pytest.mark.parametrize("nu", [3.0, 6.0, 30.0])
def test_student_es_matches_tail_integral_with_unit_variance_scaling(nu):
    c = 0.99
    mu = 0.001
    sigma = 0.02
    got = student_risk(mu, sigma, nu, c)
    q = t.ppf(c, nu)
    unit = np.sqrt((nu - 2) / nu)
    integral = quad(lambda z: (-mu + sigma * unit * z) * t.pdf(z, nu), q, np.inf)[0] / (1 - c)
    assert got["es"] == pytest.approx(integral, rel=1e-7)
    assert got["var"] == pytest.approx(-mu + sigma * unit * q)


@pytest.mark.parametrize("xi", [-0.2, 0.0, 1e-9, 0.2, 0.9])
def test_gpd_likelihood_gradient_against_numerical_derivative(xi):
    y = np.array([0.1, 0.3, 0.5, 1.0])
    theta = np.array([xi, 0.2])
    analytic = gpd_objective(theta, y)[1]
    numerical = approx_derivative(lambda p: gpd_objective(p, y)[0], theta).reshape(-1)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_exponential_limit_and_tail_domain():
    expected = 1 + 0.5 * np.log(10)
    assert tail_quantile(1, 0.1, 0, 0.5, 0.99) == pytest.approx(expected)
    assert tail_quantile(1, 0.1, 1e-10, 0.5, 0.99) == pytest.approx(expected)
    with pytest.raises(ValueError):
        tail_quantile(1, 0.1, 0.2, 0.5, 0.8)


def splice(xi=0):
    return {
        "fit_status": "OK",
        "threshold": 0.0,
        "shape": xi,
        "scale": 1.0,
        "tail_probability": 0.4,
        "exceedances": 2,
    }


def test_body_quantile_es_includes_both_body_and_gpd():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = hybrid_risk(x, splice(), 0.5)
    assert result["var"] == 0
    assert result["es"] == pytest.approx(0.8)
    result = hybrid_risk(x, splice(), 0.2)
    assert result["var"] == -2
    assert result["es"] == pytest.approx(0.25)
    at_join = hybrid_risk(x, splice(), 0.6)
    assert at_join["var"] == 0 and at_join["es"] == pytest.approx(1.0)
    above = hybrid_risk(x, splice(), 0.9)
    assert above["var"] == pytest.approx(np.log(4))
    assert above["es"] == pytest.approx(np.log(4) + 1)


def test_hybrid_es_nonfinite_shape_is_not_clipped():
    for c in [0.2, 0.9, 0.99]:
        assert np.isinf(hybrid_risk(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), splice(1.1), c)["es"])


def test_gpd_parameter_recovery_and_scale_equivariance():
    p = (np.arange(2000) + 0.5) / 2000
    y = genpareto.ppf(p, 0.2, scale=0.8)
    fit = fit_gpd(y)
    scaled = fit_gpd(10 * y)
    assert fit["fit_status"] == scaled["fit_status"] == "OK"
    assert fit["shape"] == pytest.approx(0.2, abs=0.02)
    assert fit["scale"] == pytest.approx(0.8, abs=0.02)
    assert scaled["shape"] == pytest.approx(fit["shape"], abs=1e-5)
    assert scaled["scale"] == pytest.approx(10 * fit["scale"], rel=1e-5)


def test_threshold_rule_uses_stability_not_highest_var_or_last_result():
    settings = CFG["evt"]
    rows = []
    for p, beta, q in zip(
        [0.8, 0.85, 0.9, 0.925, 0.95],
        [1.0, 1.01, 1.02, 1.03, 1.04],
        [3.0, 3.01, 3.02, 10.0, 20.0],
    ):
        rows.append(
            {
                "fit_status": "OK",
                "finite_es": True,
                "shape": 0.2,
                "scale": beta,
                "modified_scale": 0.8,
                "var_99": q,
                "threshold_percentile": p,
            }
        )
    chosen = choose_threshold(rows, settings)
    assert chosen["region_size"] == 3
    assert chosen["selected"]["threshold_percentile"] == 0.85
    rows[1]["fit_status"] = "OPTIMIZATION_FAILED"
    assert choose_threshold(rows, settings)["selected"] is None


def test_fhs_asset_contributions_and_frozen_filter_scenarios_match():
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, (180, 2))
    w = np.array([0.4, 0.6])
    m = r.mean(axis=0)
    f = ewma_filter(r @ w - m @ w, burn=60)
    result = fhs_risk(r, w, m, f, 0.95)
    assert result["components"].sum() == pytest.approx(result["es"])
    losses = -m @ w - f["next_sigma"] * f["residuals"]
    check = empirical_tail(losses, 0.95)
    assert result["es"] == pytest.approx(check["es"])
    assert result["var"] == pytest.approx(check["var"])


def test_garch_scale_and_next_variance_recursion():
    rng = np.random.default_rng(776)
    x = rng.standard_t(8, 900) * 0.01
    x -= x.mean()
    result = garch_filter(x, CFG["filter"])
    assert result["status"] == "OK"
    expected = (
        result["omega_return_units"]
        + result["alpha"] * x[-1] ** 2
        + result["beta"] * result["conditional_sigma"][-1] ** 2
    )
    assert result["next_sigma"] ** 2 == pytest.approx(expected, rel=1e-10)
    assert len(result["residuals"]) == 840 and result["nu"] > 2
    assert fixed_garch_forecast(x, result) == pytest.approx(result["next_sigma"], rel=1e-10)


def test_fixed_parameter_forecast_uses_observed_endpoint_not_bootstrap_endpoint():
    rng = np.random.default_rng(91)
    observed = rng.normal(0, 0.01, 500)
    parameters = {"omega_return_units": 1e-6, "alpha": 0.1, "beta": 0.85, "nu": 8.0}
    calm = fixed_garch_forecast(observed, parameters)
    shocked = observed.copy()
    shocked[-1] = 0.2
    stress = fixed_garch_forecast(shocked, parameters)
    assert stress > calm
    assert stress**2 - calm**2 == pytest.approx(0.1 * (0.2**2 - observed[-1] ** 2), rel=1e-8)


def test_future_data_perturbation_has_no_effect_on_past_conditional_forecast():
    cfg = deepcopy(CFG)
    cfg["run"]["lookback_returns"] = 300
    rng = np.random.default_rng(88)
    r = pd.DataFrame(
        rng.standard_t(8, (310, 2)) * 0.008,
        index=pd.bdate_range("2013-01-02", periods=310),
        columns=["A", "B"],
    )
    origin = r.index[299]
    exposure = {
        "weights": np.array([0.4, 0.6]),
        "origin_value_usd": 1e7,
        "realization_date": r.index[300],
    }
    first = forecast_portfolio(r.loc[:origin].tail(300), origin, exposure, "example", cfg)[0]
    r.loc[r.index > origin] = -0.7
    second = forecast_portfolio(r.loc[:origin].tail(300), origin, exposure, "example", cfg)[0]
    pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second))
    assert len(first) == 21 and all(row["training_end"] < row["realization_date"] for row in first)


def test_failed_garch_remains_in_ledger_under_declared_fallback(monkeypatch):
    cfg = deepcopy(CFG)
    cfg["run"]["lookback_returns"] = 150
    rng = np.random.default_rng(89)
    r = pd.DataFrame(
        rng.normal(0, 0.01, (150, 2)),
        index=pd.bdate_range("2013-01-02", periods=150),
        columns=["A", "B"],
    )
    monkeypatch.setattr(
        conditional_module,
        "garch_filter",
        lambda *a: {"status": "FIT_FAILED", "reason": "TEST_FAILURE"},
    )
    rows = conditional_module.forecast_portfolio(
        r,
        r.index[-1],
        {
            "weights": np.array([0.5, 0.5]),
            "origin_value_usd": 1e7,
            "realization_date": r.index[-1] + pd.offsets.BDay(),
        },
        "example",
        cfg,
    )[0]
    f = pd.DataFrame(rows)
    selected = f[f.method.str.startswith("garch")]
    assert len(selected) == 12 and selected.actual_model_used.eq("ewma_fhs").all()
    assert selected.fit_status.eq("FALLBACK").all()
    assert np.isfinite(f[["var", "es"]]).all().all()


def test_incomplete_export_does_not_replace_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "forecast.csv"
    target.write_text("previous complete evidence\n")
    original = pd.DataFrame.to_csv

    def incomplete(frame, *args, **kwargs):
        return original(frame.iloc[:1], *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", incomplete)
    with pytest.raises(RuntimeError, match="Incomplete CSV export"):
        csv(pd.DataFrame({"forecast": [1.0, 2.0]}), target)
    assert target.read_text() == "previous complete evidence\n"
