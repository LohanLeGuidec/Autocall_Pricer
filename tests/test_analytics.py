from dataclasses import replace

import numpy as np
import pytest

from autocall_pricer import (
    AthenaParams,
    Market,
    PhoenixParams,
    SimSettings,
    fair_coupon,
    fair_value,
    greeks,
    price,
    price_phoenix,
    run_pricing,
    simulate,
    standard_normals,
    worst_of,
)

MARKET = Market(r=0.03, q=0.02, sigma=0.20)
SIM = SimSettings(maturity=5, n_sim=20_000, seed=7, antithetic=True)


@pytest.fixture(scope="module")
def perf():
    return worst_of(simulate(MARKET, SIM))


def test_gbm_is_a_martingale_under_q():
    assets = simulate(MARKET, SIM)
    expected = np.exp((MARKET.r - MARKET.q) * np.arange(6))
    np.testing.assert_allclose(assets[:, :, 0].mean(axis=0), expected, rtol=5e-3)


def test_antithetic_pairs_are_mirrors():
    z = standard_normals(10, 3, seed=1, antithetic=True)
    np.testing.assert_array_equal(z[:5], -z[5:])


def test_antithetic_reduces_confidence_interval():
    plain = price_phoenix(worst_of(simulate(MARKET, replace(SIM, antithetic=False))), PhoenixParams(), MARKET.r)
    anti = price_phoenix(worst_of(simulate(MARKET, SIM)), PhoenixParams(), MARKET.r, antithetic=True)
    assert anti["ci95_pct"] < plain["ci95_pct"]


@pytest.mark.parametrize("product,params", [("phoenix", PhoenixParams()), ("athena", AthenaParams())])
def test_fair_coupon_hits_target(perf, product, params):
    c = fair_coupon(product, params, MARKET, perf, target_pct=98.0)
    assert 0 < c < 0.5
    assert price(product, replace(params, coupon_rate=c), MARKET, perf) == pytest.approx(98.0)


@pytest.mark.parametrize("product,params", [("phoenix", PhoenixParams()), ("athena", AthenaParams())])
def test_greek_signs(perf, product, params):
    g = greeks(product, params, MARKET, SIM, perf)
    assert g["delta"] > 0       # l'investisseur est long du sous-jacent
    assert g["vega"] < 0        # … et vendeur de la put à barrière (short vol)
    assert g["rho"] < 0         # flux lointains actualisés plus fort
    assert g["spread_10bp"] < 0


def test_coupon_and_barrier_monotonicity(perf):
    assert price("phoenix", PhoenixParams(coupon_rate=0.10), MARKET, perf) > \
        price("phoenix", PhoenixParams(coupon_rate=0.05), MARKET, perf)
    assert price("athena", AthenaParams(barrier_capital=0.50), MARKET, perf) > \
        price("athena", AthenaParams(barrier_capital=0.70), MARKET, perf)


def test_run_pricing_end_to_end():
    out = run_pricing(PhoenixParams(), AthenaParams(), MARKET, n_sim=4000, seed=1)
    for k in ("phoenix", "athena"):
        assert 70 < out["results"][k]["fair_value_pct"] < 110
        grid, values = out["sensitivities"]["sigma"][k]
        assert len(grid) == len(values)
    fwd = out["forward_check"][0]
    assert fwd["simulated"] == pytest.approx(fwd["theoretical"], rel=0.02)
    assert "correlation" not in out["sensitivities"]   # mono sous-jacent


def test_maturity_mismatch_raises():
    with pytest.raises(ValueError):
        run_pricing(PhoenixParams(maturity=5), AthenaParams(maturity=6), MARKET, n_sim=100)


def test_fair_value_helper_matches_price(perf):
    assert fair_value("phoenix", PhoenixParams(), MARKET, SIM) == pytest.approx(
        price("phoenix", PhoenixParams(), MARKET, perf))
