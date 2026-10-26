"""Worst-of corrélé et modèle de Heston."""
from dataclasses import replace

import numpy as np
import pytest

from autocall_pricer import (
    AthenaParams,
    Market,
    ModelSpec,
    PhoenixParams,
    SimSettings,
    fair_value,
    implied_smile,
    run_pricing,
    simulate,
    simulate_assets,
    worst_of,
)

SIM = SimSettings(maturity=5, n_sim=20_000, seed=11, antithetic=True)
HESTON = ModelSpec("heston", kappa=1.5, xi=0.5, rho=-0.7, term_ratio=1.0)


# ── Worst-of ────────────────────────────────────────────────────────────────

def test_worst_of_is_the_minimum_performance():
    paths = np.array([[[1, 1], [1.10, 0.95], [0.80, 1.20]]])
    np.testing.assert_allclose(worst_of(paths), [[1, 0.95, 0.80]])


def test_simulated_correlation_matches_input():
    a = simulate_assets(50_000, 1, (0.2, 0.25), (0.0, 0.0), 0.02, correlation=0.6, seed=3)
    rets = np.log(a[:, 1, :])
    assert np.corrcoef(rets.T)[0, 1] == pytest.approx(0.6, abs=0.02)


def test_worst_of_is_cheaper_than_single_underlying():
    single = Market(r=0.03, q=0.02, sigma=0.2)
    basket = Market(r=0.03, q=(0.02, 0.02, 0.02), sigma=(0.2, 0.2, 0.2), correlation=0.7)
    assert fair_value("phoenix", PhoenixParams(), basket, SIM) < fair_value("phoenix", PhoenixParams(), single, SIM)


def test_higher_correlation_makes_worst_of_more_valuable():
    low = Market(r=0.03, q=(0.02, 0.02), sigma=(0.2, 0.2), correlation=0.2)
    high = replace(low, correlation=0.9)
    assert fair_value("athena", AthenaParams(), high, SIM) > fair_value("athena", AthenaParams(), low, SIM)


def test_perfect_correlation_identical_assets_equals_single():
    single = Market(r=0.03, q=0.02, sigma=0.2)
    twins = Market(r=0.03, q=(0.02, 0.02), sigma=(0.2, 0.2), correlation=0.999999)
    assert fair_value("phoenix", PhoenixParams(), twins, SIM) == pytest.approx(
        fair_value("phoenix", PhoenixParams(), single, replace(SIM, n_sim=SIM.n_sim)), abs=0.6)


# ── Heston ──────────────────────────────────────────────────────────────────

def test_heston_is_a_martingale():
    m = Market(r=0.03, q=0.01, sigma=0.2, model=HESTON)
    a = simulate(m, SIM)
    assert a[:, -1, 0].mean() == pytest.approx(np.exp(0.02 * 5), rel=0.01)


def test_heston_without_vol_of_vol_reduces_to_gbm():
    gbm = Market(r=0.03, q=0.02, sigma=0.2)
    flat = Market(r=0.03, q=0.02, sigma=0.2, model=ModelSpec("heston", xi=1e-8, rho=0.0))
    assert fair_value("phoenix", PhoenixParams(), flat, SIM) == pytest.approx(
        fair_value("phoenix", PhoenixParams(), gbm, SIM), abs=0.5)


def test_heston_with_negative_correlation_produces_downside_skew():
    m = Market(r=0.03, q=0.02, sigma=0.2, model=HESTON)
    strikes, vols = implied_smile(simulate(m, SIM)[:, -1, 0], 5, 0.03, 0.02)
    v70, v100, v120 = (vols[np.isclose(strikes, k)][0] for k in (0.7, 1.0, 1.2))
    assert v70 > v100 > v120


def test_gbm_smile_is_flat():
    m = Market(r=0.03, q=0.02, sigma=0.2)
    strikes, vols = implied_smile(simulate(m, SIM)[:, -1, 0], 5, 0.03, 0.02)
    ok = ~np.isnan(vols)
    np.testing.assert_allclose(vols[ok], 0.2, atol=0.01)


def test_skew_lowers_phoenix_value_at_equal_atm_vol():
    """
    À vol ATM égale, le skew renchérit la put à barrière vendue par l'investisseur :
    le Phoenix vaut moins sous Heston. (L'Athena peut aller dans l'autre sens : ses gros
    coupons au rappel sont une digitale à 100 %, qui vaut plus avec un skew négatif.)
    """
    out = run_pricing(PhoenixParams(), AthenaParams(), Market(r=0.03, q=0.02, sigma=0.2), n_sim=20_000, seed=5,
                      heston_for_comparison=HESTON, with_sensitivities=False)
    cmp_ = out["model_comparison"]["phoenix"]
    assert cmp_["skew_effect"] < -0.5
    # La vol ATM du Heston est inférieure à √v₀ (convexité de la vol of vol)
    assert out["smile"]["atm_vols"][0] < 0.2


def test_run_pricing_worst_of_heston():
    m = Market(r=0.03, q=(0.02, 0.01), sigma=(0.2, 0.25), correlation=0.8, model=HESTON)
    out = run_pricing(PhoenixParams(), AthenaParams(), m, n_sim=3000, seed=2)
    assert "correlation" in out["sensitivities"]
    assert out["worst_share"].sum() == pytest.approx(100)
    assert len(out["forward_check"]) == 2


# ── Comparaison Heston / Black-Scholes à structure par terme égale ──────────

def test_constant_vol_schedule_matches_constant_vol():
    flat = simulate_assets(2_000, 3, (0.2,), (0.01,), 0.03, seed=5)
    sched = simulate_assets(2_000, 3, (0.2,), (0.01,), 0.03, seed=5, vol_schedule=[[0.2]] * 3)
    np.testing.assert_allclose(flat, sched)


def test_vol_schedule_is_a_martingale():
    a = simulate_assets(200_000, 3, (0.2,), (0.01,), 0.03, seed=2, antithetic=True,
                        vol_schedule=[[0.15], [0.25], [0.35]])
    assert a[:, 3, 0].mean() == pytest.approx(np.exp(0.02 * 3), rel=4e-3)


def test_benchmark_matches_heston_atm_vol_at_every_date():
    from autocall_pricer.analytics import atm_vol_term_structure

    m = Market(r=0.03, q=0.01, sigma=0.16, model=replace(HESTON, term_ratio=1.3))
    out = run_pricing(PhoenixParams(), AthenaParams(), m, n_sim=20_000, with_sensitivities=False)
    gbm_market = replace(m, model=ModelSpec("gbm"), sigma=out["smile"]["atm_vols"],
                         sigma_schedule=tuple(map(tuple, atm_vol_term_structure(
                             simulate(m, SIM), m.r, m.qs)[1])))
    atm_h, _ = atm_vol_term_structure(simulate(m, SIM), m.r, m.qs)
    atm_g, _ = atm_vol_term_structure(simulate(gbm_market, SIM), m.r, m.qs)
    np.testing.assert_allclose(atm_g, atm_h, atol=0.004)
    assert atm_h[0, 0] < atm_h[-1, 0]          # structure par terme croissante bien reproduite


def test_leg_decomposition_is_consistent():
    m = Market(r=0.03, q=0.01, sigma=0.18, model=HESTON)
    out = run_pricing(PhoenixParams(), AthenaParams(), m, n_sim=10_000, with_sensitivities=False)
    for model in ("heston", "gbm_atm"):
        ph = out["model_comparison"]["phoenix"]["detail"][model]
        at = out["model_comparison"]["athena"]["detail"][model]
        assert ph["nominal_leg"] == pytest.approx(at["nominal_leg"])      # mêmes barrières : même jambe
        assert ph["nominal_leg"] + ph["coupon_leg"] == pytest.approx(ph["fair_value"])
