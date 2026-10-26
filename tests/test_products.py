"""
Payoffs testés sur des trajectoires construites à la main : chaque cas a un résultat
calculable de tête. r = 0 pour lire directement les flux en euros.
"""
import numpy as np
import pytest

from autocall_pricer import AthenaParams, PhoenixParams, price_athena, price_phoenix


def path(*levels):
    """Une trajectoire : niveau initial 1, puis un niveau par année."""
    return np.array([[1.0, *levels]])


def phoenix_value(levels, **kw):
    p = PhoenixParams(maturity=len(levels), **kw)
    return price_phoenix(path(*levels), p, r=0.0)["fair_value_eur"]


def athena_value(levels, **kw):
    p = AthenaParams(maturity=len(levels), **kw)
    return price_athena(path(*levels), p, r=0.0)["fair_value_eur"]


# ── Phoenix ─────────────────────────────────────────────────────────────────

def test_phoenix_recall_year_one():
    # Rappel en t=1 : nominal + 1 coupon
    assert phoenix_value([1.05, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(1080)


def test_phoenix_memory_catches_up_missed_coupons():
    # t1 = 70 % (sous barrière coupon) ; t2 = 90 % (coupon + rattrapage t1) ; t3 = 105 % rappel
    # Flux : t2 → 160, t3 → 80 + 1000  ⇒ 1240
    assert phoenix_value([0.70, 0.90, 1.05, 1.0, 1.0]) == pytest.approx(1240)


def test_phoenix_without_memory_loses_missed_coupon():
    assert phoenix_value([0.70, 0.90, 1.05, 1.0, 1.0], memory=False) == pytest.approx(1160)


def test_phoenix_coupons_paid_are_kept_after_capital_loss():
    # 4 coupons versés (t1…t4 ≥ 80 %), puis S(T) = 50 % < barrière capital
    assert phoenix_value([0.95, 0.90, 0.85, 0.82, 0.50]) == pytest.approx(4 * 80 + 500)


def test_phoenix_maturity_between_barriers():
    # Jamais rappelé ni coupon, S(T) = 70 % : au-dessus du capital, sous la barrière coupon
    assert phoenix_value([0.7, 0.7, 0.7, 0.7, 0.7]) == pytest.approx(1000)


def test_phoenix_maturity_coupon_with_memory():
    # Aucun coupon avant T, S(T) = 85 % : nominal + 5 coupons rattrapés
    assert phoenix_value([0.7, 0.7, 0.7, 0.7, 0.85]) == pytest.approx(1000 + 5 * 80)


def test_phoenix_step_down():
    # Barrière de rappel 100 %, -5 %/an : en t3 elle vaut 90 % → rappel à 92 %
    v = phoenix_value([0.85, 0.85, 0.92, 1.0, 1.0], step_down=0.05)
    assert v == pytest.approx(3 * 80 + 1000)


# ── Athena ──────────────────────────────────────────────────────────────────

def test_athena_recall_pays_accumulated_coupons():
    assert athena_value([0.9, 1.02, 1.0, 1.0, 1.0], coupon_rate=0.10) == pytest.approx(1200)


def test_athena_maturity_between_capital_and_final_barrier_returns_nominal_only():
    assert athena_value([0.9, 0.9, 0.9, 0.9, 0.70], barrier_capital=0.60) == pytest.approx(1000)


def test_athena_maturity_above_final_barrier():
    assert athena_value([0.9, 0.9, 0.9, 0.9, 1.01], coupon_rate=0.10) == pytest.approx(1500)


def test_athena_capital_loss():
    assert athena_value([0.9, 0.9, 0.9, 0.9, 0.45]) == pytest.approx(450)


def test_athena_custom_final_barrier():
    # Barrière finale abaissée à 80 % : S(T) = 85 % → coupons versés
    assert athena_value([0.9, 0.9, 0.9, 0.9, 0.85], coupon_rate=0.10, barrier_final=0.80) == pytest.approx(1500)


# ── Actualisation et statistiques ───────────────────────────────────────────

def test_discounting_includes_credit_spread():
    p = PhoenixParams(maturity=5)
    pv = price_phoenix(path(1.05, 1, 1, 1, 1), p, r=0.03, credit_spread=0.01)["fair_value_eur"]
    assert pv == pytest.approx(1080 * np.exp(-0.04))


def test_statistics_on_mixed_paths():
    paths = np.vstack([path(1.05, 1, 1, 1, 1), path(0.9, 0.9, 0.9, 0.9, 0.5)])
    res = price_phoenix(paths, PhoenixParams(maturity=5), r=0.0)
    assert res["early_recall_pct"] == 50
    assert res["loss_pct"] == 50
    assert res["avg_duration"] == pytest.approx(3.0)
    assert res["recall_by_year"][1] == 50
