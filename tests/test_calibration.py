"""Formule semi-fermée de Heston et calibration sur une surface de volatilité."""
import numpy as np
import pytest

from autocall_pricer import (
    HestonParams,
    Market,
    OptionQuotes,
    SimSettings,
    calibrate_heston,
    fair_value,
    heston_implied_vols,
    heston_price,
    load_surface_csv,
    simulate_assets,
)
from autocall_pricer.calibration import black_implied_vol, black_price

R = 0.03
TRUE = HestonParams(v0=0.03, theta=0.05, kappa=2.0, xi=0.7, rho=-0.75)


def synthetic_quotes(params=TRUE, noise=0.0, seed=0):
    Ts = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0])
    Ks = np.linspace(0.75, 1.25, 11)
    T, K = (x.ravel() for x in np.meshgrid(Ts, Ks))
    F = np.exp(R * T)
    vol = heston_implied_vols(F, K, T, R, params)
    vol = vol + noise * np.random.default_rng(seed).standard_normal(vol.shape)
    return OptionQuotes(strike=K, T=T, forward=F, vol=vol, r=R, source="synthétique")


# ── Formule semi-fermée ─────────────────────────────────────────────────────

def test_heston_converges_to_black_when_vol_of_vol_vanishes():
    p = HestonParams(v0=0.04, theta=0.04, kappa=1.5, xi=1e-4, rho=-0.7)
    F, K, T = np.ones(3), np.array([0.8, 1.0, 1.2]), np.array([0.1, 1.0, 5.0])
    np.testing.assert_allclose(heston_price(F, K, T, R, p), black_price(F, K, T, R, 0.2), atol=2e-5)


def test_put_call_parity():
    F, K, T = 1.05, 0.9, 2.0
    c = heston_price(F, K, T, R, TRUE, call=True)
    p = heston_price(F, K, T, R, TRUE, call=False)
    assert c - p == pytest.approx(np.exp(-R * T) * (F - K), abs=1e-10)


def test_formula_matches_monte_carlo():
    """Même modèle, deux méthodes : la formule doit tomber dans l'intervalle de confiance du Monte Carlo."""
    spec = TRUE.to_model_spec(steps_per_year=104)
    paths = simulate_assets(200_000, 2, (TRUE.sigma0,), (0.0,), R, model=spec, seed=1, antithetic=True)
    for T in (1, 2):
        ST, F = paths[:, T, 0], np.exp(R * T)
        for K in (0.8, 1.0, 1.2):
            call = K >= F
            payoff = np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0)
            mc = np.exp(-R * T) * payoff.mean()
            se = np.exp(-R * T) * payoff.std() / np.sqrt(len(payoff))
            assert float(heston_price(F, K, T, R, TRUE, call)) == pytest.approx(mc, abs=4 * se)


def test_negative_rho_produces_a_skew():
    vols = heston_implied_vols(np.ones(2), np.array([0.9, 1.1]), np.ones(2), R, TRUE)
    assert vols[0] > vols[1]


def test_black_implied_vol_round_trip():
    price = float(black_price(1.0, 1.1, 0.5, R, 0.23, True))
    assert black_implied_vol(price, 1.0, 1.1, 0.5, R, True) == pytest.approx(0.23, abs=1e-8)


# ── Calibration ─────────────────────────────────────────────────────────────

def test_calibration_recovers_known_parameters():
    res = calibrate_heston(synthetic_quotes())
    p = res.params
    assert res.rmse_vol < 1e-4
    assert p.v0 == pytest.approx(TRUE.v0, rel=1e-3)
    assert p.theta == pytest.approx(TRUE.theta, rel=1e-2)
    assert p.kappa == pytest.approx(TRUE.kappa, rel=5e-2)
    assert p.xi == pytest.approx(TRUE.xi, rel=5e-2)
    assert p.rho == pytest.approx(TRUE.rho, abs=0.02)


def test_calibration_is_robust_to_quote_noise():
    """Bruit de 0,3 point de vol sur chaque cotation : l'erreur de calibration reste de cet ordre."""
    res = calibrate_heston(synthetic_quotes(noise=0.003))
    assert res.rmse_vol < 0.005
    assert res.params.rho == pytest.approx(TRUE.rho, abs=0.1)


def test_calibrated_parameters_fit_the_pricer():
    """Les paramètres calibrés passent tels quels dans le pricer d'autocalls."""
    from autocall_pricer import PhoenixParams

    p = calibrate_heston(synthetic_quotes()).params
    market = Market(r=R, q=0.0, sigma=p.sigma0, model=p.to_model_spec())
    fv = fair_value("phoenix", PhoenixParams(), market, SimSettings(5, 5_000, 3))
    assert 50 < fv < 110


def test_too_few_options_is_rejected():
    q = synthetic_quotes()
    small = OptionQuotes(q.strike[:4], q.T[:4], q.forward[:4], q.vol[:4], R)
    with pytest.raises(ValueError):
        calibrate_heston(small)


def test_load_surface_csv(tmp_path):
    path = tmp_path / "surface.csv"
    path.write_text("maturity_years;strike_pct;implied_vol_pct\n1;90;22,5\n1;100;20\n1;110;18,1\n", encoding="utf-8")
    q = load_surface_csv(str(path), r=R)
    assert len(q) == 3
    np.testing.assert_allclose(q.vol, [0.225, 0.20, 0.181])
    np.testing.assert_allclose(q.forward, np.exp(R))


# ── Interface ───────────────────────────────────────────────────────────────

def test_calibration_endpoint_fills_the_form(client, fake_market, monkeypatch):
    from pricer import views

    monkeypatch.setattr(views, "fetch_option_quotes", lambda name, r: synthetic_quotes())
    d = client.get("/calibration/", {"index": "S&P 500"}).json()
    assert d["sigma"] == pytest.approx(TRUE.sigma0 * 100, abs=0.05)
    assert d["rho"] == pytest.approx(TRUE.rho, abs=0.02)
    assert d["term_ratio"] == pytest.approx(np.sqrt(TRUE.theta / TRUE.v0), abs=0.01)
    assert d["rmse"] < 0.05


def test_calibration_endpoint_reports_missing_options(client, fake_market):
    r = client.get("/calibration/", {"index": "CAC 40"})
    assert r.status_code == 422 and "CSV" in r.json()["error"]


def test_management_command_with_csv(tmp_path):
    from io import StringIO

    from django.core.management import call_command

    q = synthetic_quotes()
    lines = ["maturity_years;strike_pct;implied_vol_pct;forward_pct"]
    lines += [f"{t};{k * 100};{v * 100};{f * 100}" for t, k, v, f in zip(q.T, q.strike, q.vol, q.forward)]
    path = tmp_path / "surface.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    out = StringIO()
    call_command("calibrate_heston", csv=str(path), r=R, stdout=out)
    text = out.getvalue()
    assert "Paramètres à saisir" in text and "-0.75" in text


def test_fetch_option_quotes_parses_a_yahoo_chain(monkeypatch):
    """Chaîne d'options simulée au format yfinance : forward par parité, vols recalculées, aucun appel réseau."""
    import datetime as dt
    import sys
    import types

    import pandas as pd

    from autocall_pricer.calibration import fetch_option_quotes

    spot, r, today = 5000.0, 0.04, dt.date.today()
    expiries = [(today + dt.timedelta(days=d)).isoformat() for d in (10, 45, 90, 180, 365, 730, 1100)]

    class FakeTicker:
        options = expiries

        def history(self, period):
            return pd.DataFrame({"Close": [spot]})

        def option_chain(self, expiry):
            T = (dt.date.fromisoformat(expiry) - today).days / 365
            F, K = spot * np.exp(r * T), np.arange(3500, 6600, 100.0)

            def frame(px):
                return pd.DataFrame({"strike": K, "bid": px * 0.99, "ask": px * 1.01, "lastPrice": px})

            return types.SimpleNamespace(calls=frame(heston_price(F, K, T, r, TRUE, True)),
                                         puts=frame(heston_price(F, K, T, r, TRUE, False)))

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=lambda t: FakeTicker()))
    q = fetch_option_quotes("S&P 500", r)
    assert len(q) > 50 and q.T.min() >= 30 / 365          # l'échéance à 10 jours est écartée
    res = calibrate_heston(q)
    assert res.rmse_vol < 1e-3 and res.params.rho == pytest.approx(TRUE.rho, abs=0.02)
def test_calibrate_button_on_home(client, fake_market):
    html = client.get("/").content.decode()
    assert 'id="calibrate-btn"' in html and "Calibrer sur les options cotées" in html


def test_summary_shows_model_comparison(client, fake_market):
    from tests.test_django import PARAMS

    html = client.get("/", {**PARAMS, "model": "heston"}).content.decode()
    assert "Black-Scholes (mêmes vols ATM)" in html and "effet du skew" in html
    assert "Jambe coupons" in html and "à chaque date d" in html
