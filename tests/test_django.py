from autocall_pricer import get_market_data

PARAMS = {
    "underlying_1": "S&P 500", "sigma_1": "18", "q_1": "1.2", "underlying_2": "", "sigma_2": "", "q_2": "",
    "underlying_3": "", "sigma_3": "", "q_3": "", "correlation": "70",
    "r": "3", "credit_spread": "0.8", "maturity": "5", "target": "98",
    "model": "gbm", "heston_kappa": "1.5", "heston_xi": "0.5", "heston_rho": "-0.7", "term_ratio": "1.0",
    "n_sim": "5000", "seed": "42", "antithetic": "on",
    "ph_coupon": "8", "ph_recall": "100", "ph_coupon_barrier": "80", "ph_capital": "60", "ph_step_down": "0",
    "ph_memory": "on",
    "at_coupon": "10", "at_recall": "100", "at_capital": "60", "at_final": "", "at_step_down": "0",
}
WORST_OF_HESTON = {**PARAMS, "underlying_2": "Euro Stoxx 50", "sigma_2": "20", "q_2": "3.2",
                   "underlying_3": "Nasdaq 100", "sigma_3": "22", "q_3": "0.7", "model": "heston"}


def test_market_data_sources(fake_market):
    d = get_market_data("S&P 500")
    assert d["spot"] == 5000 and d["sigma"] == 0.16 and d["r"] == 0.04
    assert "VIX" in d["sources"]["sigma"]
    eu = get_market_data("CAC 40")          # pas d'indice de vol ni de taux sur Yahoo
    assert eu["sigma"] == 0.18 and eu["sources"]["sigma"] == "historique 1 an"
    assert eu["sources"]["r"].startswith("défaut")
    assert d["term_ratio"] == 1.1 and "VIX1Y" in d["sources"]["term_ratio"]
    assert eu["term_ratio"] == 1.0


def test_home_prefilled_with_market_data(client, fake_market):
    html = client.get("/").content.decode()
    assert 'name="sigma_1"' in html and 'value="16.0"' in html
    assert 'name="term_ratio"' in html and 'value="1.1"' in html
    assert "Pricer" in html


def test_full_pricing_page(client, fake_market):
    html = client.get("/", PARAMS).content.decode()
    assert "Erreur" not in html
    assert "Coupon d&#x27;équilibre" in html or "Coupon d'équilibre" in html
    assert html.count('class="chart"') >= 9
    assert "Effet du skew" in html and "worst-of" not in html.split("results-header")[1][:300]


def test_worst_of_heston_page(client, fake_market):
    html = client.get("/", WORST_OF_HESTON).content.decode()
    assert "Erreur" not in html
    assert "worst-of S&amp;P 500 / Euro Stoxx 50 / Nasdaq 100" in html
    assert "Heston (volatilité stochastique)" in html
    assert "Corrélation entre sous-jacents" in html and "plus mauvais à maturité" in html


def test_duplicate_underlying_rejected(client, fake_market):
    html = client.get("/", {**PARAMS, "underlying_2": "S&P 500", "sigma_2": "18"}).content.decode()
    assert "déjà sélectionné" in html


def test_missing_vol_rejected(client, fake_market):
    html = client.get("/", {**PARAMS, "underlying_2": "DAX"}).content.decode()
    assert "Volatilité requise" in html


def test_invalid_step_down_is_rejected(client, fake_market):
    # 100 % − 4 × 10 % = 60 % < barrière de capital à 70 %
    html = client.get("/", {**PARAMS, "ph_step_down": "10", "ph_capital": "70"}).content.decode()
    assert "passe sous la barrière de capital" in html


def test_market_data_endpoint(client, fake_market):
    data = client.get("/market-data/", {"index": "S&P 500"}).json()
    assert data["sigma"] == 16.0 and data["r"] == 4.0 and data["term_ratio"] == 1.1
    assert client.get("/market-data/", {"index": "inconnu"}).status_code == 404


def test_correlation_from_weekly_history(monkeypatch):
    import numpy as np
    import pandas as pd

    from autocall_pricer import get_correlation, market_data

    idx = pd.bdate_range("2023-01-02", periods=520)
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, len(idx))
    hist = {
        "^GSPC": pd.Series(100 * np.exp(np.cumsum(common + rng.normal(0, 0.005, len(idx)))), index=idx),
        "^NDX": pd.Series(100 * np.exp(np.cumsum(common + rng.normal(0, 0.005, len(idx)))), index=idx),
    }
    monkeypatch.setattr(market_data, "_history", lambda t, period: hist[t])
    res = get_correlation(["S&P 500", "Nasdaq 100"])
    assert 0.6 < res["correlation"] < 0.95
    assert "hebdomadaire" in res["source"]


def test_correlation_endpoint(client, fake_market):
    data = client.get("/correlation/", {"indices": "S&P 500,CAC 40"}).json()
    assert 30 < data["correlation"] < 95 and "hebdomadaire" in data["source"]
    assert client.get("/correlation/", {"indices": "S&P 500"}).status_code == 404
    assert client.get("/correlation/", {"indices": "S&P 500,S&P 500"}).status_code == 404
