import numpy as np
import pandas as pd
import pytest


def fake_history(ticker, period="2y"):
    """Historiques corrélés reproductibles (facteur commun + bruit propre à chaque indice)."""
    idx = pd.bdate_range("2023-01-02", periods=520)
    common = np.random.default_rng(0).normal(0, 0.01, len(idx))
    own = np.random.default_rng(abs(hash(ticker)) % 2**32).normal(0, 0.006, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(common + own)), index=idx)


@pytest.fixture
def fake_market(monkeypatch):
    """Données de marché simulées : aucun appel réseau."""
    from autocall_pricer import market_data

    quotes = {"^GSPC": 5000.0, "^VIX": 16.0, "^VIX1Y": 17.6, "^FVX": 4.0}
    monkeypatch.setattr(market_data, "_last_close", lambda t, period="5d": quotes.get(t))
    monkeypatch.setattr(market_data, "_historical_vol", lambda t: 0.18)
    monkeypatch.setattr(market_data, "_history", fake_history)


@pytest.fixture(autouse=True)
def memory_cache(settings):
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
