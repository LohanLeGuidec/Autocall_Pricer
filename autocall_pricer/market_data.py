"""
Données de marché indicatives (Yahoo Finance) pour préremplir le pricer.

Ce sont des approximations, affichées comme telles dans l'interface :
· volatilité court terme : indice de volatilité implicite à 30 jours (VIX, VXN, VXD) quand
  il existe, sinon volatilité historique sur 1 an ;
· structure par terme : rapport entre la volatilité implicite à 1 an, 6 mois ou 3 mois
  (VIX1Y, VIX6M, VIX3M) et le VIX à 30 jours. Elle sert à calibrer la variance de long
  terme θ du modèle de Heston ;
· corrélations : rendements hebdomadaires sur 2 ans (les rendements quotidiens
  sous-estiment la corrélation entre places dont les séances ne se recouvrent pas) ;
· taux : Treasury 5 ans (^FVX) pour les indices américains ; pour la zone euro, aucune
  série gratuite fiable sur Yahoo : valeur par défaut à ajuster ;
· dividende : rendement moyen récent de l'indice (estimation fixe).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

INDICES = {
    "S&P 500": {"ticker": "^GSPC", "vol_ticker": "^VIX", "term_tickers": ("^VIX1Y", "^VIX6M", "^VIX3M"),
                "rate_ticker": "^FVX", "dividend": 0.012, "currency": "USD"},
    "Nasdaq 100": {"ticker": "^NDX", "vol_ticker": "^VXN", "term_tickers": (), "rate_ticker": "^FVX",
                   "dividend": 0.007, "currency": "USD"},
    "Dow Jones": {"ticker": "^DJI", "vol_ticker": "^VXD", "term_tickers": (), "rate_ticker": "^FVX",
                  "dividend": 0.017, "currency": "USD"},
    "Euro Stoxx 50": {"ticker": "^STOXX50E", "vol_ticker": None, "term_tickers": (), "rate_ticker": None,
                      "dividend": 0.032, "currency": "EUR"},
    "CAC 40": {"ticker": "^FCHI", "vol_ticker": None, "term_tickers": (), "rate_ticker": None,
               "dividend": 0.030, "currency": "EUR"},
    "DAX": {"ticker": "^GDAXI", "vol_ticker": None, "term_tickers": (), "rate_ticker": None,
            "dividend": 0.0, "currency": "EUR"},  # indice de rentabilité : dividendes réinvestis
}

DEFAULTS = {"spot": 100.0, "sigma": 0.20, "r": 0.025, "correlation": 0.60, "term_ratio": 1.0}

# Échéance (en années) de chaque indice de volatilité de la structure par terme
TERM_TENORS = {"^VIX1Y": 1.0, "^VIX6M": 0.5, "^VIX3M": 0.25}


def _last_close(ticker: str, period: str = "5d") -> float | None:
    try:
        import yfinance as yf

        close = yf.Ticker(ticker).history(period=period)["Close"].dropna()
        return float(close.iloc[-1]) if not close.empty else None
    except Exception as exc:  # noqa: BLE001 — réseau, ticker indisponible…
        logger.warning("Yahoo Finance indisponible pour %s : %s", ticker, exc)
        return None


def _history(ticker: str, period: str):
    try:
        import yfinance as yf

        close = yf.Ticker(ticker).history(period=period)["Close"].dropna()
        return close if len(close) > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Historique indisponible pour %s : %s", ticker, exc)
        return None


def _historical_vol(ticker: str) -> float | None:
    close = _history(ticker, "1y")
    if close is None or len(close) < 60:
        return None
    return float(np.log(close).diff().dropna().std() * np.sqrt(252))


def get_market_data(index_name: str) -> dict:
    """
    Spot, volatilité, structure par terme, taux et dividende, avec pour chacun sa source
    (« live », « historique », « estimation » ou « défaut »).
    """
    info = INDICES[index_name]
    out, sources = {}, {}

    spot = _last_close(info["ticker"])
    out["spot"], sources["spot"] = (spot, "live") if spot else (DEFAULTS["spot"], "défaut")

    sigma = None
    if info["vol_ticker"]:
        v = _last_close(info["vol_ticker"])
        if v:
            sigma, sources["sigma"] = v / 100, f"implicite 30 j ({info['vol_ticker']})"
    if sigma is None:
        sigma = _historical_vol(info["ticker"])
        if sigma:
            sources["sigma"] = "historique 1 an"
    if sigma is None:
        sigma, sources["sigma"] = DEFAULTS["sigma"], "défaut"
    out["sigma"] = sigma

    # Structure par terme : vol implicite plus longue / vol 30 jours
    out["term_ratio"], sources["term_ratio"] = DEFAULTS["term_ratio"], "plate (pas de données)"
    if sources["sigma"].startswith("implicite"):
        for t in info["term_tickers"]:
            long_vol = _last_close(t)
            if long_vol:
                out["term_ratio"] = round(long_vol / 100 / sigma, 3)
                sources["term_ratio"] = f"{t} / {info['vol_ticker']}"
                break

    rate = _last_close(info["rate_ticker"]) if info["rate_ticker"] else None
    if rate:
        out["r"], sources["r"] = rate / 100, f"Treasury 5 ans ({info['rate_ticker']})"
    else:
        out["r"], sources["r"] = DEFAULTS["r"], "défaut (à ajuster)"

    out["q"], sources["q"] = info["dividend"], "estimation"
    out["currency"] = info["currency"]
    out["sources"] = sources
    out["index"] = index_name
    return out


def get_correlation(index_names: list[str]) -> dict:
    """
    Corrélation moyenne entre les sous-jacents, estimée sur 2 ans de rendements
    hebdomadaires. Renvoie aussi la matrice complète quand elle est disponible.
    """
    if len(index_names) < 2:
        return {"correlation": 1.0, "matrix": None, "source": "un seul sous-jacent"}
    series = {}
    for name in index_names:
        h = _history(INDICES[name]["ticker"], "2y")
        if h is None:
            return {"correlation": DEFAULTS["correlation"], "matrix": None, "source": "défaut (données indisponibles)"}
        h.index = h.index.tz_localize(None) if getattr(h.index, "tz", None) is not None else h.index
        series[name] = h

    import pandas as pd

    weekly = pd.DataFrame(series).resample("W-FRI").last().dropna()
    rets = np.log(weekly).diff().dropna()
    if len(rets) < 30:
        return {"correlation": DEFAULTS["correlation"], "matrix": None, "source": "défaut (historique trop court)"}
    corr = rets.corr()
    off_diag = corr.values[~np.eye(len(corr), dtype=bool)]
    return {"correlation": float(off_diag.mean()), "matrix": corr.round(2).to_dict(),
            "source": "historique 2 ans (hebdomadaire)"}
