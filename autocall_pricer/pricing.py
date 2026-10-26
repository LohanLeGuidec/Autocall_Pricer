"""Calcul complet pour l'interface : prix, intervalles, grecques, coupons d'équilibre, sensibilités, skew."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .analytics import (
    Market,
    SimSettings,
    atm_implied_vol,
    fair_coupon,
    greeks,
    implied_smile,
    price,
    sensitivities,
    simulate,
)
from .products import AthenaParams, PhoenixParams, price_athena, price_phoenix
from .simulation import GBM, ModelSpec, worst_of

N_DISPLAY_PATHS = 100


def run_pricing(phoenix: PhoenixParams, athena: AthenaParams, market: Market, n_sim: int = 20_000,
                seed: int = 42, antithetic: bool = True, target_pct: float = 100.0,
                with_sensitivities: bool = True, heston_for_comparison: ModelSpec | None = None) -> dict:
    """
    `heston_for_comparison` : paramètres Heston utilisés pour mesurer l'impact du skew
    quand le modèle principal est le GBM (et inversement, le GBM sert de référence
    quand le modèle principal est Heston).
    """
    if phoenix.maturity != athena.maturity:
        raise ValueError("Les deux produits doivent avoir la même maturité.")
    T = phoenix.maturity
    sim = SimSettings(T, n_sim, seed, antithetic)

    assets = simulate(market, sim)
    perf = worst_of(assets)
    params = {"phoenix": phoenix, "athena": athena}

    res = {
        "phoenix": price_phoenix(perf, phoenix, market.r, market.credit_spread, antithetic),
        "athena": price_athena(perf, athena, market.r, market.credit_spread, antithetic),
    }

    # ── Impact du skew ──
    # Heston et GBM ne sont comparables qu'à volatilité à la monnaie égale : sinon on
    # mesure surtout l'écart de niveau de vol (le vega de l'autocall est fort), pas le skew.
    # On calcule donc la vol implicite ATM produite par le Heston, puis un GBM à cette vol.
    if market.model.is_heston:
        heston_market, heston_assets, heston_perf = market, assets, perf
        gbm_input_market = replace(market, model=GBM)
        gbm_input_perf = worst_of(simulate(gbm_input_market, sim))
    else:
        heston_market = replace(market, model=heston_for_comparison or ModelSpec("heston"))
        heston_assets = simulate(heston_market, sim)
        heston_perf = worst_of(heston_assets)
        gbm_input_market, gbm_input_perf = market, perf
    atm_vols = tuple(atm_implied_vol(heston_assets[:, -1, i], T, market.r, q) for i, q in enumerate(market.qs))
    gbm_atm_market = replace(market, model=GBM, sigma=atm_vols)
    gbm_atm_perf = worst_of(simulate(gbm_atm_market, sim))
    model_comparison = {}
    for k in params:
        heston_fv = price(k, params[k], heston_market, heston_perf, antithetic)
        gbm_atm_fv = price(k, params[k], gbm_atm_market, gbm_atm_perf, antithetic)
        model_comparison[k] = {
            "gbm_input": price(k, params[k], gbm_input_market, gbm_input_perf, antithetic),
            "gbm_atm": gbm_atm_fv,
            "heston": heston_fv,
            "skew_effect": heston_fv - gbm_atm_fv,
        }
    strikes, heston_vols = implied_smile(heston_assets[:, -1, 0], T, market.r, market.qs[0])

    # Quel sous-jacent est le plus mauvais à maturité ?
    worst_share = (np.bincount(assets[:, -1, :].argmin(axis=1), minlength=market.n_assets) / n_sim * 100
                   if market.n_assets > 1 else np.array([100.0]))

    out = {
        "results": res,
        "greeks": {k: greeks(k, params[k], market, sim, perf) for k in params},
        "fair_coupons": {k: fair_coupon(k, params[k], market, perf, target_pct, antithetic) for k in params},
        "model_comparison": model_comparison,
        "smile": {"strikes": strikes, "heston": heston_vols, "gbm": market.sigmas[0],
                  "atm_vols": atm_vols, "heston_spec": heston_market.model},
        "worst_share": worst_share,
        "paths_sample": perf[:N_DISPLAY_PATHS],
        "paths_quantiles": {
            "p05": np.percentile(perf, 5, axis=0),
            "median": np.median(perf, axis=0),
            "p95": np.percentile(perf, 95, axis=0),
        },
        "forward_check": [  # contrôle de martingale par sous-jacent : E[S_T] = e^{(r−q)T}
            {"simulated": float(assets[:, -1, i].mean()), "theoretical": float(np.exp((market.r - q) * T))}
            for i, q in enumerate(market.qs)
        ],
        "params": params,
        "market": market,
        "n_sim": n_sim,
        "antithetic": antithetic,
        "target_pct": target_pct,
    }
    if with_sensitivities:
        out["sensitivities"] = sensitivities(params, market, sim, perf)
    return out
