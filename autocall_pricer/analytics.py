"""
Grecques, coupon d'équilibre, sensibilités et smile implicite.

Tous les calculs réutilisent la même graine (« common random numbers ») : seule la
variation du paramètre étudié fait bouger le prix, pas le bruit Monte Carlo.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .products import PRICERS
from .simulation import GBM, ModelSpec, simulate_assets, worst_of


@dataclass(frozen=True)
class Market:
    """
    Paramètres de marché. `sigma` et `q` peuvent être des nombres (un sous-jacent)
    ou des tuples (un élément par sous-jacent, pour un worst-of).
    """
    r: float
    q: float | tuple = 0.0
    sigma: float | tuple = 0.2
    credit_spread: float = 0.0
    correlation: float = 0.0
    model: ModelSpec = field(default=GBM)

    @property
    def sigmas(self) -> tuple:
        return tuple(np.atleast_1d(self.sigma).astype(float))

    @property
    def qs(self) -> tuple:
        return tuple(np.broadcast_to(np.atleast_1d(self.q), (len(self.sigmas),)).astype(float))

    @property
    def n_assets(self) -> int:
        return len(self.sigmas)


@dataclass(frozen=True)
class SimSettings:
    maturity: int
    n_sim: int = 20_000
    seed: int = 42
    antithetic: bool = True


def simulate(market: Market, sim: SimSettings, vol_shift: float = 0.0) -> np.ndarray:
    """Trajectoires de tous les sous-jacents (n_sim, T + 1, n_assets)."""
    return simulate_assets(sim.n_sim, sim.maturity, market.sigmas, market.qs, market.r, market.correlation,
                           market.model, sim.seed, sim.antithetic, vol_shift)


def price(product: str, params, market: Market, perf: np.ndarray, antithetic: bool = True) -> float:
    """Fair value (% du nominal) sur une performance worst-of déjà simulée."""
    return PRICERS[product](perf, params, market.r, market.credit_spread, antithetic)["fair_value_pct"]


def fair_value(product: str, params, market: Market, sim: SimSettings, vol_shift: float = 0.0) -> float:
    return price(product, params, market, worst_of(simulate(market, sim, vol_shift)), sim.antithetic)


def greeks(product: str, params, market: Market, sim: SimSettings, perf: np.ndarray | None = None) -> dict:
    """
    Différences finies centrées, en points de % du nominal :
    · delta : spot de tous les sous-jacents +1 % ;   · gamma : variation du delta ;
    · vega  : volatilité de tous les sous-jacents +1 pt ;  · rho : taux +1 pt ;
    · spread : spread de crédit +10 pb.
    GBM et Heston étant homogènes en S, un choc de spot multiplie simplement les
    trajectoires : pas besoin de resimuler.
    """
    if perf is None:
        perf = worst_of(simulate(market, sim))
    a = sim.antithetic

    def p(m=market, x=perf):
        return price(product, params, m, x, a)

    base = p()
    up, down = p(x=perf * 1.01), p(x=perf * 0.99)
    vol_up = price(product, params, market, worst_of(simulate(market, sim, +0.01)), a)
    vol_down = price(product, params, market, worst_of(simulate(market, sim, -0.01)), a)
    m_up, m_down = replace(market, r=market.r + 0.001), replace(market, r=market.r - 0.001)
    r_up = price(product, params, m_up, worst_of(simulate(m_up, sim)), a)
    r_down = price(product, params, m_down, worst_of(simulate(m_down, sim)), a)
    cs_up = p(m=replace(market, credit_spread=market.credit_spread + 0.001))
    return {
        "fair_value": base,
        "delta": (up - down) / 2,
        "gamma": up - 2 * base + down,
        "vega": (vol_up - vol_down) / 2,
        "rho": (r_up - r_down) / 2 * 10,
        "spread_10bp": cs_up - base,
    }


def fair_coupon(product: str, params, market: Market, perf: np.ndarray, target_pct: float = 100.0,
                antithetic: bool = True) -> float:
    """
    Coupon qui donne une fair value égale à `target_pct` (100 − marge).
    Sur des trajectoires données, la valeur des deux produits est affine en coupon :
    deux évaluations suffisent et la solution est exacte.
    """
    fv0 = price(product, replace(params, coupon_rate=0.0), market, perf, antithetic)
    fv1 = price(product, replace(params, coupon_rate=0.10), market, perf, antithetic)
    if np.isclose(fv1, fv0):
        return float("nan")
    return float((target_pct - fv0) / (fv1 - fv0) * 0.10)


SENSITIVITY_GRIDS = {
    "vol_shift": np.round(np.arange(-0.10, 0.1601, 0.03), 4),
    "r": np.round(np.arange(0.0, 0.0601, 0.01), 4),
    "barrier_capital": np.round(np.arange(0.40, 0.801, 0.025), 4),
    "credit_spread": np.round(np.arange(0.0, 0.0301, 0.0025), 4),
    "correlation": np.round(np.arange(0.0, 0.951, 0.15), 4),
}


def sensitivities(params_by_product: dict, market: Market, sim: SimSettings, perf: np.ndarray) -> dict:
    """
    Fair value de chaque produit en fonction de : volatilité (choc parallèle), taux,
    barrière de capital, spread de crédit et, pour un worst-of, corrélation.
    Les scénarios ne sont resimulés que si le paramètre modifie la dynamique.
    """
    a = sim.antithetic
    out = {}
    # Les points qui imposent de resimuler utilisent deux fois moins de trajectoires avec
    # Heston (même graine : les courbes restent lisses) pour garder un temps de calcul raisonnable.
    sim_r = replace(sim, n_sim=max(2000, sim.n_sim // 2)) if market.model.is_heston else sim

    def run(what, grid, perf_for, market_for, params_for):
        res = {k: [] for k in params_by_product}
        for x in grid:
            pf, mk = perf_for(x), market_for(x)
            for k, prm in params_by_product.items():
                res[k].append(price(k, params_for(prm, x), mk, pf, a))
        out[what] = {k: (np.asarray(grid), np.asarray(v)) for k, v in res.items()}

    min_sigma = min(market.sigmas)
    vol_grid = [x for x in SENSITIVITY_GRIDS["vol_shift"] if min_sigma + x >= 0.03]
    run("sigma", vol_grid, lambda x: worst_of(simulate(market, sim_r, x)), lambda x: market, lambda p, x: p)
    run("r", SENSITIVITY_GRIDS["r"], lambda x: worst_of(simulate(replace(market, r=x), sim_r)),
        lambda x: replace(market, r=x), lambda p, x: p)
    run("barrier_capital", SENSITIVITY_GRIDS["barrier_capital"], lambda x: perf, lambda x: market,
        lambda p, x: replace(p, barrier_capital=float(x)))
    run("credit_spread", SENSITIVITY_GRIDS["credit_spread"], lambda x: perf,
        lambda x: replace(market, credit_spread=x), lambda p, x: p)
    if market.n_assets > 1:
        run("correlation", SENSITIVITY_GRIDS["correlation"],
            lambda x: worst_of(simulate(replace(market, correlation=x), sim_r)), lambda x: market, lambda p, x: p)
    # Axe de la volatilité exprimé en niveau moyen plutôt qu'en choc
    mean_sigma = float(np.mean(market.sigmas))
    out["sigma"] = {k: (grid + mean_sigma, vals) for k, (grid, vals) in out["sigma"].items()}
    return out


# ── Smile implicite ──────────────────────────────────────────────────────────

def bs_price(spot, strike, T, r, q, sigma, call: bool) -> float:
    if sigma <= 0 or T <= 0:
        return max(0.0, (spot - strike) if call else (strike - spot))
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return spot * np.exp(-q * T) * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    return strike * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price_, spot, strike, T, r, q, call: bool) -> float:
    try:
        return brentq(lambda s: bs_price(spot, strike, T, r, q, s, call) - price_, 1e-4, 3.0)
    except ValueError:
        return float("nan")


def _match_forward(terminal: np.ndarray, forward: float) -> np.ndarray:
    """
    Recale les niveaux simulés pour que leur moyenne soit exactement le forward
    (variable de contrôle). Sans cela, l'erreur Monte Carlo sur E[S_T] crée un
    décrochage du smile au passage des puts aux calls.
    """
    return terminal * forward / terminal.mean()


def atm_implied_vol(terminal: np.ndarray, T: float, r: float, q: float) -> float:
    """Volatilité implicite du call à la monnaie forward, à partir de niveaux simulés."""
    forward = np.exp((r - q) * T)
    terminal = _match_forward(terminal, forward)
    call = np.exp(-r * T) * np.maximum(terminal - forward, 0).mean()
    return implied_vol(call, 1.0, forward, T, r, q, True)


SMILE_STRIKES = np.round(np.arange(0.5, 1.31, 0.05), 2)


def implied_smile(terminal: np.ndarray, T: float, r: float, q: float,
                  strikes=SMILE_STRIKES) -> tuple[np.ndarray, np.ndarray]:
    """
    Volatilités implicites Black-Scholes des options européennes de maturité T, prix
    calculés sur les niveaux simulés (options hors de la monnaie : puts sous le forward,
    calls au-dessus).
    """
    forward = np.exp((r - q) * T)
    terminal = _match_forward(terminal, forward)
    vols = []
    for k in strikes:
        call = k >= forward
        payoff = np.maximum(terminal - k, 0) if call else np.maximum(k - terminal, 0)
        vols.append(implied_vol(np.exp(-r * T) * payoff.mean(), 1.0, k, T, r, q, call))
    return np.asarray(strikes), np.asarray(vols)
