"""
Calibration du modèle de Heston sur des options vanilles cotées.

Pourquoi : dans le pricer, κ, ξ et ρ sont des valeurs typiques d'indices actions. Ici on
les choisit pour que le Heston reproduise au mieux les prix d'options observés sur le
marché (la surface de volatilité implicite).

1. Prix d'un call européen sous Heston par formule semi-fermée : transformée de Fourier
   de la fonction caractéristique de ln S_T, dans la forme « little Heston trap »
   (Albrecher et al., 2007), numériquement stable pour les longues maturités. Le Monte
   Carlo serait trop lent : la calibration appelle cette formule des milliers de fois.
2. Calibration par moindres carrés (scipy.optimize.least_squares). Les écarts de prix
   sont divisés par le vega Black de l'option : on minimise ainsi, au premier ordre,
   l'écart en volatilité implicite, ce qui donne le même poids à toute la surface.
3. Les paramètres calibrés sont convertis dans la paramétrisation du pricer :
   σ = √v₀ et « vol long terme / court terme » = √(θ / v₀), avec κ, ξ, ρ inchangés.
   Les bornes de la calibration sont celles du formulaire, pour que le résultat puisse
   y être saisi tel quel.

Tout est écrit en termes de forward F = S·e^{(r−q)T} : le dividende n'a pas à être estimé
séparément, il est contenu dans le forward implicite tiré de la parité call-put.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from .simulation import ModelSpec

# Bornes alignées sur le formulaire (pricer/forms.py)
BOUNDS = {
    "sigma0": (0.03, 0.80),      # √v₀
    "term_ratio": (0.5, 2.0),    # √θ / √v₀
    "kappa": (0.05, 10.0),
    "xi": (0.01, 2.0),
    "rho": (-0.99, 0.99),
}
PARAM_ORDER = ("sigma0", "term_ratio", "kappa", "xi", "rho")

# Quadrature de Gauss-Legendre sur [0, U_MAX] pour l'intégrale de Fourier
U_MAX = 200.0
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)
_U = 0.5 * U_MAX * (_NODES + 1.0) + 1e-10
_W = 0.5 * U_MAX * _WEIGHTS


@dataclass(frozen=True)
class HestonParams:
    v0: float
    theta: float
    kappa: float
    xi: float
    rho: float

    @property
    def sigma0(self) -> float:
        return float(np.sqrt(self.v0))

    @property
    def term_ratio(self) -> float:
        return float(np.sqrt(self.theta / self.v0))

    @property
    def feller(self) -> bool:
        """2κθ ≥ ξ² : la variance ne touche jamais zéro."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def to_model_spec(self, steps_per_year: int = 26) -> ModelSpec:
        """Paramètres directement utilisables par le pricer (avec Market(sigma=self.sigma0))."""
        return ModelSpec("heston", kappa=self.kappa, xi=self.xi, rho=self.rho,
                         term_ratio=self.term_ratio, steps_per_year=steps_per_year)


# ── Black (forward) ──────────────────────────────────────────────────────────

def black_price(forward, strike, T, r, sigma, call=True):
    """Prix d'une option européenne en fonction du forward (Black-76), vectorisé."""
    forward, strike, T, sigma = np.broadcast_arrays(*map(np.asarray, (forward, strike, T, sigma)))
    sq = sigma * np.sqrt(T)
    d1 = (np.log(forward / strike) + 0.5 * sq**2) / sq
    d2 = d1 - sq
    df = np.exp(-r * T)
    c = df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    p = df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return np.where(call, c, p)


def black_vega(forward, strike, T, r, sigma):
    sq = sigma * np.sqrt(T)
    d1 = (np.log(forward / strike) + 0.5 * sq**2) / sq
    return np.exp(-r * T) * forward * norm.pdf(d1) * np.sqrt(T)


def black_implied_vol(price, forward, strike, T, r, call=True) -> float:
    """Volatilité implicite Black d'une option (nan si le prix sort des bornes d'arbitrage)."""
    try:
        return brentq(lambda s: float(black_price(forward, strike, T, r, s, call)) - price, 1e-4, 4.0)
    except ValueError:
        return float("nan")


# ── Heston semi-fermé ────────────────────────────────────────────────────────

def _log_cf(u, T, v0, kappa, theta, xi, rho):
    """ln de la fonction caractéristique de ln(S_T / F), forme « little trap ». u : (n_u, 1), T : (1, n_opt)."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    return C + D * v0


def heston_price(forward, strike, T, r, params: HestonParams, call=True):
    """
    Prix d'options européennes sous Heston (vectorisé sur les options).
    C = e^{−rT} [F·P₁ − K·P₂],  P_j = ½ + (1/π) ∫₀^∞ Re[e^{−iu ln(K/F)} φ_j(u) / (iu)] du
    """
    forward, strike, T, call = np.broadcast_arrays(*map(np.asarray, (forward, strike, T, call)))
    shape = forward.shape
    F, K, TT = forward.ravel()[None, :], strike.ravel()[None, :], T.ravel()[None, :]
    u = _U[:, None]
    p = (params.v0, params.kappa, params.theta, params.xi, params.rho)
    k = np.log(K / F)
    phi2 = np.exp(_log_cf(u, TT, *p))
    phi1 = np.exp(_log_cf(u - 1j, TT, *p))          # φ(u − i) / φ(−i), et φ(−i) = 1 en log-forward
    kernel = np.exp(-1j * u * k) / (1j * u)
    P1 = 0.5 + (_W[:, None] * (kernel * phi1).real).sum(axis=0) / np.pi
    P2 = 0.5 + (_W[:, None] * (kernel * phi2).real).sum(axis=0) / np.pi
    df = np.exp(-r * TT)
    c = df * (F * P1 - K * P2)
    put = c - df * (F - K)                          # parité call-put
    out = np.where(call.ravel()[None, :], c, put).ravel()
    return np.maximum(out, 0.0).reshape(shape)


def heston_implied_vols(forward, strike, T, r, params: HestonParams) -> np.ndarray:
    """Smile implicite Black produit par le Heston (options hors de la monnaie)."""
    forward, strike, T = np.broadcast_arrays(*map(np.asarray, (forward, strike, T)))
    call = strike >= forward
    prices = heston_price(forward, strike, T, r, params, call)
    return np.array([black_implied_vol(p, f, k, t, r, c)
                     for p, f, k, t, c in zip(prices.ravel(), forward.ravel(), strike.ravel(), T.ravel(),
                                              call.ravel())]).reshape(forward.shape)


# ── Surface de marché et calibration ─────────────────────────────────────────

@dataclass
class OptionQuotes:
    """Options hors de la monnaie cotées : une ligne par option (strike, maturité, forward, vol implicite)."""
    strike: np.ndarray
    T: np.ndarray
    forward: np.ndarray
    vol: np.ndarray
    r: float
    source: str = ""
    spot: float | None = None

    def __post_init__(self):
        self.strike, self.T, self.forward, self.vol = (np.asarray(x, dtype=float)
                                                       for x in (self.strike, self.T, self.forward, self.vol))
        ok = np.isfinite(self.vol) & (self.vol > 0) & (self.T > 0)
        self.strike, self.T, self.forward, self.vol = (x[ok] for x in (self.strike, self.T, self.forward, self.vol))

    @property
    def call(self) -> np.ndarray:
        return self.strike >= self.forward

    @property
    def price(self) -> np.ndarray:
        return black_price(self.forward, self.strike, self.T, self.r, self.vol, self.call)

    @property
    def vega(self) -> np.ndarray:
        return black_vega(self.forward, self.strike, self.T, self.r, self.vol)

    def __len__(self) -> int:
        return len(self.strike)


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_vol: float                 # erreur quadratique moyenne en points de volatilité (décimal)
    max_error_vol: float
    n_options: int
    quotes: OptionQuotes
    model_vols: np.ndarray
    success: bool
    message: str = ""
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        p = self.params
        return {
            "sigma0": round(float(p.sigma0), 4), "term_ratio": round(float(p.term_ratio), 3),
            "kappa": round(float(p.kappa), 3), "xi": round(float(p.xi), 3), "rho": round(float(p.rho), 3),
            "v0": round(float(p.v0), 5), "theta": round(float(p.theta), 5), "feller": bool(p.feller),
            "rmse_vol_pts": round(self.rmse_vol * 100, 2), "max_error_vol_pts": round(self.max_error_vol * 100, 2),
            "n_options": self.n_options, "source": self.quotes.source,
        }


def _from_vector(x) -> HestonParams:
    sigma0, tr, kappa, xi, rho = x
    v0 = sigma0**2
    return HestonParams(v0=v0, theta=v0 * tr**2, kappa=kappa, xi=xi, rho=rho)


def calibrate_heston(quotes: OptionQuotes, x0: HestonParams | None = None) -> CalibrationResult:
    """Moindres carrés pondérés par 1/vega : minimise l'écart de volatilité implicite."""
    if len(quotes) < 6:
        raise ValueError("Pas assez d'options pour calibrer 5 paramètres (6 au minimum).")
    market_price = quotes.price
    vega = np.maximum(quotes.vega, 1e-4 * quotes.forward * np.sqrt(quotes.T))
    call = quotes.call

    def residuals(x):
        model = heston_price(quotes.forward, quotes.strike, quotes.T, quotes.r, _from_vector(x), call)
        return (model - market_price) / vega

    if x0 is None:
        atm = np.argmin(np.abs(np.log(quotes.strike / quotes.forward)) + 0.1 * quotes.T)
        s0 = float(np.clip(quotes.vol[atm], *BOUNDS["sigma0"]))
        x0 = HestonParams(v0=s0**2, theta=s0**2, kappa=1.5, xi=0.6, rho=-0.6)
    start = np.array([np.clip(getattr(x0, n), *BOUNDS[n]) for n in PARAM_ORDER])
    lo, hi = np.array([BOUNDS[n] for n in PARAM_ORDER]).T
    fit = least_squares(residuals, start, bounds=(lo, hi), x_scale="jac", max_nfev=400)

    params = _from_vector(fit.x)
    model_vols = heston_implied_vols(quotes.forward, quotes.strike, quotes.T, quotes.r, params)
    err = model_vols - quotes.vol
    err = err[np.isfinite(err)]
    return CalibrationResult(
        params=params, rmse_vol=float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
        max_error_vol=float(np.max(np.abs(err))) if len(err) else float("nan"), n_options=len(quotes),
        quotes=quotes, model_vols=model_vols, success=bool(fit.success), message=fit.message,
    )


# ── Données ──────────────────────────────────────────────────────────────────

def quotes_from_table(rows, r: float, spot: float | None = None, source: str = "fichier") -> OptionQuotes:
    """
    Surface saisie ou exportée (par exemple de Bloomberg OVDV) : lignes
    (maturité en années, strike en % du spot, vol implicite en %, forward en % du spot facultatif).
    Sans forward, on prend le spot capitalisé au taux r (dividende ignoré).
    """
    T, k, v, f = [], [], [], []
    for row in rows:
        t, strike_pct, vol_pct = float(row[0]), float(row[1]), float(row[2])
        fwd_pct = float(row[3]) if len(row) > 3 and row[3] not in (None, "") else 100 * np.exp(r * t)
        T.append(t), k.append(strike_pct / 100), v.append(vol_pct / 100), f.append(fwd_pct / 100)
    return OptionQuotes(strike=k, T=T, forward=f, vol=v, r=r, source=source, spot=spot)


def load_surface_csv(path: str, r: float, spot: float | None = None) -> OptionQuotes:
    """CSV avec en-tête : maturity_years, strike_pct, implied_vol_pct[, forward_pct]."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=";" if ";" in fh.readline() else ",")
        fh.seek(0)
        next(reader)
        rows = [[c.replace(",", ".") for c in row] for row in reader if row and row[0].strip()]
    return quotes_from_table(rows, r, spot, source=path)


# Tickers Yahoo des options sur indice (seuls les indices américains ont une chaîne d'options gratuite)
OPTION_TICKERS = {"S&P 500": "^SPX", "Nasdaq 100": "^NDX"}


def fetch_option_quotes(index_name: str, r: float, min_days: int = 30, max_years: float = 3.0,
                        n_expiries: int = 8, moneyness=(0.75, 1.25), max_rel_spread: float = 0.3) -> OptionQuotes:
    """
    Options hors de la monnaie sur l'indice (Yahoo Finance), filtrées sur la liquidité.
    Le forward de chaque maturité est tiré de la parité call-put au strike le plus proche du spot :
    F = K + e^{rT} (C − P). Les volatilités implicites sont recalculées à partir des prix milieux.
    """
    import datetime as dt

    if index_name not in OPTION_TICKERS:
        raise ValueError(f"Pas de chaîne d'options gratuite pour {index_name} : importez une surface (CSV).")
    import yfinance as yf

    tk = yf.Ticker(OPTION_TICKERS[index_name])
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    today = dt.date.today()
    expiries = []
    for e in tk.options:
        T = (dt.date.fromisoformat(e) - today).days / 365.0
        if min_days / 365 <= T <= max_years:
            expiries.append((e, T))
    if not expiries:
        raise ValueError("Aucune maturité exploitable.")
    pick = np.unique(np.linspace(0, len(expiries) - 1, min(n_expiries, len(expiries))).round().astype(int))

    K, TT, F, V = [], [], [], []
    for i in pick:
        e, T = expiries[i]
        chain = tk.option_chain(e)
        calls = chain.calls.set_index("strike")
        puts = chain.puts.set_index("strike")
        both = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner")
        if both.empty:
            continue
        # Prix milieu quand la fourchette est cotée, sinon dernier prix (marché fermé)
        quoted_c = (both.bid_c > 0) & (both.ask_c > 0)
        quoted_p = (both.bid_p > 0) & (both.ask_p > 0)
        mid_c = ((both.bid_c + both.ask_c) / 2).where(quoted_c, both.lastPrice_c)
        mid_p = ((both.bid_p + both.ask_p) / 2).where(quoted_p, both.lastPrice_p)
        valid = (mid_c > 0) & (mid_p > 0)
        if not valid.any():
            continue
        strikes = both.index.to_series()[valid]
        atm = (strikes - spot).abs().idxmin()
        forward = atm + np.exp(r * T) * (mid_c[atm] - mid_p[atm])
        for strike in strikes.index:
            m = strike / forward
            if not moneyness[0] <= m <= moneyness[1]:
                continue
            is_call = strike >= forward
            row = both.loc[strike]
            bid, ask = (row.bid_c, row.ask_c) if is_call else (row.bid_p, row.ask_p)
            mid = mid_c[strike] if is_call else mid_p[strike]
            if bid > 0 and ask > 0 and (ask - bid) / mid > max_rel_spread:
                continue
            vol = black_implied_vol(mid, forward, strike, T, r, is_call)
            if np.isfinite(vol):
                K.append(strike / spot), TT.append(T), F.append(forward / spot), V.append(vol)
    source = f"options {OPTION_TICKERS[index_name]} (Yahoo Finance, {today.isoformat()})"
    return OptionQuotes(strike=K, T=TT, forward=F, vol=V, r=r, source=source, spot=spot)
