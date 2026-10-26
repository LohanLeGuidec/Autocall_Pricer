"""
Payoffs des autocalls Phoenix Memory et Athena.

Conventions (trajectoires normalisées : S(0) = 1, barrières en % du niveau initial,
une observation par an, t = 1 … T) :

Phoenix Memory
    À chaque date t, tant que le produit vit :
    · coupon : si S(t) ≥ barrière coupon, versé immédiatement à la date t ;
      avec l'effet mémoire, les coupons des dates précédentes non versées sont
      rattrapés (versés en même temps) ;
    · rappel (t < T) : si S(t) ≥ barrière de rappel(t), remboursement du nominal à t.
    À maturité T : coupon selon la même règle, puis remboursement
    · nominal si S(T) ≥ barrière capital, · nominal × S(T) sinon.
    Les coupons déjà versés restent acquis à l'investisseur, même en cas de perte.

Athena
    · rappel (t < T) : si S(t) ≥ barrière de rappel(t) → nominal × (1 + coupon × t) ;
    · maturité : si S(T) ≥ barrière finale (par défaut la barrière de rappel en T)
      → nominal × (1 + coupon × T) ; sinon si S(T) ≥ barrière capital → nominal ;
      sinon → nominal × S(T).

Actualisation : taux sans risque + spread de crédit de l'émetteur (un autocall est
une obligation de la banque émettrice).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhoenixParams:
    nominal: float = 1000.0
    coupon_rate: float = 0.08        # coupon par date d'observation (annuel)
    barrier_recall: float = 1.00     # barrière de rappel initiale
    barrier_coupon: float = 0.80     # barrière de coupon
    barrier_capital: float = 0.60    # barrière de protection du capital (à maturité)
    maturity: int = 5                # années
    step_down: float = 0.0           # baisse annuelle de la barrière de rappel
    memory: bool = True              # effet mémoire sur les coupons

    def recall_barrier_at(self, t: int) -> float:
        return self.barrier_recall - self.step_down * (t - 1)


@dataclass
class AthenaParams:
    nominal: float = 1000.0
    coupon_rate: float = 0.10        # coupon par année écoulée, versé au rappel
    barrier_recall: float = 1.00
    barrier_capital: float = 0.60
    maturity: int = 5
    step_down: float = 0.0
    barrier_final: float | None = None  # barrière du coupon à maturité (défaut : rappel en T)

    def recall_barrier_at(self, t: int) -> float:
        return self.barrier_recall - self.step_down * (t - 1)

    @property
    def final_barrier(self) -> float:
        return self.recall_barrier_at(self.maturity) if self.barrier_final is None else self.barrier_final


def _discount(t, r: float, credit_spread: float):
    return np.exp(-(r + credit_spread) * t)


def _summary(product: str, pv: np.ndarray, durations: np.ndarray, recall_year: np.ndarray,
             loss: np.ndarray, nominal: float, maturity: int, antithetic: bool, extra: dict) -> dict:
    n = len(pv)
    fv = pv.mean()
    # Avec des variables antithétiques, les paires (Z, −Z) ne sont pas indépendantes :
    # l'erreur standard se calcule sur les moyennes de paires.
    if n < 2:
        se = float("nan")
    elif antithetic and n % 2 == 0:
        half = n // 2
        pair_means = 0.5 * (pv[:half] + pv[half:])
        se = pair_means.std(ddof=1) / np.sqrt(half)
    else:
        se = pv.std(ddof=1) / np.sqrt(n)
    return {
        "product": product,
        "fair_value_pct": float(fv / nominal * 100),
        "fair_value_eur": float(fv),
        "ci95_pct": float(1.96 * se / nominal * 100),
        "recall_pct": float(np.mean(recall_year > 0) * 100),
        "early_recall_pct": float(np.mean((recall_year > 0) & (recall_year < maturity)) * 100),
        "loss_pct": float(loss.mean() * 100),
        "avg_duration": float(durations.mean()),
        "recall_by_year": {t: float(np.mean(recall_year == t) * 100) for t in range(1, maturity + 1)},
        "payoffs_pv": pv,
        "durations": durations,
        "loss_flags": loss,
        **extra,
    }


def price_phoenix(paths: np.ndarray, p: PhoenixParams, r: float, credit_spread: float = 0.0,
                  antithetic: bool = False) -> dict:
    n, T, N, c = paths.shape[0], p.maturity, p.nominal, p.coupon_rate
    pv = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    unpaid = np.zeros(n)              # coupons dus mais pas encore versés (mémoire)
    n_coupons = np.zeros(n)
    durations = np.full(n, float(T))
    recall_year = np.zeros(n, dtype=int)
    coupon_prob = {}

    for t in range(1, T + 1):
        s_t = paths[:, t]
        df = _discount(t, r, credit_spread)

        unpaid[alive] += c
        paid = alive & (s_t >= p.barrier_coupon)
        pv[paid] += N * unpaid[paid] * df
        n_coupons[paid] += unpaid[paid] / c if c > 0 else 0
        unpaid[paid] = 0.0
        if not p.memory:
            unpaid[alive & ~paid] = 0.0   # sans mémoire, le coupon manqué est perdu
        coupon_prob[t] = float(paid.mean() * 100)

        if t < T:
            recall = alive & (s_t >= p.recall_barrier_at(t))
            pv[recall] += N * df
            durations[recall] = t
            recall_year[recall] = t
            alive &= ~recall

    s_T = paths[alive, T]
    df_T = _discount(T, r, credit_spread)
    redemption = np.where(s_T >= p.barrier_capital, N, N * s_T)
    pv[alive] += redemption * df_T
    loss = np.zeros(n, dtype=bool)
    loss[alive] = s_T < p.barrier_capital
    recall_year[alive & (paths[:, T] >= p.recall_barrier_at(T))] = T

    return _summary("Phoenix Memory", pv, durations, recall_year, loss, N, T, antithetic, {
        "coupon_prob_by_year": coupon_prob,
        "avg_coupons": float(n_coupons.mean()),
    })


def price_athena(paths: np.ndarray, p: AthenaParams, r: float, credit_spread: float = 0.0,
                 antithetic: bool = False) -> dict:
    n, T, N, c = paths.shape[0], p.maturity, p.nominal, p.coupon_rate
    pv = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    durations = np.full(n, float(T))
    recall_year = np.zeros(n, dtype=int)

    for t in range(1, T):
        recall = alive & (paths[:, t] >= p.recall_barrier_at(t))
        pv[recall] = N * (1 + c * t) * _discount(t, r, credit_spread)
        durations[recall] = t
        recall_year[recall] = t
        alive &= ~recall

    s_T = paths[alive, T]
    df_T = _discount(T, r, credit_spread)
    above_final = s_T >= p.final_barrier
    above_capital = s_T >= p.barrier_capital
    redemption = np.where(above_final, N * (1 + c * T), np.where(above_capital, N, N * s_T))
    pv[alive] = redemption * df_T
    loss = np.zeros(n, dtype=bool)
    loss[alive] = ~above_capital
    final_recall = np.zeros(n, dtype=bool)
    final_recall[alive] = above_final
    recall_year[final_recall] = T

    capital_only = np.zeros(n, dtype=bool)
    capital_only[alive] = above_capital & ~above_final   # nominal rendu, sans coupon

    return _summary("Athena", pv, durations, recall_year, loss, N, T, antithetic, {
        "capital_only_pct": float(capital_only.mean() * 100),
    })


PRICERS = {"phoenix": price_phoenix, "athena": price_athena}
