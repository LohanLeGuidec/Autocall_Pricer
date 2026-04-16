"""
athena.py — Produit Structuré : Autocall Athena
================================================

Différence clé avec le Phoenix :
- PAS de barrière de coupon intermédiaire
- Le coupon est versé UNIQUEMENT si le produit est rappelé (ou à maturité si S >= barrier)
- Plus simple, plus conservateur : le client ne reçoit rien tant que le produit vit
- Le coupon est souvent plus élevé en compensation

Mécanique de l'Athena :
- Observations annuelles (t = 1, 2, ..., T)
- Barrière de RAPPEL  : si S(t) >= barrier_recall
                         → remboursement 100% + coupon * t (coupons de toutes les années)
- À maturité T (si pas rappelé) :
    · si S(T) >= barrier_capital → remboursement 100% + coupon * T
    · si S(T) <  barrier_capital → remboursement proportionnel S(T)/S(0)
                                    (perte en capital, PAS de coupon)

Variante "Double Barrière" : deux sous-jacents, rappel ssi les deux sont au-dessus.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AthenaParams:
    """Paramètres complets d'un Athena."""

    nominal: float = 1000.0          # Nominal en euros
    coupon_rate: float = 0.10        # Coupon par période observée (10%)
                                     # Versé en totalité seulement si rappel/maturité >= barrier
    barrier_recall: float = 1.00     # Barrière de rappel (100%)
    barrier_capital: float = 0.60    # Barrière de protection capital (60%)
    maturity: int = 5                # Maturité en années
    step_down: float = 0.0           # Décroissance barrière rappel (0 = classique)
    observation_dates: Optional[List[int]] = None

    def __post_init__(self):
        if self.observation_dates is None:
            self.observation_dates = list(range(1, self.maturity + 1))

    def recall_barrier_at(self, year: int) -> float:
        return self.barrier_recall - self.step_down * (year - 1)


def price_athena(
    paths: np.ndarray,
    params: AthenaParams,
    r: float,
    dt: float = 1.0,
) -> dict:
    """
    Price un Athena sur un ensemble de trajectoires simulées.

    Paramètres
    ----------
    paths  : np.ndarray (n_simulations, T+1) — trajectoires normalisées (S0=1)
    params : AthenaParams
    r      : taux d'actualisation annuel
    dt     : pas de temps (1.0 = annuel)

    Retourne
    --------
    dict avec fair_value, recall_pct, loss_pct, avg_duration,
         recall_by_year, payoffs, etc.
    """
    n_sims = paths.shape[0]
    T = params.maturity
    obs = params.observation_dates

    payoffs_pv = np.zeros(n_sims)
    durations = np.zeros(n_sims)
    recalled = np.zeros(n_sims, dtype=bool)
    recall_year = np.zeros(n_sims, dtype=int)
    loss_flag = np.zeros(n_sims, dtype=bool)

    for t in obs:
        S_t = paths[:, t]
        active = ~recalled
        recall_barrier_t = params.recall_barrier_at(t)

        # Rappel : S(t) >= barrière de rappel
        trigger_recall = active & (S_t >= recall_barrier_t)

        if np.any(trigger_recall):
            # Coupon = coupon_rate * t (toutes les périodes depuis le début)
            gross_payoff = params.nominal * (1 + params.coupon_rate * t)
            pv = gross_payoff * np.exp(-r * t * dt)
            payoffs_pv[trigger_recall] = pv
            durations[trigger_recall] = t
            recall_year[trigger_recall] = t
            recalled[trigger_recall] = True

    # À maturité
    still_alive = ~recalled
    if np.any(still_alive):
        S_T = paths[still_alive, T]          # taille : still_alive.sum()

        above_capital = S_T >= params.barrier_capital   
        below_capital = ~above_capital                 

        # Tableau de payoffs dans l'espace réduit
        payoffs_alive = np.zeros(still_alive.sum())

        # Au-dessus : 100% + coupon total
        gross_above = params.nominal * (1 + params.coupon_rate * T)
        payoffs_alive[above_capital] = gross_above * np.exp(-r * T * dt)

        # En dessous : perte proportionnelle, pas de coupon
        gross_below = params.nominal * S_T[below_capital]
        payoffs_alive[below_capital] = gross_below * np.exp(-r * T * dt)

        # Réinjection dans les tableaux globaux
        payoffs_pv[still_alive] = payoffs_alive

        loss_mask = np.zeros(n_sims, dtype=bool)
        loss_mask[still_alive] = below_capital
        loss_flag |= loss_mask

        durations[still_alive] = T

    recall_by_year = {}
    for t in obs:
        n = np.sum(recall_year == t)
        recall_by_year[t] = float(n / n_sims * 100)

    fair_value_pct = float(np.mean(payoffs_pv) / params.nominal * 100)

    return {
        "product": "Athena",
        "fair_value_pct": round(fair_value_pct, 4),
        "fair_value_eur": round(float(np.mean(payoffs_pv)), 2),
        "recall_pct": round(float(np.mean(recalled) * 100), 2),
        "loss_pct": round(float(np.mean(loss_flag) * 100), 2),
        "avg_duration": round(float(np.mean(durations)), 3),
        "recall_by_year": recall_by_year,
        "payoffs_pv": payoffs_pv,
        "payoffs_gross": payoffs_pv * np.exp(r * durations),
        "loss_flags": loss_flag,
        "durations": durations,
    }