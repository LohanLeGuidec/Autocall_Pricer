"""
phoenix.py — Produit Structuré : Autocall Phoenix Memory

Mécanique du Phoenix :
- Observations annuelles (t = 1, 2, ..., T)
- Barrière de RAPPEL   : si S(t) >= barrier_recall  → remboursement anticipé
                          + tous les coupons mémoire accumulés
- Barrière de COUPON   : si S(t) >= barrier_coupon   → coupon versé / mis en mémoire
- À maturité T (si pas rappelé) :
    · si S(T) >= barrier_capital → remboursement 100% + coupons mémoire
    · si S(T) <  barrier_capital → remboursement proportionnel à S(T)/S(0)
                                    (perte en capital)

Variante "Step-Down" : la barrière de rappel baisse chaque année (ex: 100%, 95%, 90%...)
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PhoenixParams:
    """Paramètres complets d'un Phoenix Memory."""

    nominal: float = 1000.0          # Nominal en euros
    coupon_rate: float = 0.08        # Coupon annuel (8%)
    barrier_recall: float = 1.00     # Barrière de rappel (100% du spot initial)
    barrier_coupon: float = 0.80     # Barrière de coupon / mémoire (80%)
    barrier_capital: float = 0.60    # Barrière de protection capital (60%)
    maturity: int = 5                # Maturité en années
    step_down: float = 0.0           # Décroissance annuelle de la barrière rappel
                                     # ex: 0.05 → barrière baisse de 5% par an
    observation_dates: Optional[List[int]] = None  # Années d'observation (défaut: 1..T)

    def __post_init__(self):
        if self.observation_dates is None:
            self.observation_dates = list(range(1, self.maturity + 1))

    def recall_barrier_at(self, year: int) -> float:
        """Retourne la barrière de rappel à l'année t (avec step-down éventuel)."""
        return self.barrier_recall - self.step_down * (year - 1)


def price_phoenix(
    paths: np.ndarray,
    params: PhoenixParams,
    r: float,
    dt: float = 1.0,
) -> dict:
    """
    Price un Phoenix Memory sur un ensemble de trajectoires simulées.

    Paramètres
    ----------
    paths  : np.ndarray (n_simulations, T+1) — trajectoires normalisées (S0=1)
    params : PhoenixParams
    r      : taux d'actualisation annuel
    dt     : pas de temps (1.0 = annuel)

    Retourne
    --------
    dict avec fair_value, recall_pct, loss_pct, avg_duration,
         recall_by_year, payoffs, coupon_payments
    """
    n_sims = paths.shape[0]
    T = params.maturity
    obs = params.observation_dates

    payoffs_pv = np.zeros(n_sims)
    durations = np.zeros(n_sims)
    recalled = np.zeros(n_sims, dtype=bool)
    recall_year = np.zeros(n_sims, dtype=int)
    loss_flag = np.zeros(n_sims, dtype=bool)
    coupon_memory = np.zeros(n_sims)       # coupons accumulés non versés

    for t in obs:
        S_t = paths[:, t]
        active = ~recalled

        # — Coupon mémoire : si S >= barrier_coupon, on accumule le coupon
        coupon_memory[active & (S_t >= params.barrier_coupon)] += params.coupon_rate

        # — Rappel anticipé : si S >= barrier_recall_at(t)
        recall_barrier_t = params.recall_barrier_at(t)
        trigger_recall = active & (S_t >= recall_barrier_t)

        if np.any(trigger_recall):
            # Payoff = nominal * (1 + coupons mémorisés)
            gross_payoff = params.nominal * (1 + coupon_memory[trigger_recall])
            pv = gross_payoff * np.exp(-r * t * dt)
            payoffs_pv[trigger_recall] = pv
            durations[trigger_recall] = t
            recall_year[trigger_recall] = t
            recalled[trigger_recall] = True

    # — À maturité : trajectoires non rappelées
    still_alive = ~recalled
    if np.any(still_alive):
        S_T = paths[still_alive, T]        
        mem = coupon_memory[still_alive]     

        above_capital = S_T >= params.barrier_capital 
        below_capital = ~above_capital                  

        # Tableau de payoffs dans l'espace réduit
        payoffs_alive = np.zeros(still_alive.sum())

        # Au-dessus de la barrière capital : remboursement 100% + mémoire
        gross_above = params.nominal * (1 + mem[above_capital])
        payoffs_alive[above_capital] = gross_above * np.exp(-r * T * dt)

        # En dessous : perte proportionnelle (pas de coupon)
        gross_below = params.nominal * S_T[below_capital]
        payoffs_alive[below_capital] = gross_below * np.exp(-r * T * dt)

        # Réinjection dans les tableaux globaux
        payoffs_pv[still_alive] = payoffs_alive

        loss_mask = np.zeros(n_sims, dtype=bool)
        loss_mask[still_alive] = below_capital
        loss_flag |= loss_mask

        durations[still_alive] = T

    # Statistiques par année de rappel
    recall_by_year = {}
    for t in obs:
        n = np.sum(recall_year == t)
        recall_by_year[t] = float(n / n_sims * 100)

    fair_value_pct = float(np.mean(payoffs_pv) / params.nominal * 100)

    return {
        "product": "Phoenix Memory",
        "fair_value_pct": round(fair_value_pct, 4),
        "fair_value_eur": round(float(np.mean(payoffs_pv)), 2),
        "recall_pct": round(float(np.mean(recalled) * 100), 2),
        "loss_pct": round(float(np.mean(loss_flag) * 100), 2),
        "avg_duration": round(float(np.mean(durations)), 3),
        "recall_by_year": recall_by_year,
        "payoffs_pv": payoffs_pv,
        "payoffs_gross": payoffs_pv * np.exp(r * durations),  # non actualisé
        "loss_flags": loss_flag,
        "durations": durations,
    }