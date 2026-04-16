"""
gbm.py — Mouvement Brownien Géométrique (GBM)
Brique de base pour simuler les trajectoires du sous-jacent (ex: CAC 40).

Modèle : dS = S * (r - q) * dt + S * sigma * dW
  r     : taux sans risque
  q     : taux de dividende continu
  sigma : volatilité implicite annualisée
"""

import numpy as np


def simulate_gbm(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: int,
    n_simulations: int,
    dt: float = 1.0,
    seed: int = None,
) -> np.ndarray:
    """
    Simule N trajectoires annuelles d'un sous-jacent suivant un GBM.

    Paramètres
    ----------
    S0            : prix initial (ou niveau initial normalisé à 1.0)
    r             : taux sans risque annuel (ex: 0.03 pour 3%)
    q             : taux de dividende annuel (ex: 0.02 pour 2%)
    sigma         : volatilité annuelle (ex: 0.20 pour 20%)
    T             : maturité en années (nombre d'observations annuelles)
    n_simulations : nombre de trajectoires Monte Carlo
    dt            : pas de temps (1.0 = annuel)
    seed          : graine aléatoire pour reproductibilité

    Retourne
    --------
    paths : np.ndarray de shape (n_simulations, T+1)
            paths[:, 0] = S0, paths[:, t] = valeur à l'année t
    """
    if seed is not None:
        np.random.seed(seed)

    # Tirage des chocs browniens : shape (n_simulations, T)
    
    Z = np.random.standard_normal((n_simulations, T))

    # Facteur de dérive et de diffusion
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Construction des rendements log-normaux
    log_returns = drift + diffusion * Z  # (n_simulations, T)

    # Trajectoires cumulées
    log_paths = np.concatenate(
        [np.zeros((n_simulations, 1)), np.cumsum(log_returns, axis=1)],
        axis=1,
    )  # (n_simulations, T+1)

    paths = S0 * np.exp(log_paths)
    return paths


def compute_path_stats(paths: np.ndarray) -> dict:
    """
    Retourne des statistiques descriptives sur l'ensemble des trajectoires.
    Utile pour un premier diagnostic avant pricing.
    """
    final_values = paths[:, -1]
    return {
        "mean_final": float(np.mean(final_values)),
        "std_final": float(np.std(final_values)),
        "pct_above_100": float(np.mean(final_values >= 1.0) * 100),
        "pct_below_60": float(np.mean(final_values < 0.60) * 100),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
        "median": float(np.median(final_values)),
    }
