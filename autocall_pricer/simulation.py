"""
Simulation des sous-jacents sous la probabilité risque-neutre Q.

Deux modèles :

· GBM (Black-Scholes, volatilité constante)
      S(t + Δt) = S(t) · exp[(r − q − σ²/2) Δt + σ √Δt · Z]
  Les produits n'étant observés qu'aux dates anniversaires, on simule directement les
  niveaux annuels : le schéma est exact.

· Heston (volatilité stochastique)
      dS = (r − q) S dt + √v S dW₁
      dv = κ (θ − v) dt + ξ √v dW₂,      d⟨W₁, W₂⟩ = ρ dt
  Avec ρ < 0, la volatilité monte quand le marché baisse : c'est ce qui crée le skew
  des options sur indices. Discrétisation d'Euler « full truncation » (Lord et al., 2010)
  à pas bimensuel.

Plusieurs sous-jacents : les chocs de prix sont corrélés par décomposition de Cholesky
de la matrice de corrélation.

Les tirages ne dépendent que de la graine, pas des paramètres : deux simulations avec la
même graine et des paramètres différents partagent les mêmes aléas (« common random
numbers »), ce qui rend les grecques et les sensibilités précises.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str = "gbm"            # "gbm" ou "heston"
    kappa: float = 1.5           # vitesse de retour de la variance vers θ
    xi: float = 0.5              # volatilité de la variance (vol of vol)
    rho: float = -0.7            # corrélation choc de prix / choc de variance
    term_ratio: float = 1.0      # √θ / √v₀ : vol long terme / vol court terme
    steps_per_year: int = 26       # pas bimensuel : écart < 0,1 pt avec un pas hebdomadaire

    @property
    def is_heston(self) -> bool:
        return self.name == "heston"


GBM = ModelSpec("gbm")


def correlation_matrix(n_assets: int, rho: float) -> np.ndarray:
    """Matrice d'équicorrélation (même corrélation entre toutes les paires)."""
    c = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


def _cholesky(corr: np.ndarray) -> np.ndarray:
    # Petit ajout sur la diagonale : la factorisation reste possible pour ρ proche de 1
    return np.linalg.cholesky(corr + 1e-12 * np.eye(len(corr)))


def _normals(rng: np.random.Generator, n: int, k: int, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return rng.standard_normal((n, k))
    half = (n + 1) // 2
    z = rng.standard_normal((half, k))
    return np.concatenate([z, -z])[:n]


def simulate_assets(
    n_simulations: int,
    maturity: int,
    sigmas,
    qs,
    r: float,
    correlation: float = 0.0,
    model: ModelSpec = GBM,
    seed: int | None = None,
    antithetic: bool = False,
    vol_shift: float = 0.0,
    vol_schedule=None,
) -> np.ndarray:
    """
    Performances des sous-jacents (niveau initial = 1) aux dates d'observation annuelles.
    Renvoie un tableau (n_simulations, maturity + 1, n_assets).

    `vol_schedule` (GBM uniquement) : volatilités forward annuelles, tableau (maturity, n_assets).
    L'année t utilise la volatilité vol_schedule[t − 1] : c'est un Black-Scholes à volatilité
    déterministe dépendant du temps, qui reproduit une structure par terme de vol ATM donnée.
    """
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=float)) + vol_shift
    qs = np.broadcast_to(np.atleast_1d(np.asarray(qs, dtype=float)), sigmas.shape)
    if np.any(sigmas <= 0):
        raise ValueError("Volatilité négative ou nulle.")
    n_assets = len(sigmas)
    chol = _cholesky(correlation_matrix(n_assets, correlation))
    rng = np.random.default_rng(seed)

    out = np.empty((n_simulations, maturity + 1, n_assets))
    out[:, 0, :] = 1.0
    log_s = np.zeros((n_simulations, n_assets))

    if not model.is_heston:
        if vol_schedule is None:
            schedule = np.tile(sigmas, (maturity, 1))
        else:
            schedule = np.asarray(vol_schedule, dtype=float).reshape(maturity, n_assets) + vol_shift
            if np.any(schedule <= 0):
                raise ValueError("Volatilité forward négative ou nulle.")
        for t in range(1, maturity + 1):
            sig = schedule[t - 1]
            z = _normals(rng, n_simulations, n_assets, antithetic) @ chol.T
            log_s += (r - qs - 0.5 * sig**2) + sig * z
            out[:, t, :] = np.exp(log_s)
        return out

    m = model.steps_per_year
    dt = 1.0 / m
    v = np.tile(sigmas**2, (n_simulations, 1))
    theta = (sigmas * model.term_ratio) ** 2
    rho_c = np.sqrt(1 - model.rho**2)
    for step in range(1, maturity * m + 1):
        z_price = _normals(rng, n_simulations, n_assets, antithetic) @ chol.T
        z_indep = _normals(rng, n_simulations, n_assets, antithetic)
        z_var = model.rho * z_price + rho_c * z_indep
        v_pos = np.maximum(v, 0.0)
        sq = np.sqrt(v_pos * dt)
        log_s += (r - qs - 0.5 * v_pos) * dt + sq * z_price
        v = v + model.kappa * (theta - v_pos) * dt + model.xi * sq * z_var
        if step % m == 0:
            out[:, step // m, :] = np.exp(log_s)
    return out


def worst_of(asset_paths: np.ndarray) -> np.ndarray:
    """Performance du plus mauvais sous-jacent à chaque date : (n_sim, T + 1)."""
    return asset_paths.min(axis=2)


# ── Interface mono sous-jacent (GBM), conservée pour les usages simples ──────────

def standard_normals(n_simulations: int, n_steps: int, seed: int | None = None,
                     antithetic: bool = False) -> np.ndarray:
    return _normals(np.random.default_rng(seed), n_simulations, n_steps, antithetic)


def gbm_paths_from_normals(z: np.ndarray, s0: float, r: float, q: float, sigma: float,
                           dt: float = 1.0) -> np.ndarray:
    """Trajectoires (n_simulations, n_steps + 1) d'un GBM à partir de tirages Z fixés."""
    log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return s0 * np.exp(log_paths)


def simulate_gbm(s0: float, r: float, q: float, sigma: float, maturity: int, n_simulations: int,
                 seed: int | None = None, antithetic: bool = False) -> np.ndarray:
    z = standard_normals(n_simulations, maturity, seed, antithetic)
    return gbm_paths_from_normals(z, s0, r, q, sigma)
