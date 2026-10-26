# Autocall Pricer

Valorisation Monte Carlo d'autocalls **Phoenix Memory** et **Athena**, sur un indice ou un **worst-of** de 2 à 3 indices, en **Black-Scholes** ou en **Heston** (volatilité stochastique, **calibrable sur les options cotées**) : fair value, coupon d'équilibre, grecques, sensibilités et impact du skew, dans une application web Django.

*Lohan Le Guidec — Master 2 MBFA, Ingénierie Économique et Financière, Université de Rennes*

## Contexte

Les autocalls font partie des produits structurés les plus distribués en Europe, mais leur prix n'a pas de formule fermée : barrières multiples, coupons conditionnels, durée de vie aléatoire. Ce projet les valorise par simulation, et répond aux deux questions qu'un structureur se pose réellement :

- **Combien vaut le produit ?** (fair value en % du nominal)
- **Quel coupon peut-on offrir** pour un prix cible donné (100 % − marge) ?

## Fonctionnalités

| Onglet | Contenu |
|---|---|
| Synthèse | Fair value et intervalle de confiance à 95 %, coupon d'équilibre, probabilités de rappel et de perte, durée de vie moyenne, grecques (delta, gamma, vega, rho, sensibilité au spread de crédit) |
| Trajectoires | Trajectoires simulées (performance du plus mauvais sous-jacent), médiane et intervalle 5 %–95 %, barrières ; pour un worst-of, fréquence à laquelle chaque indice est le plus mauvais |
| Distribution des flux | Valeur actualisée des flux par scénario, pour chaque produit |
| Rappels & coupons | Probabilité de remboursement par date d'observation, probabilité de coupon (Phoenix) |
| Sensibilités | Fair value en fonction de la volatilité, du taux, de la barrière de capital, du spread de crédit et, pour un worst-of, de la corrélation |
| Modèle & skew | Black-Scholes contre Heston à volatilité ATM égale, smile implicite produit par le Heston, lecture de l'effet du skew sur chaque produit |
| Méthodologie | Modèle, conventions de payoff et limites |

Dans le panneau **Modèle**, le bouton **« Calibrer sur les options cotées »** ajuste le Heston sur la surface de volatilité de l'indice (S&P 500, Nasdaq 100) et remplit les paramètres du formulaire.

Sous-jacents : S&P 500, Nasdaq 100, Dow Jones, Euro Stoxx 50, CAC 40, DAX, seuls ou en worst-of. Les paramètres de marché sont **préremplis avec des données indicatives** (Yahoo Finance) et restent modifiables :

- volatilité implicite à 30 jours (VIX, VXN, VXD) ou, à défaut, historique sur 1 an ;
- **structure par terme** de la volatilité (VIX 1 an / VIX 30 jours), qui fixe la variance de long terme du Heston ;
- **corrélation** estimée sur 2 ans de rendements **hebdomadaires** (les rendements quotidiens la sous-estiment entre places dont les séances ne se recouvrent pas) ;
- taux 5 ans américain, dividendes moyens.

## Résultat principal : le skew n'a pas le même effet sur les deux produits

À volatilité à la monnaie égale, passer de Black-Scholes à Heston (ρ = −0,7) **baisse la fair value du Phoenix** d'environ 1,5 à 2,5 points : l'investisseur est vendeur d'une put down-and-in de barrière basse, plus chère avec le skew. **L'Athena bouge beaucoup moins, voire monte** : ses gros coupons cumulés ne sont payés qu'au-dessus de la barrière de rappel, une option digitale à la monnaie que le skew rend plus probable. Un pricer à volatilité constante surestime donc le coupon que l'émetteur peut servir sur un Phoenix.

## Calibration du modèle de Heston

Les paramètres du Heston peuvent être calibrés sur le marché plutôt que fixés à des valeurs typiques :

1. **Données** : options hors de la monnaie sur l'indice (Yahoo Finance, maturités de 1 mois à 3 ans, filtres de liquidité sur la fourchette bid-ask), ou surface importée en CSV, par exemple exportée de Bloomberg OVDV (`maturity_years;strike_pct;implied_vol_pct[;forward_pct]`). Le forward de chaque maturité est tiré de la **parité call-put** : le dividende n'a pas à être estimé.
2. **Prix Heston par formule semi-fermée** : transformée de Fourier de la fonction caractéristique, forme *little Heston trap* (Albrecher et al., 2007), quadrature de Gauss-Legendre. Environ 1 seconde pour une calibration complète, là où le Monte Carlo serait trop lent.
3. **Moindres carrés pondérés par 1/vega** (`scipy.optimize.least_squares`) : on minimise l'écart en volatilité implicite sur toute la surface, sous les bornes du formulaire.
4. **Conversion** dans la paramétrisation du pricer : vol = √v₀, vol long terme / court terme = √(θ/v₀), κ, ξ et ρ inchangés. L'erreur moyenne en points de vol et la condition de Feller (2κθ ≥ ξ²) sont affichées.

En ligne de commande :

```powershell
python manage.py calibrate_heston --index "S&P 500"
python manage.py calibrate_heston --csv surface_sx5e.csv --r 0.025
```

La formule est validée contre le Monte Carlo du pricer (même modèle, deux méthodes), contre Black-Scholes quand ξ → 0, et la calibration retrouve des paramètres connus sur une surface synthétique, avec ou sans bruit de cotation.

## Produits

**Phoenix Memory** — à chaque date annuelle : coupon versé si le sous-jacent est au-dessus de la barrière de coupon, avec **rattrapage des coupons manqués** (effet mémoire) ; remboursement anticipé au-dessus de la barrière de rappel. À maturité : nominal si le sous-jacent est au-dessus de la barrière de capital, sinon nominal × S(T)/S(0). Les coupons déjà versés restent acquis.

**Athena** — aucun flux avant le rappel ; au rappel en année t : nominal × (1 + coupon × t). À maturité : coupons complets au-dessus de la barrière finale, **nominal seul** entre la barrière de capital et la barrière finale, nominal × S(T)/S(0) en dessous.

Les deux produits acceptent un **step-down** (barrière de rappel décroissante).

## Méthodologie

- **Black-Scholes** : `S(t+1) = S(t) · exp[(r − q − σ²/2) + σZ]`. Les produits n'étant observés qu'une fois par an, les niveaux annuels sont simulés directement : le schéma est exact.
- **Heston** : `dS = (r − q)S dt + √v S dW₁`, `dv = κ(θ − v)dt + ξ√v dW₂`, `d⟨W₁, W₂⟩ = ρ dt`. v₀ = σ², θ issu de la structure par terme. Schéma d'Euler *full truncation* (Lord et al., 2010), 26 pas par an.
- **Comparaison des modèles** : le Black-Scholes de référence utilise la volatilité implicite ATM produite par le Heston, pour isoler l'effet du skew de celui du niveau de volatilité.
- **Worst-of** : chocs corrélés par décomposition de Cholesky, produit observé sur `min(Sᵢ(t)/Sᵢ(0))`.
- **Contrôle de martingale** (E[S_T] = e^{(r−q)T}) affiché pour chaque sous-jacent.
- **Actualisation** : taux sans risque + **spread de crédit de l'émetteur** (un autocall est une obligation de la banque).
- **Réduction de variance** : variables antithétiques ; intervalle de confiance calculé sur les paires.
- **Coupon d'équilibre** : exact, car la valeur des deux produits est affine en coupon sur des trajectoires données.
- **Grecques et sensibilités** : différences finies centrées avec les mêmes tirages aléatoires (*common random numbers*), pour mesurer l'effet du paramètre et non le bruit Monte Carlo.

## Installation et lancement

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000. Aucune base de données n'est nécessaire.

## Tests

```powershell
pytest
```

Les payoffs sont testés sur des **trajectoires construites à la main** dont le résultat se calcule de tête (effet mémoire, coupons conservés après une perte, step-down, coupon final de l'Athena…), ainsi que le caractère martingale du GBM et du Heston, la convergence du Heston vers le GBM quand ξ → 0, le skew produit par ρ < 0, la corrélation simulée, la monotonie du worst-of en corrélation, le coupon d'équilibre et le signe des grecques. Les tests n'utilisent pas le réseau.

## Structure

```
manage.py                 Point d'entrée Django
config/                   Réglages et URLs
pricer/                   Application web : formulaire, vues, graphiques Plotly, templates, CSS/JS,
                          commande calibrate_heston
autocall_pricer/
  simulation.py           GBM, Heston, worst-of corrélé (Cholesky), tirages antithétiques
  products.py             Payoffs Phoenix Memory et Athena
  analytics.py            Grecques, coupon d'équilibre, sensibilités, smile implicite
  calibration.py          Heston semi-fermé, calibration sur la surface de volatilité
  pricing.py              Calcul complet pour l'interface (dont comparaison BS / Heston)
  market_data.py          Données de marché indicatives (Yahoo Finance)
tests/                    Tests pytest
```

## Limites

- **Calibration de Heston** : disponible pour les indices américains (options Yahoo Finance) ou par import d'une surface ; pour les indices européens, il faut fournir la surface (pas de chaîne d'options gratuite). Un Heston à paramètres constants ne reproduit jamais parfaitement toute la surface, en particulier le skew très court terme : un desk utiliserait une volatilité locale ou locale-stochastique. Pour un worst-of, seul le premier sous-jacent est calibré.
- **Données de marché indicatives** : pas de surface de volatilité complète ni de courbe de taux ; corrélation historique (la corrélation implicite est en général plus élevée en période de stress).
- Corrélation identique entre toutes les paires, pas d'ajustement quanto pour un worst-of multi-devises, dividende continu, spread de crédit constant, observations annuelles.
