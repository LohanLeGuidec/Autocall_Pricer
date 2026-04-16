# Autocall Pricing Engine

> Moteur de valorisation Monte Carlo pour produits structurés autocall — Phoenix Memory & Athena.  
> Dashboard interactif Streamlit · GBM sous mesure risque-neutre Q · Données de marché temps réel.

---

## Contexte

Étudiant en M1 Ingénierie Économique et Financière, je me suis intéressé aux produits structurés, et plus particulièrement aux autocalls. Leur fair value m’a toujours semblé opaque : pas de formule fermée, des payoffs conditionnels dépendant de plusieurs barrières, une durée de vie incertaine…
Très vite, j’ai voulu tester moi‑même comment pricer un autocall de manière réaliste, en passant par la simulation Monte‑Carlo.

Ce projet modélise deux structures majeures du marché, le **Phoenix Memory** et l'**Athena**, sous un framework Black-Scholes risk-neutral, et restitue les résultats via un dashboard interactif avec indication de données live (S&P 500, Nasdaq 100, Dow Jones).

---

## Produits modélisés

### Phoenix Memory
Autocall à **effet mémoire** : le coupon est accumulé à chaque date d'observation où le sous-jacent dépasse la barrière mémoire, puis libéré intégralement à l'événement de rappel. Protection conditionnelle du capital à maturité.

### Athena
Autocall **sans barrière de coupon intermédiaire** : aucun flux avant rappel. Le coupon est capitalisé sur toute la durée — plus élevé en contrepartie, structure plus conservative.

Les deux produits supportent une variante **step-down** : la barrière de rappel décroît annuellement, augmentant la probabilité de rappel anticipé au fil du temps.

---

## Modèle

Simulation de N trajectoires du sous-jacent par **GBM**, sous la mesure martingale équivalente Q :

```
S(t+dt) = S(t) · exp[(r - q - σ²/2)dt + σ√dt · Z],   Z ~ N(0,1)
```

La fair value est estimée par **espérance actualisée des payoffs** sur l'ensemble des scénarios simulés, avec intervalle de confiance Monte Carlo à 95%.

---

## Stack

| Composant | Technologie |
|---|---|
| Simulation GBM | NumPy (vectorisé, O(N×T)) |
| Pricing & analyse | Python, Pandas |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Market data | yfinance (spot, vol implicite, dividendes, taux) |

---

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Structure

```
AUTOCALL_PRICER/
├── app.py                  # Dashboard Streamlit
├── src/
│   ├── gbm.py              # Simulateur GBM
│   ├── phoenix.py          # Pricer Phoenix Memory
│   ├── athena.py           # Pricer Athena
│   ├── monte_carlo.py      # Moteur MC : pricing, comparaison, sensibilité
│   └── index_params.py     # Market data live
└── notebooks/
    └── 01_exploration.ipynb
```

---

## Dashboard / Streamlit

- **Trajectoires GBM** — 100 paths simulées, médiane, barrières step-down dynamiques
- **Distribution des payoffs** — histogrammes actualisés Phoenix vs Athena
- **Rappels par année** — probabilités de rappel anticipé par date d'observation
- **Analyse de sensibilité** — fair value en fonction de σ, r ou B_recall

---

## Extensions envisagées

- Variance reduction (variables antithétiques, common random numbers)
- Modèles à volatilité stochastique (Heston, SABR)
- Greeks par différences finies (delta, vega)
- Worst-of sur panier multi-sous-jacents (décomposition de Cholesky)

---

**Lohan Le Guidec** — M1 MBFA, Ingénierie Économique et Financière, Université de Rennes
