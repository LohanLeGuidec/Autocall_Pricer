"""
app.py — Dashboard Streamlit : Pricer Athena & Phoenix
=======================================================

Lancement : streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gbm import simulate_gbm
from phoenix import PhoenixParams, price_phoenix
from athena import AthenaParams, price_athena
from index_params import get_market_data, INDEX_INFO

# ── Config page ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Structured Products Pricer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Structured Products Pricer — Athena & Phoenix")
st.caption("Autocall Pricer · Méthode Monte Carlo · GBM")

# ── Sidebar : paramètres ───────────────────────────────────────────────────────
with st.sidebar:

    st.header("📈 Données indicatives")

    index = st.selectbox("Indice", INDEX_INFO.keys())
    market_data = get_market_data(index)

    st.markdown(
        f"""
    **Spot actuel :** {market_data['spot']:.2f}  
    **Dividende (%) :** {market_data['dividend_yield']*100:.2f}%  
    **Volatilité (%) :** {market_data['volatility']*100:.2f}%  
    **Taux sans risque (%) :** {market_data['risk_free_rate']*100:.2f}%  
    """
    )

    st.header("⚙️ Paramètres de marché")
    sigma = st.number_input("Volatilité implicite (%)", min_value=5.0, max_value=50.0, value=20.0, step=0.1) / 100
    r     = st.number_input("Taux sans risque (%)", min_value=0.0, max_value=6.0, value=3.0, step=0.01) / 100
    q     = st.number_input("Taux de dividende (%)", min_value=0.0, max_value=5.0, value=3.0, step=0.01) / 100
    T     = st.slider("Maturité (années)", 2, 15, 5, step=1)
    n_sim = st.select_slider("Simulations", [1000, 5000, 10000, 50000], value=10000)
    seed  = st.number_input("Seed (reproductibilité)", value=42, step=1, format="%d")

    st.divider()
    st.header("🦅 Paramètres Phoenix")
    ph_coupon   = st.slider("Coupon annuel Phoenix (%)", 3.0, 15.0, 8.0, step=0.1) / 100
    ph_recall   = st.slider("Barrière rappel Phoenix (%)", 80, 110, 100, step=1) / 100
    ph_coupon_b = st.slider("Barrière coupon mémoire (%)", 60, 95, 80, step=1) / 100
    ph_capital  = st.slider("Barrière capital Phoenix (%)", 40, 80, 60, step=1) / 100
    ph_stepdown = st.slider("Step-down annuel barrière (%)", 0, 10, 0, step=1) / 100

    st.divider()
    st.header("🏛️ Paramètres Athena")
    at_coupon  = st.slider("Coupon annuel Athena (%)", 3.0, 20.0, 12.0, step=0.1) / 100
    at_recall  = st.slider("Barrière rappel Athena (%)", 80, 110, 100, step=1) / 100
    at_capital = st.slider("Barrière capital Athena (%)", 55, 75, 65, step=1) / 100
    at_stepdown = st.slider("Step-down annuel barrière Athena (%)", 0, 10, 0, step=1) / 100

# ── Simulation ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Simulation Monte Carlo en cours…")
def run_all(sigma, r, q, T, n_sim, seed,
            ph_coupon, ph_recall, ph_coupon_b, ph_capital, ph_stepdown,
            at_coupon, at_recall, at_capital, at_stepdown):

    paths = simulate_gbm(S0=1.0, r=r, q=q, sigma=sigma, T=T,
                          n_simulations=n_sim, seed=int(seed))

    phoenix_params = PhoenixParams(
        coupon_rate=ph_coupon, barrier_recall=ph_recall,
        barrier_coupon=ph_coupon_b, barrier_capital=ph_capital,
        maturity=T, step_down=ph_stepdown,
    )
    athena_params = AthenaParams(
        coupon_rate=at_coupon, barrier_recall=at_recall,
        barrier_capital=at_capital, maturity=T, step_down=at_stepdown,
    )

    res_ph = price_phoenix(paths, phoenix_params, r=r)
    res_at = price_athena(paths, athena_params, r=r)

    return paths, res_ph, res_at, phoenix_params, athena_params

paths, res_ph, res_at, phoenix_params, athena_params = run_all(
    sigma, r, q, T, n_sim, seed,
    ph_coupon, ph_recall, ph_coupon_b, ph_capital, ph_stepdown,
    at_coupon, at_recall, at_capital, at_stepdown,
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.subheader("📌 Résultats clés")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
metrics = [
    ("Phoenix — Fair Value", f"{res_ph['fair_value_pct']:.2f}%", "% du nominal"),
    ("Phoenix — Rappel", f"{res_ph['recall_pct']:.1f}%", "scénarios"),
    ("Phoenix — Perte capital", f"{res_ph['loss_pct']:.1f}%", "scénarios"),
    ("Phoenix — Durée moy.", f"{res_ph['avg_duration']:.2f}", "années"),
    ("Athena — Fair Value", f"{res_at['fair_value_pct']:.2f}%", "% du nominal"),
    ("Athena — Rappel", f"{res_at['recall_pct']:.1f}%", "scénarios"),
    ("Athena — Perte capital", f"{res_at['loss_pct']:.1f}%", "scénarios"),
    ("Athena — Durée moy.", f"{res_at['avg_duration']:.2f}", "années"),
]
for col, (label, val, help_) in zip(
    [col1, col2, col3, col4, col5, col6, col7, col8], metrics
):
    col.metric(label, val, help=help_)

ci_ph = 1.96 * np.std(res_ph["payoffs_pv"], ddof=1) / np.sqrt(n_sim) / 1000 * 100
ci_at = 1.96 * np.std(res_at["payoffs_pv"], ddof=1) / np.sqrt(n_sim) / 1000 * 100
st.caption(f"📐 IC 95% — Phoenix : ±{ci_ph:.3f}%  |  Athena : ±{ci_at:.3f}%  · N = {n_sim:,} simulations")

st.divider()

# ── Graphiques ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trajectoires GBM",
    "💰 Distribution des payoffs",
    "📅 Rappels par année",
    "📉 Analyse de sensibilité"
])

# — Tab 1 : Trajectoires
with tab1:
    spot_actuel = market_data["spot"]
    years = list(range(T + 1))

    produit = st.radio(
        "Produit",
        ["🦅 Phoenix Memory", "🏛️ Athena"],
        horizontal=True,
        label_visibility="collapsed",
    )

    fig = go.Figure()

    # — Trajectoires
    for path in paths[:100] * spot_actuel:
        fig.add_trace(go.Scatter(
            x=years, y=path,
            mode="lines", line=dict(width=0.7, color="rgba(55,138,221,0.12)"),
            showlegend=False, hoverinfo="skip",
        ))

    # — Médiane
    median_path = np.median(paths, axis=0) * spot_actuel
    fig.add_trace(go.Scatter(
        x=years, y=median_path,
        mode="lines", line=dict(width=2.5, color="#185FA5"),
        name="Médiane",
    ))

    if produit == "🦅 Phoenix Memory":
        for t in range(1, T + 1):
            recall_t = (ph_recall - ph_stepdown * (t - 1)) * spot_actuel
            fig.add_trace(go.Scatter(
                x=[t - 1, t], y=[recall_t, recall_t],
                mode="lines", line=dict(color="#FF6B35", width=2.5, dash="dash"),
                showlegend=(t == 1), name="Rappel step-down",
            ))

        for label, val, color in [
            (f"Coupon mém. {int(ph_coupon_b*100)}%", ph_coupon_b * spot_actuel, "#EF9F27"),
            (f"Capital {int(ph_capital*100)}%",       ph_capital  * spot_actuel, "#E24B4A"),
        ]:
            fig.add_hline(y=val, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_position="right")

    else:  # Athena
        for t in range(1, T + 1):
            recall_t = (at_recall - at_stepdown * (t - 1)) * spot_actuel
            fig.add_trace(go.Scatter(
                x=[t - 1, t], y=[recall_t, recall_t],
                mode="lines", line=dict(color="#FF6B35", width=2.5, dash="dash"),
                showlegend=(t == 1), name="Rappel step-down",
            ))

        fig.add_hline(
            y=at_capital * spot_actuel, line_dash="dot", line_color="#E24B4A",
            annotation_text=f"Capital {int(at_capital*100)}%",
            annotation_position="right",
        )

    fig.update_layout(
        title=dict(text=produit, x=0.5, xanchor="center", y=0.97),
        height=500,
        margin=dict(l=0, r=140, t=80, b=0),
        yaxis_title=f"Niveau {index} (points)",
        xaxis_title="Années",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.05,
            xanchor="left", x=0,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    
# — Tab 2 : Distribution payoffs
with tab2:
    st.subheader("Distribution des payoffs actualisés (% du nominal — base 1 000 €)")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Phoenix Memory", "Athena"])

    for i, (res, color) in enumerate([(res_ph, "#378ADD"), (res_at, "#1D9E75")], 1):
        payoffs = res["payoffs_pv"] / 1000 * 100  # normalisé
        fig.add_trace(go.Histogram(
            x=payoffs, nbinsx=40,
            marker_color=color, opacity=0.75,
            showlegend=False,
        ), row=1, col=i)
        fig.add_vline(x=np.mean(payoffs), line_dash="dash", line_color="#E24B4A",
                      row=1, col=i,
                      annotation_text=f"Moy. {np.mean(payoffs):.1f}%")

    fig.update_xaxes(title_text="Payoff (% nominal)")
    fig.update_yaxes(title_text="Nombre de scénarios")
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Tableau comparatif
    st.subheader("Comparaison synthétique")
    df_compare = pd.DataFrame({
        "Métrique": [
            "Fair Value (%)", "Fair Value (€, nominal 1000€)",
            "Rappel anticipé (%)", "Perte en capital (%)",
            "Durée moyenne (ans)",
        ],
        "Phoenix Memory": [
            res_ph["fair_value_pct"], res_ph["fair_value_eur"],
            res_ph["recall_pct"], res_ph["loss_pct"], res_ph["avg_duration"],
        ],
        "Athena": [
            res_at["fair_value_pct"], res_at["fair_value_eur"],
            res_at["recall_pct"], res_at["loss_pct"], res_at["avg_duration"],
        ],
    })
    st.dataframe(df_compare.set_index("Métrique"), use_container_width=True)

# — Tab 3 : Rappels par année
with tab3:
    st.subheader("Probabilité de rappel anticipé par date d'observation")
    years_labels = [f"Année {t}" for t in range(1, T + 1)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Phoenix", x=years_labels,
        y=[res_ph["recall_by_year"].get(t, 0) for t in range(1, T + 1)],
        marker_color="#378ADD",
    ))
    fig.add_trace(go.Bar(
        name="Athena", x=years_labels,
        y=[res_at["recall_by_year"].get(t, 0) for t in range(1, T + 1)],
        marker_color="#1D9E75",
    ))
    fig.update_layout(
        barmode="group", yaxis_title="% des scénarios",
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# — Tab 4 : Analyse de sensibilité
with tab4:
    st.subheader("Sensibilité de la Fair Value aux paramètres de marché")

    param_choice = st.radio(
        "Paramètre analysé",
        ["Volatilité (σ)", "Taux sans risque (r)", "Barrière de rappel"],
        horizontal=True,
    )

    @st.cache_data(show_spinner="Calcul de la sensibilité…")
    def compute_sensitivity(param_name, sigma, r, q, T, n_sim, seed,
                            ph_coupon, ph_recall, ph_coupon_b, ph_capital, ph_stepdown,
                            at_coupon, at_recall, at_capital, at_stepdown):

        if param_name == "Volatilité (σ)":
            grid = np.linspace(0.05, 0.50, 20)
        elif param_name == "Taux sans risque (r)":
            grid = np.linspace(0.00, 0.06, 20)
        else:
            grid = np.linspace(0.80, 1.10, 20)

        fv_ph, fv_at = [], []

        for val in grid:
            s   = val  if param_name == "Volatilité (σ)"      else sigma
            ri  = val  if param_name == "Taux sans risque (r)" else r
            rec = val  if param_name == "Barrière de rappel"   else None

            paths_s = simulate_gbm(S0=1.0, r=ri, q=q, sigma=s,
                                   T=T, n_simulations=n_sim, seed=int(seed))

            ph_p = PhoenixParams(
                coupon_rate=ph_coupon,
                barrier_recall=rec if rec is not None else ph_recall,
                barrier_coupon=ph_coupon_b,
                barrier_capital=ph_capital,
                maturity=T, step_down=ph_stepdown,
            )
            at_p = AthenaParams(
                coupon_rate=at_coupon,
                barrier_recall=rec if rec is not None else at_recall,
                barrier_capital=at_capital,
                maturity=T, step_down=at_stepdown,
            )

            fv_ph.append(price_phoenix(paths_s, ph_p, r=ri)["fair_value_pct"])
            fv_at.append(price_athena(paths_s, at_p,  r=ri)["fair_value_pct"])

        return grid, fv_ph, fv_at

    grid, fv_ph, fv_at = compute_sensitivity(
        param_choice, sigma, r, q, T, n_sim, seed,
        ph_coupon, ph_recall, ph_coupon_b, ph_capital, ph_stepdown,
        at_coupon, at_recall, at_capital, at_stepdown,
    )

    # Valeur actuelle sur le graphique
    if param_choice == "Volatilité (σ)":
        current_val = sigma
        x_label = "Volatilité implicite"
        x_fmt = lambda v: f"{v*100:.0f}%"
    elif param_choice == "Taux sans risque (r)":
        current_val = r
        x_label = "Taux sans risque"
        x_fmt = lambda v: f"{v*100:.1f}%"
    else:
        x_label = "Barrière de rappel"
        x_fmt = lambda v: f"{v*100:.0f}%"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=fv_ph, mode="lines+markers",
        name="Phoenix", line=dict(color="#378ADD", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=fv_at, mode="lines+markers",
        name="Athena", line=dict(color="#1D9E75", width=2.5),
    ))

    if param_choice == "Barrière de rappel":
        fig.add_vline(x=ph_recall, line_dash="dash", line_color="#378ADD",
                      annotation_text="Rappel Phoenix")
        fig.add_vline(x=at_recall, line_dash="dash", line_color="#1D9E75",
                      annotation_text="Rappel Athena")
    else:
        fig.add_vline(
            x=current_val, line_dash="dash", line_color="#E24B4A",
            annotation_text="Valeur actuelle", annotation_position="top right",
        )

    fig.add_hline(y=100, line_dash="dot", line_color="gray",
                  annotation_text="100% nominal")

    fig.update_layout(
        height=420,
        xaxis_title=x_label,
        yaxis_title="Fair Value (% du nominal)",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau récapitulatif
    st.subheader("Points clés de la sensibilité")
    df_sensi = pd.DataFrame({
        x_label: [x_fmt(v) for v in grid],
        "Phoenix FV (%)": [round(v, 2) for v in fv_ph],
        "Athena FV (%)":  [round(v, 2) for v in fv_at],
    })
    st.dataframe(df_sensi.set_index(x_label), use_container_width=True)
# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Lohan Le Guidec - Master 1 MBFA - Ingénierie Économique et Financière - Université de Rennes · "

)