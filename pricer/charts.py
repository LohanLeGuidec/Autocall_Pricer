"""Graphiques Plotly, sérialisés en JSON pour Plotly.js."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHOENIX = "#2F6FB3"
ATHENA = "#1D8A66"
COLORS = {"phoenix": PHOENIX, "athena": ATHENA}
NAMES = {"phoenix": "Phoenix Memory", "athena": "Athena"}
RED, ORANGE, AMBER, GREY = "#C4413D", "#D9622B", "#C9901A", "#9A968E"


def _layout(height=380, **kwargs):
    layout = dict(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12, color="#2B2A27"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(gridcolor="#E7E4DD", zerolinecolor="#CFCBC2"),
        yaxis=dict(gridcolor="#E7E4DD", zerolinecolor="#CFCBC2"),
    )
    for key in ("xaxis", "yaxis"):
        if key in kwargs:
            layout[key] = {**layout[key], **kwargs.pop(key)}
    layout.update(kwargs)
    return layout


def to_json(fig: go.Figure) -> str:
    return fig.to_json().replace("</", "<\\/")


def _rgba(hex_color, a):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


def paths_chart(product: str, params, sample: np.ndarray, quantiles: dict, underlying_label: str) -> str:
    """Trajectoires en % du niveau initial (performance du plus mauvais sous-jacent si worst-of)."""
    T = params.maturity
    years = np.arange(T + 1)
    color = COLORS[product]
    fig = go.Figure()
    for p in sample * 100:
        fig.add_trace(go.Scatter(x=years, y=p, mode="lines", line=dict(width=0.6, color=_rgba(color, 0.14)),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([quantiles["p95"], quantiles["p05"][::-1]]) * 100,
        fill="toself", fillcolor=_rgba(color, 0.08), line=dict(width=0), mode="lines", name="Intervalle 5 %–95 %",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=years, y=quantiles["median"] * 100, mode="lines", name="Médiane",
                             line=dict(color=color, width=2.6),
                             hovertemplate="Année %{x}<br>%{y:.1f} %<extra>Médiane</extra>"))

    # Barrière de rappel (en escalier si step-down), observée à partir de l'année 1
    rec_x, rec_y = [], []
    for t in range(1, T + 1):
        lvl = params.recall_barrier_at(t) * 100
        rec_x += [t - 0.5, t + 0.5 if t < T else t]
        rec_y += [lvl, lvl]
    fig.add_trace(go.Scatter(x=rec_x, y=rec_y, mode="lines", name="Barrière de rappel",
                             line=dict(color=ORANGE, width=2.2, dash="dash")))
    if product == "phoenix":
        fig.add_hline(y=params.barrier_coupon * 100, line_dash="dot", line_color=AMBER,
                      annotation_text=f"Coupon {params.barrier_coupon:.0%}", annotation_position="right")
    fig.add_hline(y=params.barrier_capital * 100, line_dash="dot", line_color=RED,
                  annotation_text=f"Capital {params.barrier_capital:.0%}", annotation_position="right")
    y_name = "Worst-of" if underlying_label.startswith("worst-of") else underlying_label
    fig.update_layout(**_layout(460, xaxis=dict(title="Années", dtick=1),
                                yaxis=dict(title=f"{y_name} (% du niveau initial)",
                                           ticksuffix=" %"),
                                margin=dict(l=8, r=90, t=36, b=8)))
    return to_json(fig)


def smile_chart(smile: dict, name: str) -> str:
    """Volatilités implicites à maturité produites par le Heston, comparées à la vol plate du GBM."""
    strikes = np.asarray(smile["strikes"]) * 100
    heston = np.asarray(smile["heston"]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=heston, mode="lines+markers", name="Heston (smile implicite)",
                             line=dict(color=RED, width=2.6), marker=dict(size=6),
                             hovertemplate="Strike %{x:.0f} %<br>Vol %{y:.2f} %<extra>Heston</extra>"))
    fig.add_trace(go.Scatter(x=strikes, y=np.full_like(strikes, smile["gbm"] * 100), mode="lines",
                             name="Black-Scholes, vol saisie", line=dict(color=GREY, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=strikes, y=np.full_like(strikes, smile["atm_vols"][0] * 100), mode="lines",
                             name="Black-Scholes, vol ATM équivalente",
                             line=dict(color=PHOENIX, width=2, dash="dot")))
    fig.update_layout(**_layout(380, xaxis=dict(title=f"Strike (% du niveau initial de {name})", ticksuffix=" %"),
                                yaxis=dict(title="Volatilité implicite à maturité", ticksuffix=" %")))
    return to_json(fig)


def worst_share_chart(shares, names: list[str]) -> str:
    shares = list(shares)
    fig = go.Figure(go.Bar(x=names, y=shares, marker_color=[PHOENIX, ATHENA, AMBER][:len(names)],
                           text=[f"{v:.0f} %" for v in shares], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>"))
    fig.update_layout(**_layout(300, yaxis=dict(title="Plus mauvais à maturité", ticksuffix=" %")))
    return to_json(fig)


def payoff_histograms(results: dict) -> str:
    fig = make_subplots(rows=1, cols=2, subplot_titles=[NAMES["phoenix"], NAMES["athena"]],
                        horizontal_spacing=0.08)
    for i, k in enumerate(("phoenix", "athena"), start=1):
        res = results[k]
        x = res["payoffs_pv"] / 1000 * 100
        fig.add_trace(go.Histogram(x=x, nbinsx=60, marker_color=COLORS[k], opacity=0.8, showlegend=False,
                                   hovertemplate="%{x:.0f} %<br>%{y} scénarios<extra></extra>"), row=1, col=i)
        fig.add_vline(x=res["fair_value_pct"], line_dash="dash", line_color=RED, row=1, col=i,
                      annotation_text=f"FV {res['fair_value_pct']:.1f} %")
    fig.update_xaxes(title_text="Valeur actualisée des flux (% du nominal)", gridcolor="#E7E4DD")
    fig.update_yaxes(title_text="Scénarios", gridcolor="#E7E4DD")
    fig.update_layout(**_layout(380))
    return to_json(fig)


def recall_bars(results: dict, maturity: int) -> str:
    years = [f"Année {t}" for t in range(1, maturity + 1)]
    fig = go.Figure()
    for k in ("phoenix", "athena"):
        y = [results[k]["recall_by_year"][t] for t in range(1, maturity + 1)]
        fig.add_trace(go.Bar(name=NAMES[k], x=years, y=y, marker_color=COLORS[k],
                             hovertemplate="%{x}<br>%{y:.1f} %<extra>" + NAMES[k] + "</extra>"))
    fig.update_layout(**_layout(340, barmode="group", yaxis=dict(title="% des scénarios", ticksuffix=" %")))
    return to_json(fig)


def coupon_probability(result: dict, maturity: int) -> str:
    years = [f"Année {t}" for t in range(1, maturity + 1)]
    y = [result["coupon_prob_by_year"][t] for t in range(1, maturity + 1)]
    fig = go.Figure(go.Bar(x=years, y=y, marker_color=PHOENIX, text=[f"{v:.0f} %" for v in y],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>"))
    fig.update_layout(**_layout(300, yaxis=dict(title="Probabilité de versement", ticksuffix=" %")))
    return to_json(fig)


SENSI_LABELS = {
    "sigma": ("Volatilité", 100, "%"),
    "r": ("Taux sans risque", 100, "%"),
    "barrier_capital": ("Barrière de capital", 100, "%"),
    "credit_spread": ("Spread de crédit", 100, "%"),
    "correlation": ("Corrélation entre sous-jacents", 100, "%"),
}


def sensitivity_chart(what: str, data: dict, current: dict) -> str:
    label, scale, suffix = SENSI_LABELS[what]
    fig = go.Figure()
    for k in ("phoenix", "athena"):
        grid, values = data[k]
        fig.add_trace(go.Scatter(x=grid * scale, y=values, mode="lines+markers", name=NAMES[k],
                                 line=dict(color=COLORS[k], width=2.4), marker=dict(size=5),
                                 hovertemplate=f"{label} %{{x:.2f}} {suffix}<br>FV %{{y:.2f}} %"
                                               f"<extra>{NAMES[k]}</extra>"))
    for k, x in current.items():
        if x is not None:
            fig.add_vline(x=x * scale, line_dash="dash", line_color=COLORS[k], opacity=0.6)
    fig.add_hline(y=100, line_dash="dot", line_color=GREY, annotation_text="Pair", annotation_position="bottom right")
    fig.update_layout(**_layout(320, xaxis=dict(title=f"{label} ({suffix})"),
                                yaxis=dict(title="Fair value (% du nominal)")))
    return to_json(fig)
