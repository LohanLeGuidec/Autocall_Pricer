import hashlib
import json
from functools import lru_cache

from django.core.cache import cache
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from autocall_pricer import (
    INDICES,
    AthenaParams,
    Market,
    ModelSpec,
    PhoenixParams,
    calibrate_heston,
    fetch_option_quotes,
    get_correlation,
    get_market_data,
    run_pricing,
)

from . import charts
from .forms import MAX_UNDERLYINGS, PricingForm

ANALYSIS_VERSION = 2
MARKET_TTL = 3600
DEFAULT_UNDERLYING = PricingForm.base_fields["underlying_1"].initial


# ── Formatage ────────────────────────────────────────────────────────────────

def pct(x, digits=2):
    return f"{x:.{digits}f} %".replace("-", "−").replace(".", ",")


def num(x, digits=2):
    return f"{x:+.{digits}f}".replace("-", "−").replace(".", ",")


def dec(x, digits=2):
    return f"{x:.{digits}f}".replace("-", "−").replace(".", ",")


def table(columns, rows):
    return {"columns": columns, "rows": rows, "first_col_label": True}


# ── Données de marché (mises en cache 1 h) ───────────────────────────────────

def _key(prefix: str, value: str) -> str:
    return f"{prefix}:" + hashlib.sha256(value.encode()).hexdigest()[:16]


def cached_market_data(index_name: str) -> dict:
    key = _key("market", index_name)
    data = cache.get(key)
    if data is None:
        data = get_market_data(index_name)
        cache.set(key, data, MARKET_TTL)
    return data


def cached_correlation(names: list[str]) -> dict:
    key = _key("corr", "|".join(sorted(names)))
    data = cache.get(key)
    if data is None:
        data = get_correlation(sorted(names))
        cache.set(key, data, MARKET_TTL)
    return data


@require_GET
def market_data(request):
    index_name = request.GET.get("index", "")
    if index_name not in INDICES:
        raise Http404("Indice inconnu")
    data = cached_market_data(index_name)
    return JsonResponse({
        "spot": data["spot"],
        "sigma": round(data["sigma"] * 100, 2),
        "r": round(data["r"] * 100, 2),
        "q": round(data["q"] * 100, 2),
        "term_ratio": data["term_ratio"],
        "currency": data["currency"],
        "sources": data["sources"],
    })


@require_GET
def correlation(request):
    names = [n for n in request.GET.get("indices", "").split(",") if n]
    if len(names) < 2 or len(set(names)) != len(names) or any(n not in INDICES for n in names):
        raise Http404("Sous-jacents invalides")
    data = cached_correlation(names)
    return JsonResponse({"correlation": round(data["correlation"] * 100), "source": data["source"],
                         "matrix": data["matrix"]})


CALIBRATION_TTL = 3600


def cached_calibration(index_name: str) -> dict:
    """Calibration de Heston sur les options cotées de l'indice (mise en cache 1 h)."""
    key = _key("calib", index_name)
    data = cache.get(key)
    if data is None:
        r = cached_market_data(index_name)["r"]
        data = calibrate_heston(fetch_option_quotes(index_name, r)).summary()
        cache.set(key, data, CALIBRATION_TTL)
    return data


@require_GET
def calibration(request):
    """Paramètres Heston calibrés, au format du formulaire (vol en %)."""
    index_name = request.GET.get("index", "")
    if index_name not in INDICES:
        raise Http404("Indice inconnu")
    try:
        c = cached_calibration(index_name)
    except Exception as exc:  # noqa: BLE001 — message affiché à l'utilisateur
        return JsonResponse({"error": str(exc)}, status=422)
    return JsonResponse({
        "sigma": round(c["sigma0"] * 100, 2), "kappa": c["kappa"], "xi": c["xi"], "rho": c["rho"],
        "term_ratio": c["term_ratio"], "rmse": c["rmse_vol_pts"], "n_options": c["n_options"],
        "feller": c["feller"], "source": c["source"],
    })


# ── Pricing ──────────────────────────────────────────────────────────────────

def _heston_spec(d: dict, name: str) -> ModelSpec:
    return ModelSpec(name, kappa=d["heston_kappa"], xi=d["heston_xi"], rho=d["heston_rho"],
                     term_ratio=d["term_ratio"])


def _build_inputs(d: dict, underlyings: list):
    phoenix = PhoenixParams(
        coupon_rate=d["ph_coupon"] / 100, barrier_recall=d["ph_recall"] / 100,
        barrier_coupon=d["ph_coupon_barrier"] / 100, barrier_capital=d["ph_capital"] / 100,
        maturity=d["maturity"], step_down=d["ph_step_down"] / 100, memory=d["ph_memory"],
    )
    athena = AthenaParams(
        coupon_rate=d["at_coupon"] / 100, barrier_recall=d["at_recall"] / 100,
        barrier_capital=d["at_capital"] / 100, maturity=d["maturity"], step_down=d["at_step_down"] / 100,
        barrier_final=d["at_final"] / 100 if d.get("at_final") is not None else None,
    )
    market = Market(
        r=d["r"] / 100,
        sigma=tuple(s / 100 for _, s, _ in underlyings),
        q=tuple(q / 100 for _, _, q in underlyings),
        credit_spread=d["credit_spread"] / 100,
        correlation=d["correlation"] / 100 if len(underlyings) > 1 else 0.0,
        model=_heston_spec(d, d["model"]),
    )
    return phoenix, athena, market


def _cached_pricing(d: dict, underlyings: list):
    key = "pricing:" + hashlib.sha256(
        json.dumps({"v": ANALYSIS_VERSION, **d}, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]
    out = cache.get(key)
    if out is None:
        phoenix, athena, market = _build_inputs(d, underlyings)
        out = run_pricing(phoenix, athena, market, n_sim=d["n_sim"], seed=d["seed"],
                          antithetic=d["antithetic"], target_pct=d["target"],
                          heston_for_comparison=_heston_spec(d, "heston"))
        cache.set(key, out, 3600)
    return out


def build_context(out: dict, names: list[str]) -> dict:
    res, grk, fc = out["results"], out["greeks"], out["fair_coupons"]
    params, market = out["params"], out["market"]
    T = params["phoenix"].maturity
    keys = ("phoenix", "athena")
    worst_of = len(names) > 1
    underlying_label = "worst-of " + " / ".join(names) if worst_of else names[0]

    kpis = []
    for k in keys:
        r = res[k]
        kpis.append({
            "name": charts.NAMES[k], "color": charts.COLORS[k],
            "fv": pct(r["fair_value_pct"]), "ci": pct(r["ci95_pct"], 2),
            "coupon": pct(params[k].coupon_rate * 100, 2),
            "fair_coupon": pct(fc[k] * 100, 2),
            "recall": pct(r["early_recall_pct"], 1), "loss": pct(r["loss_pct"], 1),
            "duration": dec(r["avg_duration"]),
            "rich": r["fair_value_pct"] > 100,
        })

    summary = table(
        ["Indicateur", charts.NAMES["phoenix"], charts.NAMES["athena"]],
        [
            ["Fair value (% du nominal)", pct(res["phoenix"]["fair_value_pct"]), pct(res["athena"]["fair_value_pct"])],
            ["Intervalle de confiance 95 %", "± " + pct(res["phoenix"]["ci95_pct"], 3),
             "± " + pct(res["athena"]["ci95_pct"], 3)],
            ["Coupon proposé", pct(params["phoenix"].coupon_rate * 100), pct(params["athena"].coupon_rate * 100)],
            [f"Coupon d'équilibre (prix cible {dec(out['target_pct'], 1)} %)",
             pct(fc["phoenix"] * 100), pct(fc["athena"] * 100)],
            ["Rappel anticipé", pct(res["phoenix"]["early_recall_pct"], 1), pct(res["athena"]["early_recall_pct"], 1)],
            ["Perte en capital à maturité", pct(res["phoenix"]["loss_pct"], 1), pct(res["athena"]["loss_pct"], 1)],
            ["Durée de vie moyenne (années)", dec(res["phoenix"]["avg_duration"]), dec(res["athena"]["avg_duration"])],
            ["Nombre moyen de coupons versés (Phoenix)", dec(res["phoenix"]["avg_coupons"]), "—"],
            ["Nominal rendu sans coupon à maturité (Athena)", "—", pct(res["athena"]["capital_only_pct"], 1)],
        ],
    )

    scope = "tous les sous-jacents" if worst_of else "sous-jacent"
    greeks_table = table(
        ["Sensibilité (en points de % du nominal)", charts.NAMES["phoenix"], charts.NAMES["athena"]],
        [
            [f"Delta — spot +1 % ({scope})", num(grk["phoenix"]["delta"], 3), num(grk["athena"]["delta"], 3)],
            ["Gamma — variation du delta pour +1 %", num(grk["phoenix"]["gamma"], 3), num(grk["athena"]["gamma"], 3)],
            [f"Vega — volatilité +1 pt ({scope})", num(grk["phoenix"]["vega"], 3), num(grk["athena"]["vega"], 3)],
            ["Rho — taux +1 pt", num(grk["phoenix"]["rho"], 3), num(grk["athena"]["rho"], 3)],
            ["Spread de crédit +10 pb", num(grk["phoenix"]["spread_10bp"], 3),
             num(grk["athena"]["spread_10bp"], 3)],
        ],
    )

    # ── Modèle & skew ──
    mc = out["model_comparison"]
    smile = out["smile"]
    atm = smile["atm_vols"]
    atm_label = " / ".join(pct(v * 100, 1) for v in atm)
    input_label = " / ".join(pct(v * 100, 1) for v in market.sigmas)
    model_table = table(
        ["Fair value (% du nominal)", charts.NAMES["phoenix"], charts.NAMES["athena"]],
        [
            [f"Black-Scholes, vol saisie ({input_label})", pct(mc["phoenix"]["gbm_input"]),
             pct(mc["athena"]["gbm_input"])],
            [f"Black-Scholes, vol ATM équivalente ({atm_label})", pct(mc["phoenix"]["gbm_atm"]),
             pct(mc["athena"]["gbm_atm"])],
            ["Heston (même vol ATM, avec skew)", pct(mc["phoenix"]["heston"]), pct(mc["athena"]["heston"])],
            ["Effet du skew (Heston − BS ATM), en points", num(mc["phoenix"]["skew_effect"]),
             num(mc["athena"]["skew_effect"])],
        ],
    )
    spec = smile["heston_spec"]
    skew_90_110 = None
    strikes = list(smile["strikes"])
    if 0.9 in strikes and 1.1 in strikes:
        skew_90_110 = (smile["heston"][strikes.index(0.9)] - smile["heston"][strikes.index(1.1)]) * 100

    sensi = out["sensitivities"]
    mean_sigma = float(sum(market.sigmas) / market.n_assets)
    current = {
        "sigma": {"phoenix": mean_sigma, "athena": None},
        "r": {"phoenix": market.r, "athena": None},
        "barrier_capital": {"phoenix": params["phoenix"].barrier_capital, "athena": params["athena"].barrier_capital},
        "credit_spread": {"phoenix": market.credit_spread, "athena": None},
        "correlation": {"phoenix": market.correlation, "athena": None},
    }
    chart_map = {
        "payoffs": charts.payoff_histograms(res),
        "recalls": charts.recall_bars(res, T),
        "coupons": charts.coupon_probability(res["phoenix"], T),
        "smile": charts.smile_chart(smile, names[0]),
        "sensi_sigma": charts.sensitivity_chart("sigma", sensi["sigma"], current["sigma"]),
        "sensi_r": charts.sensitivity_chart("r", sensi["r"], current["r"]),
        "sensi_capital": charts.sensitivity_chart("barrier_capital", sensi["barrier_capital"],
                                                  current["barrier_capital"]),
        "sensi_spread": charts.sensitivity_chart("credit_spread", sensi["credit_spread"], current["credit_spread"]),
    }
    if worst_of:
        chart_map["sensi_corr"] = charts.sensitivity_chart("correlation", sensi["correlation"], current["correlation"])
        chart_map["worst_share"] = charts.worst_share_chart(out["worst_share"], names)

    model_name = ("Heston (volatilité stochastique)" if market.model.is_heston
                  else "Black-Scholes (volatilité constante)")
    return {
        "computed": True,
        "kpis": kpis,
        "maturity": T,
        "n_sim": out["n_sim"],
        "antithetic": out["antithetic"],
        "underlying_label": underlying_label,
        "worst_of": worst_of,
        "model_name": model_name,
        "is_heston": market.model.is_heston,
        "heston": {"kappa": dec(spec.kappa), "xi": dec(spec.xi), "rho": dec(spec.rho),
                   "term_ratio": dec(spec.term_ratio)},
        "skew_90_110": dec(skew_90_110, 1) if skew_90_110 is not None else None,
        "skew_effect": {k: num(mc[k]["skew_effect"]) for k in keys},
        "market": {
            "sigma": input_label, "q": " / ".join(pct(q * 100) for q in market.qs),
            "r": pct(market.r * 100), "spread": pct(market.credit_spread * 100),
            "correlation": pct(market.correlation * 100, 0),
        },
        "forward": [
            {"name": n, "sim": dec(f["simulated"], 4), "theo": dec(f["theoretical"], 4)}
            for n, f in zip(names, out["forward_check"])
        ],
        "tables": {"summary": summary, "greeks": greeks_table, "model": model_table},
        "charts": chart_map,
        "groups": {
            "paths": [(charts.NAMES[k], charts.paths_chart(k, params[k], out["paths_sample"],
                                                           out["paths_quantiles"], underlying_label))
                      for k in keys],
        },
    }


@require_GET
def index(request):
    if request.GET:
        form = PricingForm(request.GET)
    else:
        # Premier affichage : formulaire prérempli avec les données de marché du sous-jacent par défaut
        md = cached_market_data(DEFAULT_UNDERLYING)
        form = PricingForm(initial={"sigma_1": round(md["sigma"] * 100, 2), "r": round(md["r"] * 100, 2),
                                    "q_1": round(md["q"] * 100, 2), "term_ratio": md["term_ratio"]})

    context = {"form": form, "underlying_rows": [
        (form[f"underlying_{i}"], form[f"sigma_{i}"], form[f"q_{i}"]) for i in range(1, MAX_UNDERLYINGS + 1)
    ]}
    first = request.GET.get("underlying_1") or DEFAULT_UNDERLYING
    if first in INDICES:
        md = cached_market_data(first)
        context["market_sources"] = md["sources"]
        context["market_spot"] = md["spot"]

    if form.is_bound and form.is_valid():
        d = form.cleaned_data
        underlyings = form.underlyings()
        try:
            out = _cached_pricing(d, underlyings)
        except Exception as exc:  # noqa: BLE001 — erreur affichée à l'utilisateur
            context["error"] = str(exc)
        else:
            context.update(build_context(out, [n for n, _, _ in underlyings]))

    return render(request, "pricer/index.html", context)


@lru_cache(maxsize=1)
def _plotly_bundle() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


@require_GET
def plotly_js(request):
    response = HttpResponse(_plotly_bundle(), content_type="application/javascript")
    response["Cache-Control"] = "public, max-age=86400"
    return response
