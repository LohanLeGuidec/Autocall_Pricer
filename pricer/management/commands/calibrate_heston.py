"""
Calibre le modèle de Heston et affiche les paramètres à saisir dans le pricer.

    python manage.py calibrate_heston --index "S&P 500"
    python manage.py calibrate_heston --csv surface_sx5e.csv --r 0.025

Le CSV (par exemple exporté de Bloomberg OVDV) contient les colonnes
maturity_years, strike_pct, implied_vol_pct et, en option, forward_pct.
"""
import numpy as np
from django.core.management.base import BaseCommand, CommandError

from autocall_pricer import INDICES, calibrate_heston, fetch_option_quotes, get_market_data, load_surface_csv


class Command(BaseCommand):
    help = "Calibre Heston sur des options cotées (Yahoo Finance) ou sur une surface importée (CSV)."

    def add_arguments(self, parser):
        parser.add_argument("--index", default="S&P 500", choices=list(INDICES))
        parser.add_argument("--csv", help="Surface de volatilité à importer à la place des données Yahoo.")
        parser.add_argument("--r", type=float, help="Taux sans risque (décimal). Par défaut : celui du pricer.")

    def handle(self, *args, **opts):
        r = opts["r"] if opts["r"] is not None else get_market_data(opts["index"])["r"]
        try:
            quotes = load_surface_csv(opts["csv"], r) if opts["csv"] else fetch_option_quotes(opts["index"], r)
            res = calibrate_heston(quotes)
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        s = res.summary()
        out = self.stdout.write
        out(f"Source : {s['source']} · {s['n_options']} options · taux {r:.2%}")
        out(f"Erreur de calibration : {s['rmse_vol_pts']} pt de vol en moyenne (max {s['max_error_vol_pts']} pt)")
        out("")
        out("Paramètres à saisir dans le pricer :")
        out(f"  Vol (%)                       {s['sigma0'] * 100:.2f}   (√v₀)")
        out(f"  κ (retour à la moyenne)       {s['kappa']}")
        out(f"  ξ (vol de la variance)        {s['xi']}")
        out(f"  ρ (corrélation spot / vol)    {s['rho']}")
        out(f"  Vol long terme / court terme  {s['term_ratio']}   (√θ / √v₀)")
        out(f"  Condition de Feller (2κθ ≥ ξ²) : {'respectée' if s['feller'] else 'non respectée'}")
        out("")
        out("Ajustement par maturité (vol marché → vol Heston, en %) :")
        q = res.quotes
        for T in np.unique(np.round(q.T, 3)):
            m = np.round(q.T, 3) == T
            k = q.strike[m] / q.forward[m]
            idx = [int(np.argmin(np.abs(k - target))) for target in (0.9, 1.0, 1.1)]
            cells = "  ".join(f"K/F={k[i]:.2f}: {q.vol[m][i] * 100:5.1f} → {res.model_vols[m][i] * 100:5.1f}"
                              for i in idx)
            out(f"  T={T:4.2f} an   {cells}")
