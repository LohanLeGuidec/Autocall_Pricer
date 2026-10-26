from django import forms

from autocall_pricer import INDICES

UNDERLYING_CHOICES = [(k, k) for k in INDICES]
OPTIONAL_UNDERLYING = [("", "—")] + UNDERLYING_CHOICES
MAX_UNDERLYINGS = 3


def pct(label, initial, min_value, max_value, step="0.1", help_text="", required=True):
    """Champ saisi en pourcentage (converti en décimal dans la vue)."""
    return forms.FloatField(
        label=label, initial=initial, min_value=min_value, max_value=max_value, required=required,
        help_text=help_text, widget=forms.NumberInput(attrs={"step": step}),
    )


class PricingForm(forms.Form):
    # ── Sous-jacents (worst-of si plusieurs) ──
    underlying_1 = forms.ChoiceField(label="Sous-jacent 1", choices=UNDERLYING_CHOICES, initial="S&P 500")
    underlying_2 = forms.ChoiceField(label="Sous-jacent 2", choices=OPTIONAL_UNDERLYING, required=False)
    underlying_3 = forms.ChoiceField(label="Sous-jacent 3", choices=OPTIONAL_UNDERLYING, required=False)
    sigma_1 = pct("Vol (%)", 20.0, 3, 80)
    sigma_2 = pct("Vol (%)", None, 3, 80, required=False)
    sigma_3 = pct("Vol (%)", None, 3, 80, required=False)
    q_1 = pct("Div. (%)", 1.2, 0, 10, "0.01")
    q_2 = pct("Div. (%)", None, 0, 10, "0.01", required=False)
    q_3 = pct("Div. (%)", None, 0, 10, "0.01", required=False)
    correlation = pct("Corrélation (%)", 70, -50, 99, "1",
                      help_text="Entre chaque paire de sous-jacents (worst-of).")

    # ── Marché ──
    r = pct("Taux sans risque (%)", 3.0, -1, 10, "0.01")
    credit_spread = pct("Spread émetteur (%)", 0.8, 0, 5, "0.05",
                        help_text="Ajouté au taux d'actualisation : risque de défaut de la banque émettrice.")
    maturity = forms.IntegerField(label="Maturité (ans)", initial=5, min_value=2, max_value=10)
    target = pct("Prix cible (%)", 98.0, 80, 105, "0.5",
                 help_text="Pour le coupon d'équilibre : 100 % − marge de l'émetteur.")

    # ── Modèle ──
    model = forms.ChoiceField(label="Modèle principal", initial="heston",
                              help_text="L'autre modèle est toujours calculé pour comparaison.",
                              choices=[("gbm", "Black-Scholes (vol constante)"),
                                       ("heston", "Heston (vol stochastique)")])
    heston_kappa = forms.FloatField(label="κ (retour à la moyenne)", initial=1.5, min_value=0.05, max_value=10)
    heston_xi = forms.FloatField(label="ξ (vol de la variance)", initial=0.5, min_value=0.01, max_value=2)
    heston_rho = forms.FloatField(label="ρ (corrélation spot / vol)", initial=-0.7, min_value=-0.99, max_value=0.99)
    term_ratio = forms.FloatField(label="Vol long terme / court terme", initial=1.0, min_value=0.5, max_value=2.0,
                                  help_text="√θ / √v₀ : préremplie avec la structure par terme du VIX.")

    # ── Simulation ──
    n_sim = forms.TypedChoiceField(label="Simulations", coerce=int, initial=20000,
                                   choices=[(n, f"{n:,}".replace(",", " ")) for n in (5000, 10000, 20000, 50000)])
    seed = forms.IntegerField(label="Seed", initial=42, min_value=0, max_value=2**31 - 1)
    antithetic = forms.BooleanField(label="Variables antithétiques", initial=True, required=False)

    # ── Phoenix ──
    ph_coupon = pct("Coupon annuel (%)", 8, 0, 30)
    ph_recall = pct("Barrière rappel (%)", 100, 50, 130, "1")
    ph_coupon_barrier = pct("Barrière coupon (%)", 80, 30, 120, "1")
    ph_capital = pct("Barrière capital (%)", 60, 20, 100, "1")
    ph_step_down = pct("Step-down annuel (%)", 0, 0, 10, "0.5")
    ph_memory = forms.BooleanField(label="Effet mémoire", initial=True, required=False)

    # ── Athena ──
    at_coupon = pct("Coupon par année (%)", 10.0, 0, 30)
    at_recall = pct("Barrière rappel (%)", 100, 50, 130, "1")
    at_capital = pct("Barrière capital (%)", 60, 20, 100, "1")
    at_final = pct("Barrière coupon final (%)", None, 30, 130, "1", required=False,
                   help_text="Vide = barrière de rappel à maturité.")
    at_step_down = pct("Step-down annuel (%)", 0, 0, 10, "0.5")

    def underlyings(self) -> list[tuple[str, float, float]]:
        """[(nom, vol, dividende)] des sous-jacents renseignés, après validation."""
        d = self.cleaned_data
        out = []
        for i in range(1, MAX_UNDERLYINGS + 1):
            name = d.get(f"underlying_{i}")
            if name:
                out.append((name, d[f"sigma_{i}"], d.get(f"q_{i}") or 0.0))
        return out

    def clean(self):
        data = super().clean()
        names = []
        for i in range(1, MAX_UNDERLYINGS + 1):
            name = data.get(f"underlying_{i}")
            if not name:
                continue
            if name in names:
                self.add_error(f"underlying_{i}", "Ce sous-jacent est déjà sélectionné.")
            names.append(name)
            if data.get(f"sigma_{i}") is None and f"sigma_{i}" not in self.errors:
                self.add_error(f"sigma_{i}", "Volatilité requise.")

        for prefix, name in (("ph", "Phoenix"), ("at", "Athena")):
            recall, capital, step = (data.get(f"{prefix}_recall"), data.get(f"{prefix}_capital"),
                                     data.get(f"{prefix}_step_down"))
            maturity = data.get("maturity")
            if None in (recall, capital, step, maturity):
                continue
            if recall - step * (maturity - 1) < capital:
                self.add_error(f"{prefix}_step_down",
                               f"{name} : avec ce step-down, la barrière de rappel passe sous la barrière de capital.")
        cb, cap = data.get("ph_coupon_barrier"), data.get("ph_capital")
        if cb is not None and cap is not None and cb < cap:
            self.add_error("ph_coupon_barrier", "La barrière de coupon doit être au-dessus de la barrière de capital.")
        return data
