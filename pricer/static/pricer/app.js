// Autocall Pricer — interactions côté navigateur (sans dépendance hormis Plotly.js)
(function () {
  "use strict";

  const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  // ── Graphiques : rendu à la demande (seulement quand ils sont visibles) ──
  function renderVisibleCharts(root) {
    (root || document).querySelectorAll(".chart:not([data-rendered])").forEach((el) => {
      if (el.offsetParent === null) return; // masqué : rendu quand il deviendra visible
      const script = el.querySelector('script[type="application/json"]');
      if (!script || !window.Plotly) return;
      const fig = JSON.parse(script.textContent);
      el.dataset.rendered = "1";
      // Hauteur explicite : sinon, au redimensionnement de la fenêtre, Plotly (responsive)
      // relit la hauteur du conteneur, la trouve nulle et le graphique déborde de sa carte.
      el.style.height = `${(fig.layout && fig.layout.height) || 400}px`;
      Plotly.newPlot(el, fig.data, fig.layout, PLOT_CONFIG);
    });
  }

  function resizeCharts(root) {
    (root || document).querySelectorAll(".chart[data-rendered]").forEach((el) => {
      if (el.offsetParent !== null) Plotly.Plots.resize(el);
    });
  }

  // ── Onglets ──
  function initTabs() {
    const tabs = document.querySelectorAll(".tabs [data-tab]");
    tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabs.forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.hidden = panel.dataset.panel !== btn.dataset.tab;
        });
        const panel = document.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`);
        renderVisibleCharts(panel);
        resizeCharts(panel);
        try { sessionStorage.setItem("autocall-tab", btn.dataset.tab); } catch (e) { /* stockage indisponible */ }
      });
    });
    let saved = null;
    try { saved = sessionStorage.getItem("autocall-tab"); } catch (e) { /* stockage indisponible */ }
    const target = saved && document.querySelector(`.tabs [data-tab="${saved}"]`);
    if (target) target.click();
  }

  // ── Sélecteur de produit dans un groupe de graphiques ──
  function initGroups() {
    document.querySelectorAll("[data-group-select]").forEach((select) => {
      select.addEventListener("change", () => {
        const group = select.closest(".chart-group");
        group.querySelectorAll(".group-item").forEach((item) => {
          item.hidden = item.dataset.index !== select.value;
        });
        renderVisibleCharts(group);
        resizeCharts(group);
      });
    });
  }

  // ── Écran de chargement pendant l'analyse ──
  function initLoading() {
    const form = document.getElementById("analysis-form");
    const loading = document.getElementById("loading");
    if (!form || !loading) return;
    form.addEventListener("submit", () => {
      try { sessionStorage.removeItem("autocall-tab"); } catch (e) { /* stockage indisponible */ }
      loading.hidden = false;
    });
    // Retour arrière du navigateur : ne pas rester bloqué sur l'écran de chargement
    window.addEventListener("pageshow", () => { loading.hidden = true; });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initGroups();
    initLoading();
    renderVisibleCharts();
  });
})();

// ── Sous-jacents : données de marché, corrélation, choix du modèle ──
(function () {
  "use strict";
  const fmt = (v) => Number(v).toLocaleString("fr-FR", { maximumFractionDigits: 2 });

  document.addEventListener("DOMContentLoaded", () => {
    const box = document.querySelector(".underlyings");
    const hint = document.getElementById("market-hint");
    if (!box) return;
    const selects = [1, 2, 3].map((i) => document.getElementById(`id_underlying_${i}`));
    const corrField = document.getElementById("corr-field");
    const corrHint = document.getElementById("corr-hint");
    const setValue = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };

    async function loadMarket(i) {
      const name = selects[i - 1].value;
      if (!name) { setValue(`id_sigma_${i}`, ""); setValue(`id_q_${i}`, ""); return; }
      hint.textContent = `Chargement des données de marché (${name})…`;
      try {
        const res = await fetch(`${box.dataset.marketUrl}?index=${encodeURIComponent(name)}`);
        if (!res.ok) throw new Error(res.status);
        const d = await res.json();
        setValue(`id_sigma_${i}`, d.sigma);
        setValue(`id_q_${i}`, d.q);
        if (i === 1) {  // le taux et la structure par terme suivent le premier sous-jacent
          setValue("id_r", d.r);
          setValue("id_term_ratio", d.term_ratio);
        }
        hint.innerHTML = `${name} : spot <strong>${fmt(d.spot)}</strong> · vol : ${d.sources.sigma} · structure par terme : ${d.sources.term_ratio}` +
          (i === 1 ? ` · taux : ${d.sources.r}` : "");
      } catch (e) {
        hint.textContent = "Données de marché indisponibles : saisissez les paramètres manuellement.";
      }
    }

    async function loadCorrelation() {
      const names = selects.map((s) => s.value).filter(Boolean);
      corrField.hidden = names.length < 2;
      if (names.length < 2 || new Set(names).size !== names.length) return;
      corrHint.textContent = "Estimation de la corrélation…";
      try {
        const res = await fetch(`${box.dataset.corrUrl}?indices=${encodeURIComponent(names.join(","))}`);
        if (!res.ok) throw new Error(res.status);
        const d = await res.json();
        setValue("id_correlation", d.correlation);
        corrHint.textContent = `Corrélation moyenne ${d.correlation} % — ${d.source}`;
      } catch (e) {
        corrHint.textContent = "Corrélation indisponible : saisissez-la manuellement.";
      }
    }

    selects.forEach((select, idx) => {
      if (!select) return;
      select.addEventListener("change", () => { loadMarket(idx + 1); loadCorrelation(); });
    });

    const model = document.getElementById("id_model");
    const heston = document.getElementById("heston-params");
    if (model && heston) {
      model.addEventListener("change", () => { heston.hidden = model.value !== "heston"; });
    }

    // Calibration de Heston sur les options cotées du premier sous-jacent
    const calibBtn = document.getElementById("calibrate-btn");
    const calibHint = document.getElementById("calib-hint");
    if (calibBtn) {
      calibBtn.addEventListener("click", async () => {
        const name = selects[0].value;
        calibBtn.disabled = true;
        calibHint.textContent = `Calibration sur les options ${name}…`;
        try {
          const res = await fetch(`${calibBtn.dataset.calibUrl}?index=${encodeURIComponent(name)}`);
          const d = await res.json();
          if (!res.ok) throw new Error(d.error || res.status);
          setValue("id_sigma_1", d.sigma);
          setValue("id_heston_kappa", d.kappa);
          setValue("id_heston_xi", d.xi);
          setValue("id_heston_rho", d.rho);
          setValue("id_term_ratio", d.term_ratio);
          calibHint.innerHTML = `Calibré sur <strong>${d.n_options}</strong> options · erreur moyenne <strong>${fmt(d.rmse)} pt</strong> de vol` +
            ` · Feller ${d.feller ? "respectée" : "non respectée"} · ${d.source}`;
        } catch (e) {
          calibHint.textContent = `Calibration impossible : ${e.message}`;
        } finally {
          calibBtn.disabled = false;
        }
      });
    }
  });
})();
