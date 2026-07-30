"""Step 5: figures computed from the pipeline outputs, drawn to a single
typographic standard.

Produced here:
  Fig. 2  frequency-magnitude distribution with 95 % band and b(Mc) inset
  Fig. 4  training and validation loss curves at both window lengths
  Fig. 5  Molchan error diagram and ROC curves at the pre-specified threshold,
          with a pointwise 95 % moving-block bootstrap band for the
          highest-scoring architecture and the three catalog-derived ranking
          baselines
  Fig. 6  baseline and deep-model AUC comparison at FIG6_THRESHOLD
  Fig. 7  Integrated-Gradients temporal profiles at both window lengths

The seismotectonic map and the architecture schematic are not data products and
are prepared separately at the same resolution and in the same serif font.

Output:  outputs/figures/*.png, *.pdf
         outputs/figures/figure_values.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zm import config as C
from zm import catalog as cat
from zm import baselines as B
from zm import stats as S

# secondary threshold. Both are declared here rather than buried in the code.
FIG5_THRESHOLD = C.PRIMARY_THRESHOLD          # 4.0
FIG6_THRESHOLD = C.THRESHOLDS[-1]             # 4.5
FIG5_WINDOW = C.WINDOW_SIZES[0]               # 20
BAND_RESAMPLES = min(C.MBB_B, 600)            # curve bands: 4,000 is needlessly slow

ARCH_STYLE = {
    "LSTM":        dict(color="tab:blue",  ls=":",  marker=None),
    "GRU":         dict(color="tab:green", ls="-.", marker=None),
    "TCN":         dict(color="tab:red",   ls="-",  marker=None),
    "Transformer": dict(color="0.35",      ls="--", marker=None),
}
BASE_STYLE = dict(color="0.6", ls="-", lw=1.2, alpha=0.9)

VALUES = {}          # everything plotted, dumped to figure_values.json


def setup_style():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    fam = C.FONT_FAMILY if C.FONT_FAMILY in avail else C.FONT_FALLBACK
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [fam, "DejaVu Serif"],
        "font.size": C.FONT_SIZE,
        "axes.linewidth": 1.0,
        "lines.linewidth": C.LINE_WIDTH,
        "savefig.dpi": C.DPI,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })
    if fam != C.FONT_FAMILY:
        print("   !! '{}' not installed; using '{}' for EVERY figure so the set "
              "stays typographically uniform.".format(C.FONT_FAMILY, fam))
    print("   font in use: " + fam)
    VALUES["font"] = fam
    VALUES["dpi"] = C.DPI
    return fam


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG_DIR, name + "." + ext), dpi=C.DPI)
    plt.close(fig)
    print("   saved " + name)


def load(name):
    p = os.path.join(C.OUT_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mbb_indices(n, block, rng):
    """One moving-block bootstrap resample of 0..n-1."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


# ------------------------------------------------------------------ Figure 2
def figure_fmd(df, crep):
    mc = float(crep["mc_used"])
    bf = crep["b_value_full"]
    b, sb, a = bf["b"], bf["sigma_b"], bf["a"]
    m = df["MAG"].to_numpy(float)

    bins = np.round(np.arange(m.min(), m.max() + C.DM, C.DM), 5)
    non_cum = np.array([np.sum(np.abs(m - x) < C.DM / 2) for x in bins], float)
    cum = np.array([np.sum(m >= x - C.DM / 2) for x in bins], float)

    stab = crep["completeness"].get("b_stability") or []
    if stab:
        fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.6, 5.2))
    else:
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        axb = None
    ax.semilogy(bins, cum, "o", ms=5, mfc="none", color="black", label="Cumulative")
    ax.semilogy(bins, np.where(non_cum > 0, non_cum, np.nan), "s", ms=4,
                color="0.55", label="Non-cumulative")

    xf = np.linspace(mc, m.max(), 100)
    ax.semilogy(xf, 10 ** (a - b * xf), "r--", lw=2.2,
                label="Gutenberg-Richter fit, b = {:.3f} $\\pm$ {:.3f}".format(b, sb))
    ax.fill_between(xf, 10 ** (a - (b + 1.96 * sb) * xf),
                    10 ** (a - (b - 1.96 * sb) * xf),
                    color="red", alpha=0.18, lw=0, label="95 % confidence band")
    ax.axvline(mc, color="blue", ls="--", lw=1.8, label="$M_c$ = {:.2f}".format(mc))

    # Shade the incomplete part: a reader cannot judge Mc without seeing it.
    ax.axvspan(float(bins.min()) - C.DM, mc, color="0.80", alpha=0.30, lw=0,
               label="Incomplete (below $M_c$)")
    ax.set_xlabel("Magnitude (reported scale, mixed ML / mb)"
                  if C.MAG_HOMOGENISATION == "none"
                  else "Moment magnitude $M_w$ (converted)")
    ax.set_ylabel("Number of events")
    ax.set_ylim(0.7, cum.max() * 2)
    brep = crep.get("b_value_reporting")
    if brep and abs(float(brep["mc"]) - mc) > 1e-9:
        xr = np.linspace(float(brep["mc"]), m.max(), 100)
        ax.semilogy(xr, 10 ** (brep["a"] - brep["b"] * xr),
                    color="tab:green", ls="-", lw=2.0,
                    label=("GR at reporting $M_c$ = {:.2f}, b = {:.3f} $\\pm$ {:.3f}"
                           .format(float(brep["mc"]), brep["b"], brep["sigma_b"])))
        ax.axvline(float(brep["mc"]), color="tab:green", ls="-.", lw=1.5)
    ax.legend(loc="lower left", fontsize=9)

    if axb is not None:
        ax.set_title("(a) Frequency-magnitude distribution")
        mcs = np.array([q["mc"] for q in stab], float)
        bvs = np.array([q["b"] for q in stab], float)
        sbv = np.array([q["sigma_b"] for q in stab], float)
        axb.errorbar(mcs, bvs, yerr=sbv, fmt="ko-", ms=4, lw=1.2, capsize=3,
                     label="b($M_c$) $\\pm$ 1$\\sigma$ (Aki-Utsu)")
        axb.axvline(mc, color="blue", ls="--", lw=1.8,
                    label="adopted $M_c$ = {:.2f}".format(mc))
        axb.axhline(b, color="red", ls="--", lw=1.6,
                    label="adopted b = {:.3f}".format(b))
        for key, col in (("MAXC", "tab:orange"), ("GFT", "tab:purple"),
                         ("EMR", "tab:brown")):
            v = (crep["completeness"].get(key) or {}).get("mc")
            if v is not None:
                axb.axvline(float(v), color=col, ls=":", lw=1.4,
                            label="{} = {:.2f}".format(key, float(v)))
        axb.set_xlabel("Cut-off magnitude $M_c$")
        axb.set_ylabel("b-value")
        axb.set_title("(b) Stability of b against the cut-off")
        axb.grid(alpha=0.3)
        bst = crep["completeness"].get("B_STABILITY") or {}
        if bst.get("mc") is not None:
            m0 = float(bst["mc"])
            wdt = float(bst.get("window") or 0.5)
            axb.axvspan(m0, m0 + wdt, color="tab:green", alpha=0.15, lw=0,
                        label="stability window ({:.1f} mag units)".format(wdt))
            axb.axvline(m0, color="tab:green", ls="-", lw=2.0,
                        label="reporting $M_c$ = {:.2f}, b = {:.3f}"
                              .format(m0, float(bst["b"])))
        elif bst.get("closest_mc") is not None:
            # The Wiemer & Wyss (2000) criterion is met nowhere. Say so ON the
            # figure and mark the cut-off that comes closest, because that is
            # invites the reader to eyeball a plateau that is not there.
            m0 = float(bst["closest_mc"])
            wdt = float(bst.get("window") or 0.5)
            dev = float(bst["closest_dev_sigma"])
            axb.axvspan(m0, m0 + wdt, color="tab:green", alpha=0.12, lw=0,
                        label="closest approach to b-stability: "
                              "$M_c$ = {:.2f}, drift {:.2f}$\\sigma$ over {:.1f} "
                              "mag (criterion needs $\\leq$ 1$\\sigma$)"
                              .format(m0, dev, wdt))
            axb.axvline(m0, color="tab:green", ls="-.", lw=1.6)
            axb.text(0.98, 0.04,
                     "b-stability criterion (Wiemer & Wyss 2000)\n"
                     "not satisfied for any $M_c$ in [{:.1f}, {:.1f}]"
                     .format(*(bst.get("mc_range") or [0.0, 0.0])),
                     transform=axb.transAxes, ha="right", va="bottom",
                     fontsize=8, color="tab:green",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec="tab:green", alpha=0.85))
        axb.legend(fontsize=8, loc="upper left")

    cm = crep["completeness"]
    VALUES["Fig2"] = {
        "mc_adopted": mc, "mc_maxc_raw": cm["MAXC"]["mc_raw"],
        "mc_maxc_corrected": cm["MAXC"]["mc"], "mc_gft": cm["GFT"]["mc"],
        "mc_emr": cm["EMR"]["mc"], "b": b, "sigma_b": sb, "a": a,
        "n_events_above_mc": int((m >= mc - 1e-9).sum()),
        "caption_note": ("Mc from three methods (MAXC+{:.1f}, GFT, EMR); adopted "
                             "Mc = {:.2f}; dashed red line is the MLE fit and the shaded "
                             "band is its 95 % confidence band; panel (b) shows b against Mc."
                             .format(cm["MAXC"]["correction"], mc)),
        "b_stability": {
            "mc": (crep["completeness"].get("B_STABILITY") or {}).get("mc"),
            "closest_mc": (crep["completeness"].get("B_STABILITY")
                           or {}).get("closest_mc"),
            "closest_dev_sigma": (crep["completeness"].get("B_STABILITY")
                                  or {}).get("closest_dev_sigma"),
            "b_range": (crep["completeness"].get("B_STABILITY")
                        or {}).get("b_range"),
            "monotonic_increasing": (crep["completeness"].get("B_STABILITY")
                                     or {}).get("monotonic_increasing"),
        },
    }
    save(fig, "Fig2_FMD")


# ------------------------------------------------------------------ Figure 4
def figure_loss(trep):
    runs = trep["runs"]
    ws = sorted({r["w"] for r in runs})
    budget = sorted({r["budget"] for r in runs})[-1]
    fig, axes = plt.subplots(1, len(ws), figsize=(6.2 * len(ws), 4.8), squeeze=False)
    VALUES["Fig4"] = {}

    for j, w in enumerate(ws):
        ax = axes[0][j]
        best_losses, stopped_early = {}, {}
        for arch in C.ARCHITECTURES:
            sel = [r for r in runs if r["w"] == w and r["arch"] == arch
                   and r["budget"] == budget]
            if not sel:
                continue
            st = ARCH_STYLE[arch]
            L = min(len(r["history_val_loss"]) for r in sel)
            V = np.array([r["history_val_loss"][:L] for r in sel])
            T = np.array([r["history_loss"][:L] for r in sel])
            ep = np.arange(2, L + 1)          # epoch 1 omitted: initialisation transient
            mv, sv = V[:, 1:].mean(0), V[:, 1:].std(0)
            ax.plot(ep, mv, color=st["color"], ls=st["ls"], lw=2.2,
                    label=arch + " (val)")
            ax.fill_between(ep, mv - sv, mv + sv, color=st["color"], alpha=0.15, lw=0)
            ax.plot(ep, T[:, 1:].mean(0), color=st["color"], ls=":", lw=1.0, alpha=0.7)
            # One tick per seed. Averaging seeds that stopped at epoch 8 and at
            # epoch 1 prints "4.5", which describes neither run.
            for r in sel:
                ax.axvline(float(r["best_epoch"]), color=st["color"], ls="--",
                           lw=0.9, alpha=0.45)
            best_losses[arch] = float(np.mean([r["val_loss_at_best"] for r in sel]))
            stopped_early[arch] = bool(np.mean([r["epochs_run"] for r in sel]) < C.EPOCHS)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE, magnitude$^2$)")
        ax.set_title("({}) w = {}, nominal budget {:,} parameters"
                     .format("ab"[j], w, budget))
        ax.legend(fontsize=9, ncol=2)
        if best_losses:
            spread = max(best_losses.values()) - min(best_losses.values())
            VALUES["Fig4"]["w{}".format(w)] = {
                "mean_best_val_loss": best_losses,
                "inter_architecture_spread_magnitude_squared": round(spread, 4),
                "early_stopped_before_epoch_limit": stopped_early,
                "epoch_limit": C.EPOCHS,
            }
    fig.suptitle("Colour and line style identify the ARCHITECTURE. Thick line = "
                 "validation (mean $\\pm$ 1 SD over {} seeds); thin dotted line = "
                 "training; vertical dashes = best epoch of each seed; epoch 1 "
                 "omitted".format(C.N_SEEDS), fontsize=9, y=1.02)
    save(fig, "Fig4_loss_curves")


# ------------------------------------------------------------------ Figure 5
def figure_molchan_roc(Z, res, M_all, t_all):
    w = FIG5_WINDOW
    thr = FIG5_THRESHOLD
    wk, tk = "w{}".format(w), "M>={}".format(thr)
    if wk not in res["windows"]:
        print("   Fig5 skipped: no results for " + wk)
        return
    budget = C.PARAM_BUDGETS[-1]
    W = res["windows"][wk]

    y_test = Z["y_test_w{}".format(w)]
    idx_test = Z["idx_test_w{}".format(w)]
    yb = (y_test >= thr).astype(int)
    n_pos, n_tot = int(yb.sum()), int(yb.size)
    if n_pos < 5:
        print("   Fig5 skipped: only {} positives".format(n_pos))
        return
    pi = n_pos / n_tot

    scores = {}
    for arch in C.ARCHITECTURES:
        k = "pred_test_{}_w{}_p{}".format(arch, w, budget)
        if k in Z:
            scores[arch] = Z[k].mean(axis=0)          # ensemble mean prediction
    rank = B.ranking_baselines(M_all, t_all, idx_test, w)
    for nm in B.FIG5_BASELINES:
        if nm in rank:
            scores[B.DISPLAY[nm]] = rank[nm]

    arch_auc = {a: roc_auc_score(yb, s) for a, s in scores.items()
                if a in C.ARCHITECTURES}
    # The band follows the model chosen on the VALIDATION split, never the one
    # that happens to score highest on TEST. Choosing it by test AUC would
    # disagree with Fig. 6, which marks the validation choice.
    top, top_basis = None, "validation"
    sel_key = (W.get("model_selection") or {}).get("selected")
    if sel_key:
        cand = str(sel_key).split("_p")[0]
        if cand in scores:
            top = cand
    if top is None and arch_auc:
        top = max(arch_auc, key=arch_auc.get)
        top_basis = "test (FALLBACK - no validation selection found)"
        print("   !! Fig5: no validation selection in results.json; the band is "
              "drawn for the test-best model, which is not a valid selection "
              "rule.")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.0, 5.8))
    VALUES.setdefault("Fig5", {})["band_model"] = top
    VALUES["Fig5"]["band_selection_basis"] = top_basis

    # ---------------- pointwise 95 % MBB band for the top architecture
    if top is not None:
        rng = np.random.default_rng(C.RNG_SEED)
        tau_grid = np.linspace(0.0, 1.0, 101)
        fpr_grid = np.linspace(0.0, 1.0, 101)
        nu_bs, tpr_bs = [], []
        s_top = scores[top]
        for _ in range(BAND_RESAMPLES):
            ii = mbb_indices(n_tot, C.MBB_BLOCK, rng)
            yy, ss = yb[ii], s_top[ii]
            if yy.sum() == 0 or yy.sum() == yy.size:
                continue
            m = S.molchan(yy, ss)
            nu_bs.append(np.interp(tau_grid, m["tau"], m["nu"]))
            fp, tp, _ = roc_curve(yy, ss)
            tpr_bs.append(np.interp(fpr_grid, fp, tp))
        if nu_bs:
            nu_bs = np.array(nu_bs)
            axa.fill_between(tau_grid, np.percentile(nu_bs, 2.5, axis=0),
                             np.percentile(nu_bs, 97.5, axis=0),
                             color="tab:blue", alpha=0.20, lw=0,
                             label="95 % MBB band ({}, selected on validation)".format(top))
        if tpr_bs:
            tpr_bs = np.array(tpr_bs)
            axb.fill_between(fpr_grid, np.percentile(tpr_bs, 2.5, axis=0),
                             np.percentile(tpr_bs, 97.5, axis=0),
                             color="tab:blue", alpha=0.20, lw=0,
                             label="95 % MBB band ({}, selected on validation)".format(top))

    # ---------------- panel (a) Molchan
    axa.fill_between([0, 1], [1, 0], [0, 0], color="green", alpha=0.07, lw=0,
                     label="Zone of positive skill")
    axa.plot([0, 1], [1, 0], color="0.35", ls="--", lw=1.6,
             label=r"No skill ($\nu = 1 - \tau$)")
    fig5 = {"window": w, "threshold": thr, "n_pos": n_pos, "n_test": n_tot,
            "base_rate_pi": round(pi, 4), "n_seeds": C.N_SEEDS,
            "band_architecture": top, "band_resamples": BAND_RESAMPLES,
            "mbb_block": C.MBB_BLOCK, "curves": {}}

    for nm, s in scores.items():
        st = ARCH_STYLE.get(nm)
        kw = (dict(color=st["color"], ls=st["ls"], lw=C.LINE_WIDTH) if st
              else dict(**BASE_STYLE))
        if st and nm == top:
            kw["lw"] = C.LINE_WIDTH * 1.9
            kw["zorder"] = 5
        mol = S.molchan(yb, s)
        auc = float(roc_auc_score(yb, s))
        stats = (W["models"].get("{}_p{}".format(nm, budget), {})
                 .get("thresholds", {}).get(tk, {})) if st else {}
        p_cs = stats.get("p_circular_shift")
        axa.plot(mol["tau"], mol["nu"],
                 label="{}  S = {:+.3f}".format(nm, mol["S"]), **kw)
        fp, tp, _ = roc_curve(yb, s)
        lab = "{}  AUC = {:.3f}".format(nm, auc)
        if p_cs is not None:
            lab += ", p = {:.3f}".format(p_cs)
        axb.plot(fp, tp, label=lab, **kw)
        ident = S.molchan_auc_identity(mol["S"], auc, pi)
        fig5["curves"][nm] = {
            "molchan_S": mol["S"], "AUC": auc, "p_circular_shift": p_cs,
            "identity_S_from_AUC": ident["S_from_identity"],
            "identity_consistent": ident["consistent"],
            "is_architecture": st is not None,
        }

    axa.set_xlabel(r"Alarm fraction $\tau$")
    axa.set_ylabel(r"Miss fraction $\nu$")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    axa.set_title("(a) Molchan error diagram")
    axa.legend(fontsize=8, loc="upper right")

    axb.plot([0, 1], [0, 1], color="0.35", ls="--", lw=1.6, label="No skill")
    axb.set_xlabel("False positive rate")
    axb.set_ylabel("True positive rate")
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.set_title("(b) Receiver operating characteristic")
    axb.legend(fontsize=8, loc="lower right")

    fig.suptitle("M $\\geq$ {} (n$^+$ = {} of {} test windows, base rate {:.1%}); "
                 "w = {}; N = {} seeds per architecture"
                 .format(thr, n_pos, n_tot, pi, w, C.N_SEEDS), fontsize=10, y=1.00)
    VALUES["Fig5"] = fig5
    save(fig, "Fig5_molchan_roc")


# ------------------------------------------------------------------ Figure 6
def figure_baselines(Z, res, M_all, t_all):
    thr = FIG6_THRESHOLD
    tk = "M>={}".format(thr)
    w = FIG5_WINDOW
    wk = "w{}".format(w)
    if wk not in res["windows"]:
        return
    W = res["windows"][wk]
    budget = C.PARAM_BUDGETS[-1]

    y_test = Z["y_test_w{}".format(w)]
    idx_test = Z["idx_test_w{}".format(w)]
    yb = (y_test >= thr).astype(int)
    n_pos = int(yb.sum())
    if n_pos < 5:
        print("   Fig6 skipped: only {} positives at M>={}".format(n_pos, thr))
        return

    # deep model chosen on VALIDATION, never on the test set
    selected = W["model_selection"]["selected"]
    sel_arch = selected.split("_p")[0]
    sel = (W["models"].get(selected, {}).get("thresholds", {}).get(tk, {}))
    sel_auc = sel.get("AUC_per_seed_mean")
    sel_sd = sel.get("AUC_per_seed_sd")

    rows = []
    rank = B.ranking_baselines(M_all, t_all, idx_test, w)
    for nm, s in rank.items():
        rows.append((B.DISPLAY[nm], float(roc_auc_score(yb, s)), "0.6"))
    ba = W.get("baseline_auc", {}).get(tk, {}) or {}
    if "etas_conditional_intensity" in ba:
        rows.append((B.DISPLAY["etas_conditional_intensity"],
                     float(ba["etas_conditional_intensity"]["AUC"]), "0.4"))
    for arch in C.ARCHITECTURES:
        t = (W["models"].get("{}_p{}".format(arch, budget), {})
             .get("thresholds", {}).get(tk, {}))
        if "AUC_per_seed_mean" in t:
            rows.append((arch, float(t["AUC_per_seed_mean"]),
                         ARCH_STYLE[arch]["color"]))

    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    cols = [r[2] for r in rows]

    # standard error of AUC (Hanley & McNeil style approximation), for honesty
    n_neg = int(yb.size - n_pos)
    se = float(np.sqrt(0.25 / n_pos + 0.25 / n_neg))

    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(names) + 2.4))
    ypos = np.arange(len(names))
    ax.barh(ypos, vals, color=cols, alpha=0.85, height=0.62)
    ax.errorbar(vals, ypos, xerr=se, fmt="none", ecolor="black", capsize=3, lw=1.0)
    ax.axvline(0.5, color="0.35", ls=":", lw=1.6, label="No skill (AUC = 0.50)")
    if sel_auc is not None:
        ax.axvline(sel_auc, color="red", ls="--", lw=1.8,
                   label="{} (selected on validation), AUC = {:.3f}"
                         .format(sel_arch, sel_auc))
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ROC-AUC")
    # Never let a bar fall off the axis: "recent seismicity rate" sits well
    ax.set_xlim(min(0.35, float(vals.min()) - se - 0.02),
                max(0.72, float(vals.max()) + se + 0.03))
    ax.set_title("M $\\geq$ {}  (n$^+$ = {} of {}); error bars are $\\pm$1 SE $\\approx$ {:.3f}"
                 .format(thr, n_pos, yb.size, se), fontsize=11)
    ax.legend(fontsize=9, loc="lower right")

    VALUES["Fig6"] = {
        "window": w, "threshold": thr, "n_pos": n_pos, "n_test": int(yb.size),
        "auc_standard_error": round(se, 4),
        "selected_model": selected, "selected_auc_mean": sel_auc,
        "selected_auc_sd": sel_sd,
        "values": {n: round(v, 4) for n, v in zip(names, vals)},
        "baseline_definitions": B.DEFINITIONS,
        "caption_note": ("AUC values at this threshold carry substantial sampling "
                             "uncertainty (SE ~ {:.2f}) because the positive test class "
                             "contains only {} events.".format(se, n_pos)),
    }
    save(fig, "Fig6_baseline_comparison")


# ------------------------------------------------------------------ Figure 7
def figure_ig(igrep):
    budget = C.PARAM_BUDGETS[-1]
    ws = C.WINDOW_SIZES
    fig, axes = plt.subplots(1, len(ws), figsize=(6.6 * len(ws), 4.8), squeeze=False)
    VALUES["Fig7"] = {"budget": budget, "panels": {}}

    for j, w in enumerate(ws):
        ax = axes[0][j]
        panel = {"architectures_shown": [], "missing": []}
        env = []
        for arch in C.ARCHITECTURES:
            r = igrep["models"].get("{}_w{}_p{}".format(arch, w, budget))
            if not r or r.get("status") != "ok":
                panel["missing"].append(arch)
                continue
            prof = np.asarray(r["temporal_profile_normalised"], float)
            st = ARCH_STYLE[arch]
            ax.plot(np.arange(len(prof)), prof, color=st["color"], ls=st["ls"],
                    label="{}  (mem50 = {})".format(
                        arch, r["effective_memory_steps_50pct"]))
            panel["architectures_shown"].append(arch)
            panel[arch] = {
                "effective_memory_steps_50pct": r["effective_memory_steps_50pct"],
                "effective_memory_steps_90pct": r["effective_memory_steps_90pct"],
                "attribution_share_last_10_steps": r["attribution_share_last_10_steps"],
                "feature_ranking": r["feature_ranking"],
            }
            env.append(prof)

        # highlight the recent interval that actually carries the attribution,
        # computed from the data instead of hard-coded as "steps 40-49"
        if env:
            mean_prof = np.mean(env, axis=0)
            cum = np.cumsum(mean_prof[::-1]) / mean_prof.sum()
            k = int(np.searchsorted(cum, 0.50) + 1)          # newest k steps hold 50 %
            lo = len(mean_prof) - k
            ax.axvspan(lo - 0.5, len(mean_prof) - 0.5, color="gold", alpha=0.20, lw=0,
                       label="steps {}-{}: 50 % of the across-architecture "
                             "MEAN attribution ({} models)"
                             .format(lo, len(mean_prof) - 1, len(env)))
            panel["highlight_steps"] = [lo, len(mean_prof) - 1]
            panel["highlight_n_events"] = k

        ax.set_xlabel("Position in input window (0 = oldest, {} = most recent)".format(w - 1))
        ax.set_ylabel("Normalised mean $|$IG$|$ attribution")
        ax.set_title("({}) w = {}".format("ab"[j], w))
        if panel["architectures_shown"]:
            ax.legend(fontsize=9)
        if panel["missing"]:
            print("   !! Fig7 panel w={}: missing {} - the caption cannot report "
                  "they are shown".format(w, panel["missing"]))
        VALUES["Fig7"]["panels"]["w{}".format(w)] = panel

    save(fig, "Fig7_integrated_gradients")


def main():
    print("Config: " + C.summary())
    setup_style()

    crep = load("catalog_report.json")
    trep = load("training_report.json")
    res = load("results.json")
    igrep = load("ig_report.json")

    M_all = t_all = None
    if crep:
        df_full = cat.load_clean_catalog()
        df = df_full[df_full["MAG"] >= float(crep["mc_used"]) - 1e-9]
        df = df.sort_values("datetime").reset_index(drop=True)
        M_all = df["MAG"].to_numpy(float)
        t_all = df["datetime"].to_numpy("datetime64[s]").astype("int64") / 86400.0
        figure_fmd(df_full, crep)   # Fig.2 needs the sub-Mc roll-off
    if trep:
        figure_loss(trep)

    npz = os.path.join(C.OUT_DIR, "predictions.npz")
    if res and os.path.exists(npz) and M_all is not None:
        Z = np.load(npz)
        figure_molchan_roc(Z, res, M_all, t_all)
        figure_baselines(Z, res, M_all, t_all)
    if igrep:
        figure_ig(igrep)

    with open(os.path.join(C.FIG_DIR, "figure_values.json"), "w") as f:
        json.dump(VALUES, f, indent=2, default=float)

    print("\nProduced: Fig2, Fig4, Fig5, Fig6, Fig7  (png + pdf, {} dpi)".format(C.DPI))
    print("Not produced here (not data products):")
    print("   Fig. 1  seismotectonic map")
    print("   Fig. 3  architecture schematic")
    print("\nEvery plotted value is written to outputs/figures/figure_values.json.")


if __name__ == "__main__":
    main()
