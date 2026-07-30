"""Step 3: baselines, ETAS, MAE, AUC, Molchan diagrams, resampling inference
and information gain.

Baselines are evaluated on exactly the test windows used by the models, and no
model or baseline is selected on the test partition. Uncertainty is quantified
with a moving-block bootstrap throughout. AUC significance is assessed with an
exact circular-shift randomisation; the independent-permutation p-value is
reported alongside it only to quantify the inflation it produces. ETAS is fitted
on training events, as is the Gutenberg-Richter reference b-value. Multiplicity
is controlled with Holm-Bonferroni over an explicitly declared family.

Output:  outputs/results.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zm import config as C
from zm import catalog as cat
from zm import completeness as comp
from zm import stats as S
from zm import etas as E
from zm import baselines as B


def main():
    with open(os.path.join(C.OUT_DIR, "catalog_report.json")) as f:
        crep = json.load(f)
    mc = float(crep["mc_used"])
    b_train = float(crep["b_value_train_only"]["b"])

    df = cat.load_clean_catalog()
    df = df[df["MAG"] >= mc - 1e-9].sort_values("datetime").reset_index(drop=True)
    df = cat.add_time_features(df)
    M_all = df["MAG"].to_numpy(float)
    t_all = df["datetime"].to_numpy("datetime64[s]").astype("int64") / 86400.0

    Z = np.load(os.path.join(C.OUT_DIR, "predictions.npz"))
    R = {"mc": mc, "b_train": b_train, "windows": {}}

    n_ev = len(df)
    i_tr = int(C.TRAIN_FRAC * n_ev)
    i_va = int((C.TRAIN_FRAC + C.VAL_FRAC) * n_ev)

    # ================================================================= ETAS
    print("Fitting ETAS on training events only (corrected likelihood) ...")
    et = E.fit_etas(t_all[:i_tr] - t_all[0], M_all[:i_tr], mc)
    R["etas"] = et
    def _pm(v, e, nd=4):
        return f"{v:.{nd}f}" + (f" +/- {e:.{nd}f}" if e is not None else "")
    if et.get("converged"):
        se = et["se"]
        print(f"   mu={_pm(et['mu_per_day'], se['mu'])} /day   "
              f"K={_pm(et['K'], se['K'])}   c={_pm(et['c_days'], se['c'])} d")
        print(f"   alpha={_pm(et['alpha'], se['alpha'], 3)}   "
              f"p={_pm(et['p'], se['p'], 4)}")
        if et.get("p_at_lower_bound"):
            print("   !! p is at its lower bound (p -> 1, logarithmic Omori limit).")
            print("      Quote it as 'p ~ 1.0 (boundary)', not as a fitted value.")
        print(f"   logL={et['loglik']:.1f}  AIC={et['AIC']:.1f}  n={et['n_events']}")
    else:
        print("   !! ETAS did not converge")

    # ---- second ETAS fit at the REPORTING Mc, for the text only.
    # The baseline used in Figs 5 and 6 stays at the learning Mc so that it
    # remains comparable with the deep models.
    mc_rp = float(crep.get("mc_reporting", mc))
    if abs(mc_rp - mc) > 1e-9:
        print(f"\nRefitting ETAS at the REPORTING Mc = {mc_rp:.2f} "
              f"(this is the fit quoted in the text) ...")
        dfr = cat.load_clean_catalog()
        dfr = (dfr[dfr["MAG"] >= mc_rp - 1e-9]
               .sort_values("datetime").reset_index(drop=True))
        Mr = dfr["MAG"].to_numpy(float)
        tr = dfr["datetime"].to_numpy("datetime64[s]").astype("int64") / 86400.0
        j_tr = int(C.TRAIN_FRAC * len(dfr))
        etr = E.fit_etas(tr[:j_tr] - tr[0], Mr[:j_tr], mc_rp)
        R["etas_reporting"] = etr
        if etr.get("converged"):
            s2 = etr["se"]
            print(f"   mu={_pm(etr['mu_per_day'], s2['mu'])} /day   "
                  f"K={_pm(etr['K'], s2['K'])}   c={_pm(etr['c_days'], s2['c'])} d")
            print(f"   alpha={_pm(etr['alpha'], s2['alpha'], 3)}   "
                  f"p={_pm(etr['p'], s2['p'], 4)}")
            if etr.get("p_at_lower_bound"):
                print("   !! p is at its lower bound here as well.")
            print(f"   logL={etr['loglik']:.1f}  AIC={etr['AIC']:.1f}  "
                  f"n={etr['n_events']}")
        else:
            print("   !! reporting ETAS did not converge")
    else:
        R["etas_reporting"] = et

    for w in C.WINDOW_SIZES:
        print(f"\n{'='*70}\nWINDOW w = {w}\n{'='*70}")
        y_val = Z[f"y_val_w{w}"]
        y_test = Z[f"y_test_w{w}"]
        idx_test = Z[f"idx_test_w{w}"]
        idx_val = Z[f"idx_val_w{w}"]
        y_train = Z[f"y_train_w{w}"]
        n_test = y_test.size
        W = {"n_train": int(y_train.size), "n_val": int(y_val.size),
             "n_test": n_test, "models": {}, "baselines": {}}

        # ------------------------------------------------------ baselines
        clim_med = float(np.median(y_train))
        clim_mean = float(np.mean(y_train))
        clim_sd = float(np.std(y_train, ddof=1))
        base = {
            "median_climatology": np.full(n_test, clim_med),
            "mean_climatology": np.full(n_test, clim_mean),
            "persistence_last": M_all[idx_test - 1],
            "persistence_window_max": np.array([M_all[i - w:i].max() for i in idx_test]),
            "persistence_window_mean": np.array([M_all[i - w:i].mean() for i in idx_test]),
            "gr_expectation": E.etas_magnitude_forecast(b_train, mc, n_test),
        }
        W["climatology"] = {"median": clim_med, "mean": clim_mean, "sd": clim_sd,
                            "source": "training windows only"}
        for nm, pr in base.items():
            err = np.abs(pr - y_test)
            ci = S.mbb_ci(err)
            W["baselines"][nm] = {
                "MAE": float(err.mean()),
                "MAE_ci": [ci["ci_lo"], ci["ci_hi"]],
                "RMSE": float(np.sqrt(((pr - y_test) ** 2).mean())),
            }
            print(f"   baseline {nm:26s} MAE = {err.mean():.4f} "
                  f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

        ref_name = "median_climatology"
        ref_pred = base[ref_name]
        ref_err = np.abs(ref_pred - y_test)

        # ------------------------------------------------------ models
        pvals_mae, pvals_auc = {}, {}
        for budget in C.PARAM_BUDGETS:
            for arch in C.ARCHITECTURES:
                key = f"{arch}_p{budget}"
                pv = Z[f"pred_val_{arch}_w{w}_p{budget}"]
                pt = Z[f"pred_test_{arch}_w{w}_p{budget}"]

                per_seed_mae = np.abs(pt - y_test[None, :]).mean(axis=1)
                ens = pt.mean(axis=0)                      # ensemble prediction
                ens_err = np.abs(ens - y_test)

                # sigma of the predictive density from VALIDATION residuals
                val_res = (pv.mean(axis=0) - y_val)
                sigma = float(np.sqrt(np.mean(val_res ** 2)))

                entry = {
                    "params": None,
                    "MAE_per_seed_mean": float(per_seed_mae.mean()),
                    "MAE_per_seed_sd": float(per_seed_mae.std(ddof=1)) if per_seed_mae.size > 1 else 0.0,
                    "MAE_ensemble": float(ens_err.mean()),
                    "sigma_validation": sigma,
                    "thresholds": {},
                }
                ci = S.mbb_ci(ens_err)
                entry["MAE_ensemble_ci"] = [ci["ci_lo"], ci["ci_hi"]]
                print(f"   {key:22s} MAE = {ens_err.mean():.4f} "
                      f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]   "
                      f"(per-seed {per_seed_mae.mean():.4f} +/- {entry['MAE_per_seed_sd']:.4f})")

                # ---- paired MBB vs the climatology reference
                pd_ = S.mbb_paired_diff(ens_err, ref_err)
                entry["vs_median_climatology"] = pd_
                pvals_mae[f"MAE:{key}_vs_{ref_name}"] = pd_["p"]

                # ---- information gain
                entry["info_gain"] = {
                    "vs_gaussian_climatology": S.information_gain(
                        y_test, ens, sigma, "gaussian_climatology",
                        clim_mu=clim_mean, clim_sigma=clim_sd),
                    "vs_gutenberg_richter": S.information_gain(
                        y_test, ens, sigma, "gutenberg_richter",
                        b=b_train, mc=mc, dm=C.DM),
                }

                # ---- classification at each threshold
                for thr in C.THRESHOLDS:
                    yb = (y_test >= thr).astype(int)
                    n_pos = int(yb.sum())
                    if n_pos < 5 or n_pos == yb.size:
                        entry["thresholds"][f"M>={thr}"] = {
                            "n_pos": n_pos, "skipped": "too few positives"}
                        continue
                    auc_seeds = np.array([roc_auc_score(yb, p) for p in pt])
                    auc_ens = float(roc_auc_score(yb, ens))
                    cs = S.circular_shift_auc_test(yb, ens)
                    perm = S.iid_permutation_auc_test(yb, ens)
                    mol = S.molchan(yb, ens)
                    pi = n_pos / yb.size
                    ident = S.molchan_auc_identity(mol["S"], auc_ens, pi)

                    entry["thresholds"][f"M>={thr}"] = {
                        "n_pos": n_pos, "n": int(yb.size), "base_rate": round(pi, 4),
                        "AUC_per_seed_mean": float(auc_seeds.mean()),
                        "AUC_per_seed_sd": float(auc_seeds.std(ddof=1)) if auc_seeds.size > 1 else 0.0,
                        "AUC_ensemble": auc_ens,
                        "AUC_ci": S.mbb_auc_ci(yb, ens),
                        "p_circular_shift": cs["p_circular_shift"],
                        "circular_shift_null": {"mean": cs["null_mean"], "sd": cs["null_sd"],
                                                "q95": cs["null_q95"], "n": cs["n_shifts"]},
                        "p_iid_permutation_INVALID": perm["p_iid_permutation"],
                        "p_inflation_factor": round(
                            cs["p_circular_shift"] / max(perm["p_iid_permutation"], 1e-12), 1),
                        "molchan_S": mol["S"], "molchan_A": mol["A"],
                        "molchan_tau": mol["tau"], "molchan_nu": mol["nu"],
                        "identity_check": ident,
                    }
                    if thr == C.PRIMARY_THRESHOLD:
                        pvals_auc[f"AUC:{key}@M>={thr}"] = cs["p_circular_shift"]
                    print(f"   {key:22s} M>={thr}: AUC={auc_ens:.3f} "
                          f"S={mol['S']:+.3f} p_cs={cs['p_circular_shift']:.3f} "
                          f"p_iid={perm['p_iid_permutation']:.4f} "
                          f"(inflation x{cs['p_circular_shift']/max(perm['p_iid_permutation'],1e-12):.0f})")

                W["models"][key] = entry

        # ------------------------------------------------------ baseline AUCs
        W["baseline_auc"] = {}
        for thr in C.THRESHOLDS:
            yb = (y_test >= thr).astype(int)
            if yb.sum() < 5:
                continue
            row = {}
            rank = B.ranking_baselines(M_all, t_all, idx_test, w)
            for nm, pr in list(base.items()) + list(rank.items()):
                if np.ptp(pr) == 0:
                    row[nm] = {"AUC": 0.5, "note": "constant forecast - AUC undefined, set to 0.5"}
                    continue
                a = float(roc_auc_score(yb, pr))
                cs = S.circular_shift_auc_test(yb, pr)
                row[nm] = {"AUC": a, "p_circular_shift": cs["p_circular_shift"],
                           "molchan_S": S.molchan(yb, pr)["S"]}
            # ETAS occurrence-rate score, evaluated strictly causally
            if R["etas"].get("converged"):
                lam = E.conditional_intensity(R["etas"], t_all, M_all,
                                              t_all[idx_test], mc)
                a = float(roc_auc_score(yb, lam))
                row["etas_conditional_intensity"] = {
                    "AUC": a,
                    "p_circular_shift": S.circular_shift_auc_test(yb, lam)["p_circular_shift"],
                    "note": ("ETAS lambda(t) as a MAGNITUDE discriminator. ETAS assumes "
                             "magnitude independence, so any departure from 0.50 reflects "
                             "rate-magnitude covariance in the catalog, not ETAS skill."),
                }
            W["baseline_auc"][f"M>={thr}"] = row
        W["ranking_baseline_definitions"] = B.DEFINITIONS

        # ------------------------------------------------------ Holm
        W["holm_MAE"] = S.holm_bonferroni(pvals_mae)
        W["holm_AUC"] = S.holm_bonferroni(pvals_auc)
        W["holm_note"] = ("Family membership is fixed a priori: all architecture-vs-"
                          "climatology MAE comparisons, and all architecture AUC tests "
                          f"at the pre-specified threshold M >= {C.PRIMARY_THRESHOLD}.")

        # ------------------------------------------------------ best model, chosen on VALIDATION
        val_scores = {}
        for budget in C.PARAM_BUDGETS:
            for arch in C.ARCHITECTURES:
                pv = Z[f"pred_val_{arch}_w{w}_p{budget}"].mean(axis=0)
                val_scores[f"{arch}_p{budget}"] = float(np.abs(pv - y_val).mean())
        best = min(val_scores, key=val_scores.get)
        W["model_selection"] = {
            "criterion": "lowest ensemble MAE on the VALIDATION split",
            "validation_MAE": val_scores, "selected": best,
            "note": "Selection never touches the test set."}
        print(f"\n   selected on validation: {best}")

        R["windows"][f"w{w}"] = W

    with open(os.path.join(C.OUT_DIR, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("\nOK -> outputs/results.json")


if __name__ == "__main__":
    main()
