"""Step 1: catalog statistics.

Magnitude of completeness by three methods (MAXC, GFT, EMR), Aki b-value with
Shi and Bolt uncertainty, Gardner-Knopoff declustering for descriptive purposes,
window durations, and the class balance of each chronological partition.

Output:  outputs/catalog_report.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zm import config as C
from zm import catalog as cat
from zm import completeness as comp


def main():
    df_all = cat.load_clean_catalog()
    rep = {"n_events_clean": int(len(df_all)),
           "date_min": str(df_all["datetime"].min()),
           "date_max": str(df_all["datetime"].max())}

    print(f"Clean catalog: {len(df_all)} events")

    # ------------------------------------------------ Mc: three methods
    print("\nEstimating Mc (MAXC, GFT, EMR) ... this takes a minute")
    mc_rep = comp.estimate_mc(df_all["MAG"].to_numpy(), C.DM, C.MAXC_CORRECTION,
                              C.GFT_TARGET_R, C.MC_SEARCH, C.MC_MIN_EVENTS,
                              rule=C.MC_RULE,
                              stability_window=C.MC_STABILITY_WINDOW)
    rep["completeness"] = mc_rep
    mc = C.MC_OVERRIDE if C.MC_OVERRIDE is not None else mc_rep["mc_adopted"]
    rep["mc_used"] = float(mc)

    print(f"   MAXC raw     : {mc_rep['MAXC']['mc_raw']:.2f}  "
          f"(+{mc_rep['MAXC']['correction']}) -> {mc_rep['MAXC']['mc']:.2f}")
    print(f"   GFT          : {mc_rep['GFT']['mc']}  "
          f"(R={mc_rep['GFT']['R_at_mc']}, level={mc_rep['GFT']['level_reached']})")
    print(f"   EMR          : {mc_rep['EMR']['mc']}")
    _bs = mc_rep.get("B_STABILITY") or {}
    if _bs.get("mc") is not None:
        print(f"   b-STABILITY  : {_bs['mc']:.2f}   "
              f"(b={_bs['b']:.3f} +/- {_bs['sigma_b']:.3f}, Wiemer & Wyss 2000)")
    else:
        _rng = _bs.get("mc_range") or [None, None]
        print("   b-STABILITY  : NOT REACHED anywhere in "
              f"Mc = {_rng[0]} .. {_rng[1]} (Wiemer & Wyss 2000)")
        _cm, _cd = _bs.get("closest_mc"), _bs.get("closest_dev_sigma")
        if _cm is not None and _cd is not None:
            print(f"                  closest approach at Mc = {_cm:.2f}, "
                  f"deviation = {_cd:.2f} sigma  (criterion needs <= 1.00)")
        _br = _bs.get("b_range")
        if _br:
            print(f"                  b drifts {_br[0]:.3f} -> {_br[1]:.3f} across "
                  f"that range; monotonic rise = {_bs.get('monotonic_increasing')}")
        print("                  -> b is NOT stable against the cut-off. The "
              "statistical sigma below is")
        print("                     therefore an underestimate of the real "
              "uncertainty on b. Say so in the text.")
    print(f"   ADOPTED Mc   : {mc:.2f}   [{mc_rep['mc_rule']}]")

    # ------------------------------------------------ working catalog
    df = df_all[df_all["MAG"] >= mc - 1e-9].sort_values("datetime").reset_index(drop=True)
    df = cat.add_time_features(df)
    rep["n_events_above_mc"] = int(len(df))
    print(f"\nWorking catalog (M >= {mc:.2f}): {len(df)} events")

    # ------------------------------------------------ b-value
    b, sb, nb = comp.aki_b_value(df["MAG"].to_numpy(), mc, C.DM)
    a = comp.a_value(df["MAG"].to_numpy(), mc, b)
    rep["b_value_full"] = {"b": b, "sigma_b": sb, "n": nb, "a": a, "mc": float(mc),
                           "method": "Aki (1965) MLE; sigma after Shi & Bolt (1982)"}
    print(f"   b = {b:.3f} +/- {sb:.3f}   (n={nb}, a={a:.3f})")

    # b on the training period only; this is the Gutenberg-Richter reference
    n = len(df)
    i_tr = int(C.TRAIN_FRAC * n)
    b_tr, sb_tr, n_tr = comp.aki_b_value(df["MAG"].to_numpy()[:i_tr], mc, C.DM)
    rep["b_value_train_only"] = {"b": b_tr, "sigma_b": sb_tr, "n": n_tr,
                                 "note": "used as the Gutenberg-Richter reference "
                                         "density - fitted on training data only "
                                         "to avoid leakage"}
    print(f"   b (train only) = {b_tr:.3f} +/- {sb_tr:.3f}  (n={n_tr})  <- used for IG")

    # ------------------------------------------------ REPORTING Mc
    # The learning catalog above keeps Mc low on purpose. b and ETAS are
    # physical quantities and are quoted at the stability Mc instead.
    _bs = mc_rep.get("B_STABILITY") or {}
    _mr = _bs.get("mc")
    if str(getattr(C, "MC_REPORTING_RULE", "same")) == "b_stability" and _mr:
        mr = float(_mr)
        mr_rule = "b-value stability (Wiemer & Wyss 2000)"
    else:
        mr = float(mc)
        mr_rule = "same as the learning Mc"
    _mall = df_all["MAG"].to_numpy()
    br, sbr, nbr = comp.aki_b_value(_mall, mr, C.DM)
    ar = comp.a_value(_mall, mr, br)
    rep["mc_reporting"] = mr
    rep["b_value_reporting"] = {"b": br, "sigma_b": sbr, "n": nbr, "a": ar,
                                "mc": mr, "rule": mr_rule,
                                "note": ("quoted in the text and used for the "
                                         "ETAS parameters; the learning catalog "
                                         "is unchanged")}
    print(f"\n   REPORTING Mc = {mr:.2f}   [{mr_rule}]")
    print(f"   b (reporting) = {br:.3f} +/- {sbr:.3f}  (n={nbr}, a={ar:.3f})")
    if abs(mr - mc) > 1e-9:
        print(f"   NOTE: learning Mc = {mc:.2f} (n={len(df)}) is intentionally "
              f"lower than the reporting Mc.")
        print( "         Quote b at the reporting Mc; quote the event count at "
               "the learning Mc. Never mix them.")

    # ------------------------------------------------ declustering (descriptive)
    keep = cat.gardner_knopoff(df)
    rep["declustering"] = {
        "method": "Gardner & Knopoff (1974) space-time windows",
        "n_background": int(keep.sum()),
        "n_dependent": int((~keep).sum()),
        "pct_background": round(100.0 * keep.mean(), 2),
        "pct_dependent": round(100.0 * (~keep).mean(), 2),
        "note": "DESCRIPTIVE ONLY - the modelling catalog is deliberately not declustered",
    }
    b_bg, sb_bg, _ = comp.aki_b_value(df["MAG"].to_numpy()[keep], mc, C.DM)
    rep["b_value_declustered"] = {"b": b_bg, "sigma_b": sb_bg}
    print(f"   declustered: {keep.sum()} background ({100*keep.mean():.1f}%), "
          f"{(~keep).sum()} dependent;  b_bg = {b_bg:.3f} +/- {sb_bg:.3f}")

    # ------------------------------------------------ magnitude-scale robustness
    # The ISC preferred magnitude mixes local (ML) and body-wave (mb) scales.
    # Fitting b separately per scale shows whether the mixture biases b, which
    # matters because b is the Gutenberg-Richter reference used for information
    # gain. Descriptive only: the modelling catalog is not declustered.
    if "MAG_TYPE" in df.columns:
        fam = df["MAG_TYPE"].astype(str).str.strip().str.lower()
        rep["magnitude_family_counts"] = {str(k): int(v)
                                          for k, v in fam.value_counts().items()}
        n_tot = max(int(fam.size), 1)
        rep["magnitude_family_dominant_share"] = round(
            max(rep["magnitude_family_counts"].values()) / n_tot, 4)
        per = {}
        print("\n   b-value per magnitude family (mixed-scale robustness check):")
        for name in rep["magnitude_family_counts"]:
            sel = (fam == name).to_numpy()
            n_sel = int(sel.sum())
            if n_sel < C.MC_MIN_EVENTS:
                per[name] = {"n": n_sel, "b": None,
                             "note": "too few events above Mc for a stable Aki fit"}
                print(f"      {name:8s} n={n_sel:5d}   (too few - not fitted)")
                continue
            bf, sbf, nf = comp.aki_b_value(df["MAG"].to_numpy()[sel], mc, C.DM)
            per[name] = {"n": int(nf), "b": float(bf), "sigma_b": float(sbf)}
            print(f"      {name:8s} n={nf:5d}   b={bf:.3f} +/- {sbf:.3f}")
        rep["b_value_by_magnitude_family"] = per
        fitted = [(k, v) for k, v in per.items() if v.get("b") is not None]
        if len(fitted) >= 2:
            bs = [v["b"] for _, v in fitted]
            ses = [v["sigma_b"] for _, v in fitted]
            spread = float(max(bs) - min(bs))
            pooled = float(np.sqrt(sum(x * x for x in ses)))
            ok = bool(spread <= 2.0 * pooled)
            rep["b_family_spread"] = {
                "families_fitted": [k for k, _ in fitted],
                "b_min": float(min(bs)), "b_max": float(max(bs)),
                "spread": round(spread, 4),
                "two_sigma": round(2.0 * pooled, 4),
                "consistent_within_2sigma": ok,
                "note": "if consistent, mixing ML and mb does not detectably bias "
                        "b and the mixture can be reported as a stated limitation "
                        "rather than a confound"}
            print(f"      spread = {spread:.3f}   2-sigma = {2.0*pooled:.3f}")
            print("      -> " + ("CONSISTENT: the scale mixture does not detectably bias b"
                                 if ok else
                                 "NOT CONSISTENT: report the mixture as a limitation"))

    # ------------------------------------------------ per-window bookkeeping
    rep["datasets"] = {}
    for w in C.WINDOW_SIZES:
        d = cat.build_dataset(df, w)
        dur = cat.window_durations(df, d["idx_test"], w)
        entry = {
            "n_train_windows": d["n_train"], "n_val_windows": d["n_val"],
            "n_test_windows": d["n_test"],
            "split_event_index": d["split_event_index"],
            "depth_median_train_km": d["depth_median_train"],
            "n_depth_imputed": d["n_depth_imputed"],
            "window_duration_test_days": dur,
            "window_duration_all_days": cat.window_durations(
                df, np.arange(w, len(df)), w),
            "class_balance": {},
        }
        for thr in C.THRESHOLDS:
            cb = {}
            for part in ("train", "val", "test"):
                y = d[f"y_{part}"]
                pos = int((y >= thr).sum())
                cb[part] = {"n": int(y.size), "n_pos": pos, "n_neg": int(y.size - pos),
                            "base_rate": round(pos / max(1, y.size), 4)}
            entry["class_balance"][f"M>={thr}"] = cb
        rep["datasets"][f"w{w}"] = entry

        print(f"\n   w={w}: train {d['n_train']} / val {d['n_val']} / test {d['n_test']} windows")
        print(f"        test-window duration: median {dur['median_days']:.1f} d "
              f"(IQR {dur['q1']:.1f}-{dur['q3']:.1f}, p10 {dur['p10']:.1f}, p90 {dur['p90']:.1f})")
        for thr in C.THRESHOLDS:
            cb = entry["class_balance"][f"M>={thr}"]
            print(f"        M>={thr}: train {cb['train']['n_pos']}/{cb['train']['n']}  "
                  f"val {cb['val']['n_pos']}/{cb['val']['n']}  "
                  f"test {cb['test']['n_pos']}/{cb['test']['n']} "
                  f"(base rate {cb['test']['base_rate']:.3f})")

    # ------------------------------------------------ capacity vs data warning
    n_train = rep["datasets"][f"w{C.WINDOW_SIZES[0]}"]["n_train_windows"]
    rep["capacity_check"] = {
        "n_train_windows": n_train,
        "budgets": {str(bud): {"params_per_training_window": round(bud / n_train, 2),
                               "interpretation": ("MORE parameters than samples - "
                                                  "heavy regularisation / early stopping "
                                                  "is essential")}
                    for bud in C.PARAM_BUDGETS},
        "note": "A ratio of parameters to samples above one indicates an "
                "over-parameterised regime, i.e. data scarcity relative to model "
                "capacity.",
    }

    with open(os.path.join(C.OUT_DIR, "catalog_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print("\nOK -> outputs/catalog_report.json")


if __name__ == "__main__":
    main()
