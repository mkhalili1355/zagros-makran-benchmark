"""Step 4: Integrated Gradients attribution (Sundararajan et al., 2017).

Attribution is computed for all four architectures at both window lengths. The
reference input is the mean of N random training windows drawn under a fixed
seed. Attributions are averaged over seeds and then min-max normalised per
model, and both the temporal and the per-feature profiles are exported.

Output:  outputs/ig_results.npz
         outputs/ig_report.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import tensorflow as tf
# Keras import shim.
# TensorFlow 2.16+ ships Keras 3 as a separate package and exposes a *lazy*
# `tensorflow.keras` alias that points at `keras._tf_keras.keras`. That alias is
# a compatibility shim and does NOT carry the full Keras 3 API - notably
# `keras.saving` is missing, which raises
#     AttributeError: module 'keras._tf_keras.keras' has no attribute 'saving'
# on TF 2.20. Importing the standalone `keras` package gives the real API.
try:
    import keras  # Keras 3 (TensorFlow >= 2.16)
except ImportError:  # pragma: no cover - TensorFlow <= 2.15 only
    from tensorflow import keras
layers = keras.layers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zm import config as C
from zm import catalog as cat
from zm import models as M


def integrated_gradients(model, x, baseline, n_steps=C.IG_N_STEPS):
    """IG for a scalar-output regression model. x has shape (B, T, F)."""
    x = tf.convert_to_tensor(x, tf.float32)
    baseline = tf.convert_to_tensor(baseline, tf.float32)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, n_steps + 1), (n_steps + 1, 1, 1, 1))
    delta = tf.expand_dims(x - baseline, 0)
    path = tf.expand_dims(baseline, 0) + alphas * delta

    grads = []
    for s in range(n_steps + 1):
        pt = path[s]
        with tf.GradientTape() as tape:
            tape.watch(pt)
            out = model(pt, training=False)
        grads.append(tape.gradient(out, pt))
    grads = tf.stack(grads)
    avg = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)
    return ((x - baseline) * avg).numpy()


def main():
    with open(os.path.join(C.OUT_DIR, "catalog_report.json")) as f:
        crep = json.load(f)
    mc = float(crep["mc_used"])

    df = cat.load_clean_catalog()
    df = df[df["MAG"] >= mc - 1e-9].sort_values("datetime").reset_index(drop=True)
    df = cat.add_time_features(df)

    store = {}
    report = {"config": {"n_steps": C.IG_N_STEPS,
                         "n_baseline_windows": C.IG_N_BASELINE,
                         "baseline_seed": C.IG_BASELINE_SEED,
                         "seeds": C.SEEDS,
                         "baseline": "mean of N random training windows"},
              "models": {}}

    for w in C.WINDOW_SIZES:
        d = cat.build_dataset(df, w)
        rng = np.random.default_rng(C.IG_BASELINE_SEED)
        pick = rng.choice(d["n_train"], size=min(C.IG_N_BASELINE, d["n_train"]),
                          replace=False)
        base = d["X_train"][pick].mean(axis=0, keepdims=True).astype("float32")
        store["baseline_w{}".format(w)] = base

        for budget in C.PARAM_BUDGETS:
            for arch in C.ARCHITECTURES:
                key = "{}_w{}_p{}".format(arch, w, budget)
                acc = np.zeros((d["n_test"], w, C.N_FEAT), dtype=np.float64)
                ok = 0
                for seed in C.SEEDS:
                    wp = M.weights_path(arch, w, budget, seed)
                    if not os.path.exists(wp):
                        print("   !! missing weights: " + wp)
                        continue
                    keras.backend.clear_session()
                    M.set_seeds(seed)
                    model, _ = M.build(arch, w, budget)
                    model.load_weights(wp)
                    b = np.repeat(base, d["n_test"], axis=0)
                    acc += integrated_gradients(model, d["X_test"], b)
                    ok += 1
                if ok == 0:
                    report["models"][key] = {"status": "FAILED - no weights found"}
                    continue
                ig = acc / ok

                temporal = np.abs(ig).mean(axis=(0, 2))
                per_feat = np.abs(ig).mean(axis=(0, 1))
                tmin, tmax = temporal.min(), temporal.max()
                if tmax > tmin:
                    temporal_n = (temporal - tmin) / (tmax - tmin)
                else:
                    temporal_n = temporal * 0.0

                store["ig_" + key] = ig.astype("float32")
                store["temporal_" + key] = temporal_n
                store["perfeat_" + key] = per_feat

                rev = temporal[::-1]
                cum = np.cumsum(rev) / rev.sum()
                mem50 = int(np.searchsorted(cum, 0.50) + 1)
                mem90 = int(np.searchsorted(cum, 0.90) + 1)

                report["models"][key] = {
                    "status": "ok",
                    "n_seeds_used": ok,
                    "n_test_windows": int(d["n_test"]),
                    "feature_importance": dict(zip(C.FEATURES,
                                                   [float(v) for v in per_feat])),
                    "feature_ranking": [C.FEATURES[i] for i in np.argsort(-per_feat)],
                    "temporal_profile_normalised": [float(x) for x in temporal_n],
                    "effective_memory_steps_50pct": mem50,
                    "effective_memory_steps_90pct": mem90,
                    "attribution_share_last_10_steps": float(
                        temporal[-10:].sum() / temporal.sum()),
                }
                print("   {:28s} top={:14s} mem50={:3d} last10share={:.2f}".format(
                    key,
                    report["models"][key]["feature_ranking"][0],
                    mem50,
                    report["models"][key]["attribution_share_last_10_steps"]))

    np.savez_compressed(os.path.join(C.OUT_DIR, "ig_results.npz"), **store)
    with open(os.path.join(C.OUT_DIR, "ig_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    failed = [k for k, v in report["models"].items() if v.get("status") != "ok"]
    print("\nOK  {}/{} models".format(len(report["models"]) - len(failed),
                                      len(report["models"])))
    if failed:
        print("    FAILED: " + ", ".join(failed))
    print("    -> outputs/ig_results.npz, outputs/ig_report.json")


if __name__ == "__main__":
    main()
