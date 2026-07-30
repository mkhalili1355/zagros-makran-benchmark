"""Step 2: training of every architecture at each parameter budget, window
length and random seed.

The chronological split is applied to events before windows are formed, so no
window spans a partition boundary. The feature scaler and the depth imputation
are fitted on the training partition alone. The regression target is the raw
magnitude in physical units. Parameter budgets are enforced and the realised
counts are recorded. Only weights are stored; each graph is rebuilt from code.
Validation and test predictions are cached so that evaluation never reloads a
model.

Run:     python run_02_train.py
         ZM_QUICK=1 python run_02_train.py   (2 seeds, 8 epochs)
Output:  outputs/weights/*.weights.h5
         outputs/predictions.npz
         outputs/training_report.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zm import config as C
from zm import catalog as cat
from zm import models as M

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


def main():
    with open(os.path.join(C.OUT_DIR, "catalog_report.json")) as f:
        crep = json.load(f)
    mc = float(crep["mc_used"])

    df = cat.load_clean_catalog()
    df = df[df["MAG"] >= mc - 1e-9].sort_values("datetime").reset_index(drop=True)
    df = cat.add_time_features(df)

    report = {"mc": mc, "n_events": int(len(df)), "config": {
        "budgets": C.PARAM_BUDGETS, "seeds": C.SEEDS, "epochs": C.EPOCHS,
        "batch": C.BATCH, "patience": C.PATIENCE, "lr": C.LR,
        "quick_mode": C.QUICK}, "architectures": {}, "runs": []}

    store = {}
    t_start = time.time()

    for w in C.WINDOW_SIZES:
        d = cat.build_dataset(df, w)
        store[f"y_val_w{w}"] = d["y_val"]
        store[f"y_test_w{w}"] = d["y_test"]
        store[f"idx_test_w{w}"] = d["idx_test"]
        store[f"idx_val_w{w}"] = d["idx_val"]
        store[f"y_train_w{w}"] = d["y_train"]

        for budget in C.PARAM_BUDGETS:
            for arch in C.ARCHITECTURES:
                meta = M.tune_hyperparam(arch, w, budget)
                report["architectures"][f"{arch}_w{w}_p{budget}"] = meta
                if not meta["within_tol"]:
                    print(f"   !! {arch} w={w} budget={budget}: closest is "
                          f"{meta['params']} params ({meta['param_name']}={meta['value']}) "
                          f"- outside +/-{C.PARAM_TOL:.0%}")

                pv = np.zeros((len(C.SEEDS), d["n_val"]), dtype=np.float32)
                pt = np.zeros((len(C.SEEDS), d["n_test"]), dtype=np.float32)

                for si, seed in enumerate(C.SEEDS):
                    keras.backend.clear_session()
                    M.set_seeds(seed)
                    model, _ = M.build(arch, w, budget)
                    es = keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=C.PATIENCE,
                        restore_best_weights=True, verbose=0)
                    t0 = time.time()
                    hist = model.fit(
                        d["X_train"], d["y_train"],
                        validation_data=(d["X_val"], d["y_val"]),
                        epochs=C.EPOCHS, batch_size=C.BATCH,
                        callbacks=[es], verbose=0, shuffle=True)
                    dur = time.time() - t0

                    model.save_weights(M.weights_path(arch, w, budget, seed))
                    pv[si] = model.predict(d["X_val"], verbose=0).ravel()
                    pt[si] = model.predict(d["X_test"], verbose=0).ravel()

                    vl = hist.history["val_loss"]
                    tl = hist.history["loss"]
                    be = int(np.argmin(vl))
                    report["runs"].append({
                        "arch": arch, "w": w, "budget": budget, "seed": seed,
                        "params": meta["params"], "epochs_run": len(vl),
                        "best_epoch": be + 1,
                        "train_loss_at_best": float(tl[be]),
                        "val_loss_at_best": float(vl[be]),
                        "gap_train_minus_val": float(tl[be] - vl[be]),
                        "seconds": round(dur, 1),
                        "history_loss": [float(x) for x in tl],
                        "history_val_loss": [float(x) for x in vl],
                    })
                    print(f"   {arch:12s} w={w:2d} p={budget:6d} seed={seed} "
                          f"best_ep={be+1:3d} val={vl[be]:.5f} "
                          f"gap={tl[be]-vl[be]:+.4f}  {dur:5.1f}s")

                store[f"pred_val_{arch}_w{w}_p{budget}"] = pv
                store[f"pred_test_{arch}_w{w}_p{budget}"] = pt

    np.savez_compressed(os.path.join(C.OUT_DIR, "predictions.npz"), **store)
    report["total_minutes"] = round((time.time() - t_start) / 60.0, 1)
    with open(os.path.join(C.OUT_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOK  {report['total_minutes']} min")
    print("    -> outputs/predictions.npz")
    print("    -> outputs/training_report.json")
    print("\nRealised parameter counts:")
    for k, v in report["architectures"].items():
        flag = "" if v["within_tol"] else "   <-- OUTSIDE TOLERANCE"
        print(f"   {k:32s} {v['param_name']}={v['value']:<5d} "
              f"params={v['params']:,}{flag}")


if __name__ == "__main__":
    main()
