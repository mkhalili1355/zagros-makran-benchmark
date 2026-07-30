"""Model constructors for the four candidate architectures.

Each architecture is built from code and tuned to the requested parameter
budget, so that the realised parameter count can be measured rather than
assumed. Only weights are persisted, which keeps model construction independent
of the serialisation format.
"""
from __future__ import annotations
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

from . import config as C


# --------------------------------------------------------------------- [M1]
@keras.saving.register_keras_serializable(package="zm")
class LastStep(layers.Layer):
    """Take the last time step: (B, T, F) -> (B, F). Replaces Lambda."""

    def call(self, x):
        return x[:, -1, :]

    def compute_output_shape(self, s):
        return (s[0], s[2])


@keras.saving.register_keras_serializable(package="zm")
class AddPositionalEmbedding(layers.Layer):
    """Learned positional embedding added to the sequence. Replaces Lambda."""

    def __init__(self, seq_len, d_model, **kw):
        super().__init__(**kw)
        self.seq_len, self.d_model = int(seq_len), int(d_model)

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos", shape=(self.seq_len, self.d_model),
            initializer="random_normal", trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


# --------------------------------------------------------------------- builders
def build_lstm(w, units, dense=32):
    inp = keras.Input((w, C.N_FEAT))
    x = layers.LSTM(units)(inp)
    x = layers.Dense(dense, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="LSTM")


def build_gru(w, units, dense=32):
    inp = keras.Input((w, C.N_FEAT))
    x = layers.GRU(units)(inp)
    x = layers.Dense(dense, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="GRU")


def build_tcn(w, filters, dense=32):
    """Residual dilated causal TCN, implemented natively (no external package).

    Receptive field = 1 + 2 * sum(dilations) * (kernel-1)/2 ... for kernel 3 and
    dilations [1,2,4,8,16] with one conv per block this is 1 + 2*31 = 63 steps.
    """
    inp = keras.Input((w, C.N_FEAT))
    x = layers.Conv1D(filters, 1, padding="same")(inp)          # channel match
    for d in C.TCN_DILATIONS:
        skip = x
        h = layers.Conv1D(filters, C.TCN_KERNEL, padding="causal",
                          dilation_rate=d, activation="relu")(x)
        h = layers.Conv1D(filters, 1, padding="same")(h)
        x = layers.Add()([skip, h])
        x = layers.Activation("relu")(x)
    x = LastStep()(x)                                            # [M1]
    x = layers.Dense(dense, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="TCN")


def build_transformer(w, d_model, n_heads=4, ff_mult=2, dense=32):
    key_dim = max(1, d_model // n_heads)
    inp = keras.Input((w, C.N_FEAT))
    x = layers.Dense(d_model)(inp)
    x = AddPositionalEmbedding(w, d_model)(x)                    # [M1]
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim)(x, x)
    x = layers.LayerNormalization()(layers.Add()([x, attn]))
    ff = layers.Dense(d_model * ff_mult, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization()(layers.Add()([x, ff]))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out, name="Transformer")


_BUILDERS = {
    "LSTM": (build_lstm, "units", range(4, 400)),
    "GRU": (build_gru, "units", range(4, 400)),
    "TCN": (build_tcn, "filters", range(4, 200)),
    "Transformer": (build_transformer, "d_model", range(4, 256, 4)),
}


def count_params(model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_weights))


# --------------------------------------------------------------------- [M2]
_CACHE: dict = {}


def tune_hyperparam(arch: str, w: int, budget: int) -> dict:
    """Find the width that puts the parameter count closest to `budget`."""
    key = (arch, w, budget)
    if key in _CACHE:
        return _CACHE[key]
    builder, pname, grid = _BUILDERS[arch]
    best = None
    for v in grid:
        try:
            m = builder(w, v)
            p = count_params(m)
            keras.backend.clear_session()
        except Exception:
            continue
        err = abs(p - budget)
        if best is None or err < best["err"]:
            best = {"value": int(v), "params": int(p), "err": int(err)}
        if p > budget * 2:
            break
    best["arch"], best["w"], best["budget"], best["param_name"] = arch, w, budget, pname
    best["within_tol"] = bool(best["err"] <= C.PARAM_TOL * budget)
    _CACHE[key] = best
    return best


def build(arch: str, w: int, budget: int):
    """Build a model at the given parameter budget. Returns (model, meta)."""
    meta = tune_hyperparam(arch, w, budget)
    builder, _, _ = _BUILDERS[arch]
    model = builder(w, meta["value"])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=C.LR, beta_1=C.ADAM_BETA1,
            beta_2=C.ADAM_BETA2, epsilon=C.ADAM_EPS),
        loss="mse", metrics=["mae"])
    return model, meta


def set_seeds(seed: int) -> None:
    import random
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def weights_path(arch: str, w: int, budget: int, seed: int) -> str:
    import os
    return os.path.join(C.MODEL_DIR, f"{arch}_w{w}_p{budget}_s{seed}.weights.h5")
