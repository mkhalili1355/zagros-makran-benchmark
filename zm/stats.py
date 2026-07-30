"""Resampling inference for dependent, overlapping windows.

Provides the moving-block bootstrap, the exact circular-shift randomisation test
for ranking statistics, the independent-permutation reference used only to
quantify inflation, Holm-Bonferroni correction over a declared family, and the
information gain per event relative to a Gutenberg-Richter reference.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C


# --------------------------------------------------------------------- blocks
def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample index vector of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def mbb_ci(values: np.ndarray, stat=np.mean, block: int | None = None,
           B: int | None = None, seed: int = C.RNG_SEED, alpha: float = 0.05):
    """Percentile MBB confidence interval for a statistic of a 1-D series."""
    v = np.asarray(values, float)
    block = block or C.MBB_BLOCK
    B = B or C.MBB_B
    rng = np.random.default_rng(seed)
    n = v.size
    boots = np.empty(B)
    for b in range(B):
        boots[b] = stat(v[_block_indices(n, block, rng)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(stat(v)), "ci_lo": float(lo), "ci_hi": float(hi),
            "block": block, "B": B}


# --------------------------------------------------------------------- [S1]
def mbb_paired_diff(a_err: np.ndarray, b_err: np.ndarray, block: int | None = None,
                    B: int | None = None, seed: int = C.RNG_SEED, alpha: float = 0.05):
    """Paired MBB test on mean(a_err) - mean(b_err) with a SHARED resample index.

    a_err, b_err are per-observation absolute errors of two forecasts on the
    same test set. Returns the difference, its CI and a two-sided p-value.
    """
    a, b = np.asarray(a_err, float), np.asarray(b_err, float)
    assert a.shape == b.shape
    block = block or C.MBB_BLOCK
    B = B or C.MBB_B
    rng = np.random.default_rng(seed)
    n = a.size
    d = a - b
    boots = np.empty(B)
    for i in range(B):
        idx = _block_indices(n, block, rng)          # shared -> pairing preserved
        boots[i] = d[idx].mean()
    obs = float(d.mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value, centred on the null of zero difference
    centred = boots - boots.mean()
    p = (1.0 + np.sum(np.abs(centred) >= abs(obs))) / (1.0 + B)   # [S5]
    return {"diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(1.0, p)), "block": block, "B": B, "n": int(n)}


def mbb_auc_ci(y: np.ndarray, scores: np.ndarray, block: int | None = None,
               B: int | None = None, seed: int = C.RNG_SEED, alpha: float = 0.05):
    """Moving-block bootstrap percentile CI for a single AUC."""
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    block = block or C.MBB_BLOCK
    B = B or C.MBB_B
    rng = np.random.default_rng(seed)
    n = y.size
    boots = []
    for _ in range(B):
        idx = _block_indices(n, block, rng)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        boots.append(roc_auc_score(yy, s[idx]))
    if len(boots) < 50:
        return {"auc": float(roc_auc_score(y, s)), "ci_lo": None, "ci_hi": None,
                "n_valid_boot": len(boots)}
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(roc_auc_score(y, s)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_valid_boot": len(boots), "block": block, "B": B}


def mbb_paired_auc(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                   block: int | None = None, B: int | None = None,
                   seed: int = C.RNG_SEED, alpha: float = 0.05):
    """Paired MBB test on AUC(a) - AUC(b) with a shared resample index."""
    y = np.asarray(y, int)
    block = block or C.MBB_BLOCK
    B = B or C.MBB_B
    rng = np.random.default_rng(seed)
    n = y.size
    obs = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    boots, ok = [], 0
    for _ in range(B):
        idx = _block_indices(n, block, rng)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        boots.append(roc_auc_score(yy, s_a[idx]) - roc_auc_score(yy, s_b[idx]))
        ok += 1
    boots = np.asarray(boots)
    if boots.size < 50:
        return {"diff": obs, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                "n_valid_boot": int(boots.size)}
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    centred = boots - boots.mean()
    p = (1.0 + np.sum(np.abs(centred) >= abs(obs))) / (1.0 + boots.size)
    return {"diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(1.0, p)), "n_valid_boot": int(boots.size),
            "block": block, "B": B}


# --------------------------------------------------------------------- [S2]
def circular_shift_auc_test(y: np.ndarray, scores: np.ndarray):
    """EXACT circular-shift randomisation test for AUC.

    The score series is rotated against the label series over all n-1
    non-trivial lags. This preserves the autocorrelation of the forecast,
    which an i.i.d. label permutation destroys.
    """
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    n = y.size
    obs = float(roc_auc_score(y, s))
    null = np.empty(n - 1)
    for k in range(1, n):
        null[k - 1] = roc_auc_score(y, np.roll(s, k))
    r = int(np.sum(null >= obs))
    p = (1.0 + r) / (1.0 + null.size)                    # [S5]
    return {"auc": obs, "p_circular_shift": float(p),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "n_shifts": int(null.size),
            "null_q95": float(np.percentile(null, 95))}


def iid_permutation_auc_test(y: np.ndarray, scores: np.ndarray, n_perm: int = 3000,
                             seed: int = C.RNG_SEED):
    """i.i.d. label permutation (Ojala & Garriga 2010).

    COMPUTED FOR CONTRAST ONLY. This test is invalid here because it assumes
    exchangeable labels, which overlapping seismic windows are not.
    """
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    obs = float(roc_auc_score(y, s))
    cnt = 0
    for _ in range(n_perm):
        if roc_auc_score(rng.permutation(y), s) >= obs:
            cnt += 1
    return {"auc": obs, "p_iid_permutation": float((1.0 + cnt) / (1.0 + n_perm)),
            "n_perm": int(n_perm)}


# --------------------------------------------------------------------- [S3]
def molchan(y: np.ndarray, scores: np.ndarray):
    """Molchan error diagram with correct tie handling.

    Alarms are declared for the highest-scoring windows. The trajectory is
    advanced one DISTINCT score value at a time, so tied scores never produce
    an artificially favourable operating point.

    Returns tau (alarm fraction), nu (miss fraction), area A and skill
    S = 1 - 2A  (Zechar & Jordan 2008).
    """
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    n = y.size
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n:
        return {"tau": [], "nu": [], "A": np.nan, "S": np.nan, "n_pos": n_pos}

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # boundaries where the score value changes
    change = np.r_[np.flatnonzero(np.diff(s_sorted)) + 1, n]

    tau = [0.0]
    nu = [1.0]
    hits = 0
    for cut in change:
        hits = int(y_sorted[:cut].sum())
        tau.append(cut / n)
        nu.append(1.0 - hits / n_pos)
    tau, nu = np.asarray(tau), np.asarray(nu)
    A = float(np.trapz(nu, tau))
    return {"tau": tau.tolist(), "nu": nu.tolist(), "A": A, "S": float(1.0 - 2.0 * A),
            "n_pos": n_pos, "n": int(n)}


def molchan_auc_identity(S: float, auc: float, pi: float) -> dict:
    """Internal consistency check  S = (1 - pi) * (2*AUC - 1).

    This is an algebraic identity, NOT independent corroboration. It only
    verifies that the two routines were fed the same scores and labels.
    """
    pred = (1.0 - pi) * (2.0 * auc - 1.0)
    return {"S_observed": float(S), "S_from_identity": float(pred),
            "abs_diff": float(abs(S - pred)), "pi": float(pi),
            "consistent": bool(abs(S - pred) < 5e-3)}


# --------------------------------------------------------------------- [S4]
def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm (1979) step-down correction over an explicitly named family."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": float(p), "p_holm": float(adj),
                  "reject_at_0.05": bool(adj < alpha)}
    return {"family_size": m, "alpha": alpha, "results": out}


# --------------------------------------------------------------------- info gain
def gaussian_logpdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-9)
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((x - mu) / sigma) ** 2


def gr_logpdf(m, b, mc, dm):
    """Log density of the binned Gutenberg-Richter (truncated exponential).

    f(m) = beta * exp(-beta * (m - mc)),  m >= mc,  beta = b * ln(10)
    Evaluated as a per-bin probability of width dm so it is comparable with a
    continuous density integrated over the same bin.
    """
    beta = b * np.log(10.0)
    m = np.asarray(m, float)
    val = np.where(m >= mc - dm / 2, beta * np.exp(-beta * np.clip(m - mc, 0, None)), 1e-300)
    return np.log(np.clip(val, 1e-300, None))


def information_gain(y_true, y_pred, sigma, reference: str,
                     clim_mu=None, clim_sigma=None, b=None, mc=None, dm=None,
                     block=None, B=None, seed=C.RNG_SEED):
    """Mean information gain in BITS per earthquake, with an MBB CI.

    reference = 'gaussian_climatology' | 'gutenberg_richter'
    """
    y = np.asarray(y_true, float)
    ll_model = gaussian_logpdf(y, np.asarray(y_pred, float), sigma)
    if reference == "gaussian_climatology":
        ll_ref = gaussian_logpdf(y, clim_mu, clim_sigma)
    elif reference == "gutenberg_richter":
        ll_ref = gr_logpdf(y, b, mc, dm)
    else:
        raise ValueError(reference)
    per_event = (ll_model - ll_ref) / np.log(2.0)          # nats -> bits
    ci = mbb_ci(per_event, np.mean, block, B, seed)
    # two-sided MBB p-value against IG = 0
    rng = np.random.default_rng(seed)
    Bn = B or C.MBB_B
    bl = block or C.MBB_BLOCK
    boots = np.array([per_event[_block_indices(per_event.size, bl, rng)].mean()
                      for _ in range(Bn)])
    centred = boots - boots.mean()
    p = (1.0 + np.sum(np.abs(centred) >= abs(per_event.mean()))) / (1.0 + Bn)
    return {"reference": reference, "ig_bits": float(per_event.mean()),
            "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "p": float(min(1.0, p)),
            "sigma": float(sigma), "n": int(y.size)}
