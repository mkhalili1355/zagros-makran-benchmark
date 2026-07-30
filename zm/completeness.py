"""Magnitude of completeness and b-value.

Three completeness estimators are implemented:
  MAXC  maximum curvature with correction (Wiemer and Wyss, 2000;
        Woessner and Wiemer, 2005)
  GFT   goodness-of-fit test at the 90 % and 95 % residual levels
        (Wiemer and Wyss, 2000)
  EMR   entire-magnitude-range method (Woessner and Wiemer, 2005)

The b-value follows the Aki (1965) maximum-likelihood estimator with the
uncertainty of Shi and Bolt (1982).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

LOG10E = np.log10(np.e)


# --------------------------------------------------------------------- b-value
def aki_b_value(mags: np.ndarray, mc: float, dm: float) -> tuple[float, float, int]:
    """Aki (1965) MLE b-value with Shi & Bolt (1982) standard error.

    Returns (b, sigma_b, n_used).
    """
    m = np.asarray(mags, dtype=float)
    m = m[m >= mc - dm / 2.0 - 1e-9]
    n = m.size
    if n < 2:
        return np.nan, np.nan, n
    mbar = m.mean()
    denom = mbar - (mc - dm / 2.0)
    if denom <= 0:
        return np.nan, np.nan, n
    b = LOG10E / denom
    # Shi & Bolt (1982)
    var = np.sum((m - mbar) ** 2) / (n * (n - 1))
    sigma = 2.30 * (b ** 2) * np.sqrt(var)
    return float(b), float(sigma), int(n)


def a_value(mags: np.ndarray, mc: float, b: float) -> float:
    """a-value of log10 N = a - b M, normalised to the whole catalog period."""
    n = int(np.sum(np.asarray(mags) >= mc - 1e-9))
    if n == 0 or not np.isfinite(b):
        return np.nan
    return float(np.log10(n) + b * mc)


# --------------------------------------------------------------------- helpers
def _fmd(mags: np.ndarray, dm: float):
    """Non-cumulative and cumulative FMD on a common bin grid."""
    m = np.asarray(mags, dtype=float)
    lo = np.round(m.min() / dm) * dm
    hi = np.round(m.max() / dm) * dm
    bins = np.round(np.arange(lo, hi + dm / 2, dm), 5)
    idx = np.round((m - lo) / dm).astype(int)
    idx = np.clip(idx, 0, bins.size - 1)
    non_cum = np.bincount(idx, minlength=bins.size).astype(float)
    cum = np.cumsum(non_cum[::-1])[::-1]
    return bins, non_cum, cum


# --------------------------------------------------------------------- MAXC
def mc_maxc(mags: np.ndarray, dm: float, correction: float = 0.2) -> dict:
    bins, non_cum, _ = _fmd(mags, dm)
    raw = float(bins[int(np.argmax(non_cum))])
    return {"method": "MAXC", "mc_raw": raw, "correction": correction,
            "mc": round(raw + correction, 4)}


# --------------------------------------------------------------------- GFT
def mc_gft(mags: np.ndarray, dm: float, target_r: float = 90.0,
           search=(2.0, 4.5), min_events: int = 200) -> dict:
    """Goodness-of-fit test (Wiemer & Wyss 2000).

    For each candidate Mc, fit GR by Aki MLE and compare the *synthetic*
    cumulative FMD with the observed one:
        R = 100 * (1 - sum|obs - syn| / sum obs)
    The smallest Mc reaching R >= target_r is returned.
    """
    m = np.asarray(mags, dtype=float)
    cands = np.round(np.arange(search[0], search[1] + dm / 2, dm), 5)
    rows = []
    for mc in cands:
        sub = m[m >= mc - 1e-9]
        if sub.size < min_events:
            continue
        b, _, _ = aki_b_value(sub, mc, dm)
        if not np.isfinite(b):
            continue
        bins, _, cum = _fmd(sub, dm)
        a = np.log10(sub.size) + b * mc
        syn = 10.0 ** (a - b * bins)
        # only compare over the observed support
        r = 100.0 * (1.0 - np.sum(np.abs(cum - syn)) / np.sum(cum))
        rows.append({"mc": float(mc), "b": float(b), "R": float(r), "n": int(sub.size)})

    chosen, level = None, None
    for lvl in (target_r, 95.0, 90.0):
        ok = [r for r in rows if r["R"] >= lvl]
        if ok:
            chosen, level = min(ok, key=lambda r: r["mc"]), lvl
            break
    if chosen is None and rows:                      # fall back to best R
        chosen = max(rows, key=lambda r: r["R"])
        level = None
    return {"method": "GFT", "mc": None if chosen is None else chosen["mc"],
            "R_at_mc": None if chosen is None else chosen["R"],
            "level_reached": level, "curve": rows}


# --------------------------------------------------------------------- EMR
def _emr_negloglik(theta, m, dm):
    """Negative log-likelihood of the EMR model.

    Above mc : GR exponential, detection probability = 1
    Below mc : GR exponential x cumulative-normal detection q(m; mu, sigma)
    theta = (mc, beta, mu, sigma)
    """
    mc, beta, mu, sigma = theta
    if beta <= 1e-6 or sigma <= 1e-4 or not np.isfinite(mc):
        return 1e12
    mmin = m.min() - dm / 2.0
    hi = m >= mc - 1e-9
    lo = ~hi

    # unnormalised density
    dens = np.empty_like(m)
    dens[hi] = beta * np.exp(-beta * (m[hi] - mc))
    if lo.any():
        q = norm.cdf(m[lo], loc=mu, scale=sigma)
        dens[lo] = beta * np.exp(-beta * (m[lo] - mc)) * q
    dens = np.clip(dens, 1e-300, None)

    # normalising constant over [mmin, inf)
    grid = np.arange(mmin, mc, dm / 4.0)
    if grid.size:
        qg = norm.cdf(grid, loc=mu, scale=sigma)
        below = np.trapz(beta * np.exp(-beta * (grid - mc)) * qg, grid)
    else:
        below = 0.0
    z = below + 1.0                      # integral above mc of beta*exp(...) = 1
    return float(-np.sum(np.log(dens / z)))


def mc_emr(mags: np.ndarray, dm: float, search=(2.0, 4.5),
           min_events: int = 200) -> dict:
    """Entire-magnitude-range method (Woessner & Wiemer 2005)."""
    m = np.asarray(mags, dtype=float)
    cands = np.round(np.arange(search[0], search[1] + dm / 2, dm), 5)
    best, rows = None, []
    for mc in cands:
        if np.sum(m >= mc) < min_events:
            continue
        b0, _, _ = aki_b_value(m, mc, dm)
        if not np.isfinite(b0):
            continue
        beta0 = b0 * np.log(10.0)
        res = minimize(
            lambda th: _emr_negloglik(np.r_[mc, th], m, dm),
            x0=[beta0, mc - 0.3, 0.3],
            method="Nelder-Mead",
            options={"maxiter": 2000, "fatol": 1e-6, "xatol": 1e-4},
        )
        k = 4
        aic = 2 * res.fun + 2 * k
        row = {"mc": float(mc), "negloglik": float(res.fun), "aic": float(aic),
               "beta": float(res.x[0]), "mu": float(res.x[1]), "sigma": float(res.x[2])}
        rows.append(row)
        if best is None or row["aic"] < best["aic"]:
            best = row
    return {"method": "EMR", "mc": None if best is None else best["mc"],
            "best": best, "curve": rows}


# --------------------------------------------------------------------- stability
def b_stability(mags: np.ndarray, dm: float, search=(2.0, 4.5),
                min_events: int = 100) -> list[dict]:
    """b-value as a function of the assumed cut-off, the standard stability diagnostic."""
    m = np.asarray(mags, dtype=float)
    out = []
    for mc in np.round(np.arange(search[0], search[1] + dm / 2, dm), 5):
        sub = m[m >= mc - 1e-9]
        if sub.size < min_events:
            continue
        b, s, n = aki_b_value(sub, float(mc), dm)
        out.append({"mc": float(mc), "b": b, "sigma_b": s, "n": n})
    return out


def mc_b_stability(mags: np.ndarray, dm: float, search=(2.0, 4.5),
                   min_events: int = 100, window: float = 0.5) -> dict:
    """Mc from the b-value stability criterion of Wiemer & Wyss (2000).

    Mc is the LOWEST cut-off at which b has stopped drifting: the mean of b
    over [Mc, Mc + window] differs from b(Mc) by no more than the Aki-Utsu
    uncertainty of b(Mc).

    Why this estimator matters here. MAXC, GFT and EMR all read the SHAPE of
    the frequency-magnitude distribution near its mode. When a catalog mixes
    magnitude scales the mode belongs to whichever scale dominates the small
    events, so those three can lock onto the completeness of a SUB-catalog
    rather than of the whole. The stability criterion does not use the mode at
    all, so it is the appropriate cross-check for a mixed ML / mb catalog.
    """
    curve = b_stability(mags, dm, search, min_events)
    crit = ("lowest Mc with |mean b over [Mc, Mc+%.1f] - b(Mc)| <= sigma_b(Mc); "
            "Wiemer & Wyss (2000)" % window)
    if len(curve) < 3:
        return {"mc": None, "b": None, "sigma_b": None, "n": None,
                "window": window, "criterion": crit,
                "note": "b(Mc) curve too short to test stability"}
    mcs = np.array([q["mc"] for q in curve], dtype=float)
    bs = np.array([q["b"] for q in curve], dtype=float)
    sbs = np.array([q["sigma_b"] for q in curve], dtype=float)
    nstep = max(2, int(round(window / dm)))
    dev_mc, dev_sig = [], []
    for i in range(len(mcs)):
        j = int(np.searchsorted(mcs, mcs[i] + window + 1e-9, side="right"))
        if j - i < nstep:
            break
        dev = abs(float(np.mean(bs[i:j])) - float(bs[i]))
        dev_mc.append(float(mcs[i]))
        dev_sig.append(dev / float(sbs[i]) if float(sbs[i]) > 0 else float("inf"))
        if dev <= float(sbs[i]):
            return {"mc": float(mcs[i]), "b": float(bs[i]),
                    "sigma_b": float(sbs[i]), "n": int(curve[i]["n"]),
                    "window": window, "criterion": crit,
                    "closest_mc": float(mcs[i]),
                    "closest_dev_sigma": float(dev_sig[-1])}
    # No cut-off satisfies the criterion. Report HOW BADLY it fails, because a
    # b(Mc) curve that never flattens is itself a diagnostic: it says the
    # departure from a single Gutenberg-Richter law is not a completeness
    # artefact that disappears above some threshold.
    k = int(np.argmin(dev_sig)) if dev_sig else None
    return {"mc": None, "b": None, "sigma_b": None, "n": None,
            "window": window, "criterion": crit,
            "closest_mc": (dev_mc[k] if k is not None else None),
            "closest_dev_sigma": (dev_sig[k] if k is not None else None),
            "mc_range": [float(mcs[0]), float(mcs[-1])],
            "b_range": [float(np.min(bs)), float(np.max(bs))],
            "b_at_mc_range": [{"mc": float(mcs[i]), "b": float(bs[i]),
                               "sigma_b": float(sbs[i]), "n": int(curve[i]["n"])}
                              for i in range(0, len(mcs), max(1, len(mcs) // 12))],
            "monotonic_increasing": bool(np.all(np.diff(bs) > -1e-12)),
            "note": "b never stabilises inside the search range - report this"}


def estimate_mc(mags: np.ndarray, dm: float, maxc_correction: float,
                gft_target: float, search, min_events: int,
                rule: str = "maxc",
                stability_window: float = 0.5) -> dict:
    """Run all three completeness estimators and adopt one of them.

    rule="maxc" (default) adopts the maximum-curvature estimate with the
    Woessner & Wiemer (2005) correction. This is the standard choice and the
    the one reported.

    rule="max" adopts the largest of the three. That sounds conservative but is
    not robust: the goodness-of-fit test degrades whenever the observed FMD
    departs from a single power law, and it then returns an Mc so high that the
    working catalog collapses. GFT and EMR are always reported alongside as
    robustness checks regardless of the rule.
    """
    r_maxc = mc_maxc(mags, dm, maxc_correction)
    r_gft = mc_gft(mags, dm, gft_target, search, min_events)
    r_emr = mc_emr(mags, dm, search, min_events)

    vals = [v for v in (r_maxc["mc"], r_gft["mc"], r_emr["mc"]) if v is not None]
    rule = str(rule).lower().strip()
    if rule == "maxc":
        adopted = r_maxc["mc"]
        label = "MAXC + %.2f (Woessner & Wiemer 2005); GFT and EMR reported as checks" % maxc_correction
    elif rule == "gft":
        adopted = r_gft["mc"] if r_gft["mc"] is not None else r_maxc["mc"]
        label = "GFT at R >= %.0f%% (Wiemer & Wyss 2000)" % gft_target
    elif rule == "emr":
        adopted = r_emr["mc"] if r_emr["mc"] is not None else r_maxc["mc"]
        label = "EMR / AIC (Woessner & Wiemer 2005)"
    elif rule == "max":
        adopted = float(np.max(vals)) if vals else r_maxc["mc"]
        label = "max(MAXC+%.2f, GFT@R>=%.0f%%, EMR) - most conservative" % (maxc_correction, gft_target)
    else:
        raise ValueError("unknown MC_RULE %r - use maxc, gft, emr or max" % rule)
    adopted = float(adopted) if adopted is not None else float("nan")

    spread = None
    if len(vals) >= 2:
        spread = round(float(np.max(vals) - np.min(vals)), 4)

    return {
        "MAXC": r_maxc,
        "GFT": {k: v for k, v in r_gft.items() if k != "curve"},
        "GFT_curve": r_gft["curve"],
        "EMR": {k: v for k, v in r_emr.items() if k != "curve"},
        "EMR_curve": r_emr["curve"],
        "mc_adopted": adopted,
        "mc_rule_key": rule,
        "mc_rule": label,
        "mc_estimates_spread": spread,
        "b_stability": b_stability(mags, dm, search),
        "B_STABILITY": mc_b_stability(mags, dm, search,
                                      window=stability_window),
    }
