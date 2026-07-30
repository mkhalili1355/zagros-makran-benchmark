"""Temporal ETAS model (Ogata, 1988).

The log-likelihood is

    L = sum_i log lambda(t_i) - INT_0^T lambda(t) dt

with conditional intensity

    lambda(t) = mu + sum_{t_j < t} K exp(alpha (m_j - Mc)) (t - t_j + c)^(-p)

The background term contributes mu*T to the integral, not one count per event,
and the Omori integral is evaluated in closed form for p = 1 and p != 1
separately. Parameters are estimated by maximum likelihood on training events
above Mc with multiple random restarts.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from . import config as C


def _unpack(theta):
    """theta is in log-space for mu, K, c and for (p-1); alpha is linear."""
    log_mu, log_K, log_c, alpha, log_pm1 = theta
    return (np.exp(log_mu), np.exp(log_K), np.exp(log_c),
            alpha, 1.0 + np.exp(log_pm1))


def _pack(mu, K, c, alpha, p):
    return np.array([np.log(mu), np.log(K), np.log(c), alpha,
                     np.log(max(p - 1.0, 1e-6))])


def etas_loglik(theta, t, m, mc, T):
    """Exact temporal ETAS log-likelihood on [0, T]."""
    mu, K, c, alpha, p = _unpack(theta)
    if not np.all(np.isfinite([mu, K, c, alpha, p])):
        return 1e12
    if mu <= 0 or K <= 0 or c <= 0 or p <= 1.0 + 1e-9 or abs(alpha) > 5:
        return 1e12

    g = K * np.exp(alpha * (m - mc))                 # productivity of each event

    # ---- sum of log conditional intensities
    lam = np.full(t.size, mu, dtype=float)
    for j in range(1, t.size):
        dt = t[j] - t[:j]
        lam[j] += np.sum(g[:j] * (dt + c) ** (-p))
    if np.any(lam <= 0):
        return 1e12
    s1 = np.sum(np.log(lam))

    # ---- integral of the intensity  (background + ALL triggered terms)
    # Written with expm1 so it stays accurate as p -> 1. The naive form
    #     (c**(1-p) - (T-t+c)**(1-p)) / (p-1)
    # is a difference of two numbers that both approach 1 divided by a quantity
    # approaching 0, so it loses all precision near p = 1 and makes the
    # likelihood surface look flat and noisy exactly where the optimiser is
    # searching. The limit is the logarithmic Omori case, ln((T-t+c)/c).
    q = 1.0 - p
    upper = T - t + c
    if abs(q) < 1e-12:
        trig = np.log(upper / c)
    else:
        trig = (np.expm1(q * np.log(c)) - np.expm1(q * np.log(upper))) / (-q)
    integ = mu * T + np.sum(g * trig)

    ll = s1 - integ
    return -ll if np.isfinite(ll) else 1e12


def fit_etas(times_days: np.ndarray, mags: np.ndarray, mc: float,
             n_restarts: int = None, seed: int = None) -> dict:
    """Multi-start MLE. `times_days` must start at 0 and be sorted."""
    t = np.asarray(times_days, float)
    m = np.asarray(mags, float)
    order = np.argsort(t, kind="mergesort")
    t, m = t[order], m[order]
    t = t - t[0]
    T = float(t[-1])
    n_restarts = n_restarts or C.ETAS_RESTARTS
    rng = np.random.default_rng(seed or C.ETAS_SEED)

    best = None
    for i in range(n_restarts):
        if i == 0:
            x0 = _pack(len(t) / T * 0.4, 0.02, 0.01, 1.0, 1.1)
        else:
            x0 = _pack(10 ** rng.uniform(-2.5, 0.5),
                       10 ** rng.uniform(-2.5, 0.5),
                       10 ** rng.uniform(-3.0, 0.0),
                       rng.uniform(0.3, 2.5),
                       1.0 + 10 ** rng.uniform(-2.0, 0.0))
        try:
            res = minimize(etas_loglik, x0, args=(t, m, mc, T), method="Nelder-Mead",
                           options={"maxiter": 20000, "maxfev": 20000,
                                    "fatol": 1e-8, "xatol": 1e-8})
        except Exception:
            continue
        if res.fun >= 1e11 or not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        return {"converged": False}

    mu, K, c, alpha, p = _unpack(best.x)

    # ---- numerical standard errors from the Hessian of the negative logL
    se = {}
    try:
        h = 1e-4
        k = best.x.size
        H = np.zeros((k, k))
        f0 = etas_loglik(best.x, t, m, mc, T)
        for a in range(k):
            for b in range(a, k):
                xa, xb = best.x.copy(), best.x.copy()
                xab = best.x.copy()
                xa[a] += h
                xb[b] += h
                xab[a] += h
                xab[b] += h
                H[a, b] = H[b, a] = (etas_loglik(xab, t, m, mc, T)
                                     - etas_loglik(xa, t, m, mc, T)
                                     - etas_loglik(xb, t, m, mc, T) + f0) / (h * h)
        cov = np.linalg.inv(H)
        sd_log = np.sqrt(np.clip(np.diag(cov), 0, None))
        # delta method back to the natural scale
        se = {"mu": float(mu * sd_log[0]), "K": float(K * sd_log[1]),
              "c": float(c * sd_log[2]), "alpha": float(sd_log[3]),
              "p": float((p - 1.0) * sd_log[4])}
    except Exception:
        se = {kk: None for kk in ("mu", "K", "c", "alpha", "p")}

    # p is parameterised as 1 + exp(theta), so it can only approach 1 from above.
    # If the optimiser drives it to that edge the Omori decay exponent is not
    # identified by the data and p must not be quoted as a fitted value.
    p_at_bound = bool((p - 1.0) < 1e-4)

    return {"converged": True, "mu_per_day": float(mu), "K": float(K),
            "c_days": float(c), "alpha": float(alpha), "p": float(p),
            "p_at_lower_bound": p_at_bound,
            "p_note": ("p reached the lower bound p -> 1 (logarithmic Omori limit); "
                       "report it as p ~ 1.0, not as a precisely estimated value")
                      if p_at_bound else "p estimated in the interior",
            "se": se, "loglik": float(-best.fun), "n_events": int(t.size),
            "T_days": T, "mc": float(mc), "n_restarts": int(n_restarts),
            "n_params": 5,
            "AIC": float(2 * 5 + 2 * best.fun),
            "BIC": float(5 * np.log(t.size) + 2 * best.fun)}


def conditional_intensity(params: dict, t_hist: np.ndarray, m_hist: np.ndarray,
                          t_eval: np.ndarray, mc: float) -> np.ndarray:
    """lambda(t) at t_eval given the STRICTLY PRIOR history (t_hist < t_eval)."""
    mu, K, c = params["mu_per_day"], params["K"], params["c_days"]
    alpha, p = params["alpha"], params["p"]
    g = K * np.exp(alpha * (np.asarray(m_hist, float) - mc))
    th = np.asarray(t_hist, float)
    out = np.empty(len(t_eval), dtype=float)
    for i, te in enumerate(np.asarray(t_eval, float)):
        prior = th < te                              # strictly causal
        if not prior.any():
            out[i] = mu
        else:
            out[i] = mu + np.sum(g[prior] * (te - th[prior] + c) ** (-p))
    return out


def etas_magnitude_forecast(b: float, mc: float, n: int) -> np.ndarray:
    """ETAS says nothing about magnitude: the forecast is the GR expectation.

    E[M | M >= mc] = mc + 1 / (b * ln 10)

    The value is constant across events, so an ETAS-derived magnitude
    baseline is by construction the Gutenberg-Richter mean and carries no
    discriminative information.
    """
    return np.full(n, mc + 1.0 / (b * np.log(10.0)), dtype=float)
