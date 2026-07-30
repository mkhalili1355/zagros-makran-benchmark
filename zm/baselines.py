"""Catalog-derived ranking baselines.

These quantities are alarm scores rather than magnitude predictions, so they are
evaluated with AUC and Molchan statistics and never with a mean absolute error.
Each score is computed strictly from events preceding the target event, on the
same test windows used by the models.

Sign convention: a higher score denotes a higher alarm level. The time elapsed
since the last large event has the opposite natural sense and is therefore
negated, which is recorded in DEFINITIONS.
"""
from __future__ import annotations
import numpy as np

from . import config as C

COUNT_THRESHOLD = 3.5      # "event count M >= 3.5 in window"
ELAPSED_THRESHOLD = 4.5    # "time since last M >= 4.5 event"

DEFINITIONS = {
    "random": "Uniform random score, fixed seed. Reference for AUC = 0.50.",
    "recent_seismicity_rate": (
        "w / (t[i-1] - t[i-w]) = mean event rate in events per day over the "
        "input window."),
    "event_count_M35_in_window": (
        "Number of events with M >= %.1f among the w events of the input "
        "window." % COUNT_THRESHOLD),
    "recent_max_magnitude": "Maximum magnitude among the w events of the input window.",
    "time_since_last_M45": (
        "NEGATIVE elapsed days since the most recent prior M >= %.1f event, so "
        "that a shorter elapsed time gives a higher alarm score. Windows with "
        "no prior qualifying event are assigned the largest observed elapsed "
        "time. AUC for the un-negated quantity is 1 - AUC." % ELAPSED_THRESHOLD),
}


def ranking_baselines(M_all, t_all, idx_test, w, seed=None):
    """
    Parameters
    ----------
    M_all : (n_events,) magnitudes of the working catalog, chronological
    t_all : (n_events,) event times in DAYS, chronological
    idx_test : (n_test,) index of the TARGET event of each test window; the
               input window is M_all[i-w:i], so nothing at or after i is used
    w : window length in events

    Returns
    -------
    dict name -> (n_test,) float score array, higher = stronger alarm
    """
    seed = C.RNG_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    idx_test = np.asarray(idx_test, dtype=int)
    M_all = np.asarray(M_all, dtype=float)
    t_all = np.asarray(t_all, dtype=float)
    n = idx_test.size

    out = {}
    out["random"] = rng.random(n)

    dur = np.array([max(t_all[i - 1] - t_all[i - w], 1e-6) for i in idx_test])
    out["recent_seismicity_rate"] = w / dur

    out["event_count_M35_in_window"] = np.array(
        [np.count_nonzero(M_all[i - w:i] >= COUNT_THRESHOLD) for i in idx_test],
        dtype=float)

    out["recent_max_magnitude"] = np.array(
        [M_all[i - w:i].max() for i in idx_test], dtype=float)

    elapsed = np.full(n, np.nan)
    for k, i in enumerate(idx_test):
        prior = np.flatnonzero(M_all[:i] >= ELAPSED_THRESHOLD)
        if prior.size:
            elapsed[k] = t_all[i - 1] - t_all[prior[-1]]
    if np.all(np.isnan(elapsed)):
        elapsed[:] = 0.0
    else:
        elapsed[np.isnan(elapsed)] = np.nanmax(elapsed)
    out["time_since_last_M45"] = -elapsed

    return out


# Display names used in the figures and tables
DISPLAY = {
    "random": "Random",
    "recent_seismicity_rate": "Recent seismicity rate",
    "event_count_M35_in_window": "Event count M >= 3.5 in window",
    "recent_max_magnitude": "Recent maximum magnitude",
    "time_since_last_M45": "Time since last M >= 4.5",
    "etas_conditional_intensity": "ETAS occurrence intensity",
    "median_climatology": "Climatology (median)",
    "mean_climatology": "Climatology (mean)",
    "persistence_last": "Persistence (last event)",
    "persistence_window_max": "Persistence (window max)",
    "persistence_window_mean": "Persistence (window mean)",
    "gr_expectation": "Gutenberg-Richter expectation",
}

# The three catalog-derived rankings drawn alongside the architectures in Fig. 5
FIG5_BASELINES = [
    "recent_seismicity_rate",
    "event_count_M35_in_window",
    "recent_max_magnitude",
]
