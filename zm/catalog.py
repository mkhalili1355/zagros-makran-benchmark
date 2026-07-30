"""Catalog loading, homogenisation, declustering, windowing and splitting.

Column mapping is performed by name. Timestamps follow one explicit convention
with a documented fallback, and parse failures are counted. Magnitude types are
homogenised or filtered before any frequency-magnitude analysis. Depth is
imputed with the training median. The chronological split is applied to events
and windows are formed within each partition. The prediction target is the raw
magnitude; only the input features are min-max scaled, with the scaler fitted on
the training partition.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------- [F1]
_ALIASES = {
    "DATE": ["date", "origin date", "yyyy/mm/dd", "day"],
    "TIME": ["time", "origin time", "hh:mm:ss"],
    "LAT": ["lat", "latitude"],
    "LON": ["lon", "long", "longitude"],
    "DEPTH": ["depth", "dep", "depth_km"],
    "MAG": ["mag", "magnitude", "m"],
    "MAG_TYPE": ["magtype", "mag type", "magnitude type", "type", "mtype"],
    "AUTHOR": ["author", "agency", "contributor"],
    "EVENT_TYPE": ["event type", "eventtype", "etype"],
}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum() or ch == " ").strip()


# --------------------------------------------------------------------- [F1a]
# ISC bulletin CSV headers are space-padded and the format repeats some labels:
# there are TWO columns called TYPE (event type, then magnitude type) and TWO
# called AUTHOR (location author, then magnitude author). Stripping whitespace
# from the header therefore CREATES duplicate labels, and df["TYPE"] then
# returns a two-column DataFrame instead of a Series. Both problems are fixed
# here rather than downstream.

_MAG_TYPE_CODES = {
    "mb", "mb1", "mbtmp", "ms", "ms1", "ms7", "msz", "ml", "mlv", "mlr", "mn",
    "mw", "mww", "mwc", "mwb", "mwr", "mwp", "md", "mpv", "mpva", "m", "me",
    "mjma", "mbLg".lower(), "mblg",
}
_EVENT_TYPE_CODES = {
    "ke", "se", "kr", "sr", "km", "sm", "ki", "si", "kx", "sx", "kn", "sn",
    "uk", "ls", "de", "qb", "kh", "sh", "ex", "eq", "earthquake", "quarry blast",
}


def dedupe_columns(df: pd.DataFrame):
    """Make every column label unique, appending .1, .2 ... in order.

    Returns (new_df, duplicates_found) where duplicates_found maps the original
    label to the list of unique labels it became, for the cleaning report.
    """
    seen, new = {}, []
    for c in df.columns:
        c = str(c)
        if c in seen:
            seen[c] += 1
            new.append("{}.{}".format(c, seen[c]))
        else:
            seen[c] = 0
            new.append(c)
    dups = {}
    for c, n in seen.items():
        if n:
            dups[c] = [c] + ["{}.{}".format(c, i) for i in range(1, n + 1)]
    out = df.copy()
    out.columns = new
    return out, dups


def as_series(df: pd.DataFrame, col) -> pd.Series:
    """Always return a Series, even if the label is still duplicated."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def _classify_type_columns(df: pd.DataFrame, cands):
    """Decide which TYPE-like column is the magnitude type and which is the
    event type by inspecting the VALUES, never the column position."""
    diag = {}
    for c in cands:
        v = as_series(df, c).astype(str).str.strip().str.lower()
        v = v[(v != "") & (v != "nan")]
        n = max(len(v), 1)
        diag[c] = {
            "n_values": int(len(v)),
            "magnitude_code_fraction": round(float(v.isin(_MAG_TYPE_CODES).sum()) / n, 3),
            "event_code_fraction": round(float(v.isin(_EVENT_TYPE_CODES).sum()) / n, 3),
            "examples": sorted(v.unique().tolist())[:6],
        }
    mag_col = ev_col = None
    ranked_mag = sorted(cands, key=lambda c: -diag[c]["magnitude_code_fraction"])
    if ranked_mag and diag[ranked_mag[0]]["magnitude_code_fraction"] >= 0.5:
        mag_col = ranked_mag[0]
    rest = [c for c in cands if c != mag_col]
    ranked_ev = sorted(rest, key=lambda c: -diag[c]["event_code_fraction"])
    if ranked_ev and diag[ranked_ev[0]]["event_code_fraction"] >= 0.5:
        ev_col = ranked_ev[0]
    return mag_col, ev_col, diag


def resolve_columns(df: pd.DataFrame):
    """map_columns plus value-based disambiguation of repeated TYPE columns."""
    colmap = map_columns(df)
    cands = [c for c in df.columns if "type" in _norm(c)]
    diag = {}
    if len(cands) >= 1:
        mag_col, ev_col, diag = _classify_type_columns(df, cands)
        if mag_col is not None:
            colmap["MAG_TYPE"] = mag_col
        elif colmap.get("MAG_TYPE") in cands and len(cands) > 1:
            colmap.pop("MAG_TYPE", None)
        if ev_col is not None:
            colmap["EVENT_TYPE"] = ev_col
        elif colmap.get("EVENT_TYPE") in cands and colmap.get("EVENT_TYPE") == mag_col:
            colmap.pop("EVENT_TYPE", None)
    return colmap, diag


def map_columns(df: pd.DataFrame) -> dict:
    """Resolve canonical column names by fuzzy matching. Reports what it found."""
    found, used = {}, set()
    norm_cols = {c: _norm(c) for c in df.columns}
    for canon, aliases in _ALIASES.items():
        for col, nc in norm_cols.items():
            if col in used:
                continue
            if nc == _norm(canon) or nc in [_norm(a) for a in aliases]:
                found[canon] = col
                used.add(col)
                break
        if canon in found:
            continue
        for col, nc in norm_cols.items():                # substring fallback
            if col in used:
                continue
            if any(_norm(a) and _norm(a) in nc for a in aliases):
                found[canon] = col
                used.add(col)
                break
    return found


# --------------------------------------------------------------------- [F2]
_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]


def parse_datetime(date_s: pd.Series, time_s: pd.Series | None) -> tuple[pd.Series, dict]:
    raw = date_s.astype(str).str.strip()
    if time_s is not None:
        raw = raw + " " + time_s.astype(str).str.strip()
    best, best_fmt, best_ok = None, None, -1
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        except Exception:
            continue
        ok = int(parsed.notna().sum())
        if ok > best_ok:
            best, best_fmt, best_ok = parsed, fmt, ok
    fallback = pd.to_datetime(raw, errors="coerce")
    if int(fallback.notna().sum()) > best_ok:
        best, best_fmt, best_ok = fallback, "inferred", int(fallback.notna().sum())
    return best, {"format_used": best_fmt, "parsed": best_ok, "total": int(len(raw)),
                  "failed": int(len(raw) - best_ok)}


# --------------------------------------------------------------------- [F3]
def to_mw(mag: np.ndarray, mtype: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert mb / Ms / ML to a common Mw-equivalent scale.

    Regressions (global / regional, widely used for Iranian catalogs):
      Mw = 1.0319 * mb  - 0.0223      (Scordilis 2006, 3.5<=mb<=6.2)
      Mw = 0.67  * Ms   + 2.07        (Scordilis 2006, 3.0<=Ms<=6.1)
      Mw = 0.85  * ML   + 0.60        (commonly used ML->Mw for Iran)
      Mw = Mw   (identity)

    The conversion coefficients follow Scordilis (2006). An alternative
    region-specific relation, change them here - nothing else needs editing.
    """
    mag = np.asarray(mag, dtype=float)
    # Use pandas string methods, not np.char: a Series of Python strings becomes
    # an object-dtype array, and np.char requires a fixed-width string dtype.
    ser = pd.Series(np.asarray(mtype, dtype=object)).astype("string")
    ser = ser.fillna("").str.strip().str.lower()
    out = mag.copy()
    counts = {}

    def _starts(prefix: str) -> np.ndarray:
        return ser.str.startswith(prefix).fillna(False).to_numpy(dtype=bool)

    def _apply(mask, fn, label):
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            out[mask] = fn(mag[mask])
            counts[label] = int(mask.sum())

    is_mb = _starts("mb")
    is_ms = _starts("ms")
    is_mw = _starts("mw")
    # "ml" must not swallow "mw"/"ms"; also catch md/mn/mjma-style local scales
    is_ml = _starts("ml") & ~(is_mw | is_ms)
    other = ~(is_mb | is_ms | is_ml | is_mw)

    # Report how many events fall outside the published validity ranges of the
    # Scordilis (2006) regressions.
    n_mb_out = int((is_mb & ((mag < 3.5) | (mag > 6.2))).sum())
    n_ms_out = int((is_ms & ((mag < 3.0) | (mag > 6.1))).sum())
    if n_mb_out:
        counts["mb outside 3.5-6.2 (extrapolated)"] = n_mb_out
    if n_ms_out:
        counts["Ms outside 3.0-6.1 (extrapolated)"] = n_ms_out

    _apply(is_mb, lambda x: 1.0319 * x - 0.0223, "mb->Mw")
    _apply(is_ms, lambda x: 0.67 * x + 2.07, "Ms->Mw")
    _apply(is_ml, lambda x: 0.85 * x + 0.60, "ML->Mw")
    _apply(is_mw, lambda x: x, "Mw (identity)")
    _apply(other, lambda x: x, "unknown type (kept as reported)")
    return out, counts


# --------------------------------------------------------------------- loading
def load_clean_catalog(path: str | None = None) -> pd.DataFrame:
    """Load the cleaned catalog produced by run_00_clean.py."""
    path = path or C.CLEAN_CATALOG
    if not os.path.exists(path):
        raise SystemExit(
            "\n*** catalog_clean.csv not found ***\n"
            "Expected at: %s\n" % path +
            "This file is produced by STEP 0. Run run_00_clean.py first,\n"
            "then re-run this step.")
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inter-event time in seconds and its log10.

    The first event has no predecessor, so both features are NaN there. This is
    computed in NumPy rather than with Series.iloc assignment, because pandas
    silently DISCARDS an .iloc write into the result of .diff().dt on a
    datetimelike column, which left the first row holding a wrong value instead
    of NaN.
    """
    df = df.copy()
    t = df["datetime"].to_numpy("datetime64[s]").astype("float64")
    dt = np.full(len(df), np.nan, dtype=float)
    if len(df) > 1:
        dt[1:] = np.diff(t)
    df["time_diff"] = dt
    with np.errstate(invalid="ignore"):
        df["time_diff_log"] = np.log10(np.clip(dt, 1.0, None))
    assert np.isnan(df["time_diff"].to_numpy()[0]), "first inter-event time must be NaN"
    return df


# --------------------------------------------------------------------- decluster
def gardner_knopoff(df: pd.DataFrame) -> np.ndarray:
    """Gardner & Knopoff (1974) window declustering.

    Returns a boolean array: True = mainshock/background, False = dependent event.
    Used for DESCRIPTIVE statistics only.
    """
    m = df["MAG"].to_numpy(float)
    t = df["datetime"].to_numpy("datetime64[s]").astype("int64") / 86400.0  # days
    lat = np.radians(df["LAT"].to_numpy(float))
    lon = np.radians(df["LON"].to_numpy(float))

    d_km = 10.0 ** (0.1238 * m + 0.983)
    t_day = np.where(m >= 6.5,
                     10.0 ** (0.032 * m + 2.7389),
                     10.0 ** (0.5409 * m - 0.547))

    n = len(df)
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-m, kind="mergesort")      # largest first
    R = 6371.0
    for i in order:
        if not keep[i]:
            continue
        dt = t - t[i]
        cand = np.where((dt > 0) & (dt <= t_day[i]) & keep)[0]
        if cand.size == 0:
            continue
        dlat = lat[cand] - lat[i]
        dlon = lon[cand] - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[cand]) * np.sin(dlon / 2) ** 2
        dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        hit = cand[(dist <= d_km[i]) & (m[cand] < m[i])]
        keep[hit] = False
    return keep


# --------------------------------------------------------------------- [F5][F6]
class MinMaxScaler3D:
    """Min-Max scaler fitted on 2-D event features, applied to 3-D windows."""

    def __init__(self):
        self.lo = None
        self.hi = None

    def fit(self, X2d: np.ndarray) -> "MinMaxScaler3D":
        self.lo = np.nanmin(X2d, axis=0)
        self.hi = np.nanmax(X2d, axis=0)
        rng = self.hi - self.lo
        rng[rng == 0] = 1.0
        self._rng = rng
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lo) / self._rng


def make_windows(X2d: np.ndarray, y1d: np.ndarray, w: int):
    """Sliding windows. Window i uses events [i-w, i) to predict event i.

    Strictly causal: the target is never part of its own input window.
    """
    n = len(X2d)
    if n <= w:
        return (np.empty((0, w, X2d.shape[1])), np.empty((0,)), np.empty((0,), dtype=int))
    idx = np.arange(w, n)
    Xs = np.stack([X2d[i - w:i] for i in idx])
    ys = y1d[idx]
    return Xs, ys, idx


def build_dataset(df: pd.DataFrame, w: int, report: dict | None = None) -> dict:
    """Full split-then-window construction with train-only preprocessing.

    Returns a dict with X/y for train, val, test plus bookkeeping indices.
    """
    n = len(df)
    i_tr = int(C.TRAIN_FRAC * n)
    i_va = int((C.TRAIN_FRAC + C.VAL_FRAC) * n)

    # ---- [F4] impute DEPTH and time_diff_log with TRAINING statistics only
    d = df["DEPTH"].to_numpy(float).copy()
    depth_med = float(np.nanmedian(d[:i_tr]))
    n_imputed = int(np.isnan(d).sum())
    d[np.isnan(d)] = depth_med

    tdl = df["time_diff_log"].to_numpy(float).copy()
    tdl_med = float(np.nanmedian(tdl[:i_tr][np.isfinite(tdl[:i_tr])]))
    tdl[~np.isfinite(tdl)] = tdl_med

    feat = df[["LAT", "LON"]].to_numpy(float)
    X2d = np.column_stack([feat[:, 0], feat[:, 1], d, df["MAG"].to_numpy(float), tdl])
    y1d = df["MAG"].to_numpy(float)                      # [F6] physical units

    parts = {"train": (0, i_tr), "val": (i_tr, i_va), "test": (i_va, n)}

    scaler = MinMaxScaler3D().fit(X2d[:i_tr])            # train-only fit
    Xs = scaler.transform(X2d)

    out = {"w": w, "scaler_lo": scaler.lo.tolist(), "scaler_hi": scaler.hi.tolist(),
           "depth_median_train": depth_med, "n_depth_imputed": n_imputed,
           "split_event_index": {"train": [0, i_tr], "val": [i_tr, i_va], "test": [i_va, n]}}

    for name, (lo, hi) in parts.items():
        Xw, yw, idx = make_windows(Xs[lo:hi], y1d[lo:hi], w)
        out[f"X_{name}"] = Xw.astype("float32")
        out[f"y_{name}"] = yw.astype("float32")
        out[f"idx_{name}"] = (idx + lo).astype(int)      # index into the full catalog
        out[f"n_{name}"] = int(len(yw))

    if report is not None:
        report[f"dataset_w{w}"] = {
            "n_events": n, "n_train_windows": out["n_train"],
            "n_val_windows": out["n_val"], "n_test_windows": out["n_test"],
            "depth_median_train_km": depth_med, "n_depth_imputed": n_imputed,
            "split_event_index": out["split_event_index"],
        }
    return out


def window_durations(df: pd.DataFrame, idx: np.ndarray, w: int) -> dict:
    """Wall-clock duration statistics of the input windows (days)."""
    t = df["datetime"].to_numpy("datetime64[s]").astype("int64") / 86400.0
    dur = np.array([t[i - 1] - t[i - w] for i in idx if i - w >= 0], dtype=float)
    if dur.size == 0:
        return {}
    q = np.percentile(dur, [10, 25, 50, 75, 90])
    return {"w": w, "n": int(dur.size), "mean_days": float(dur.mean()),
            "p10": float(q[0]), "q1": float(q[1]), "median_days": float(q[2]),
            "q3": float(q[3]), "p90": float(q[4]),
            "min_days": float(dur.min()), "max_days": float(dur.max())}
