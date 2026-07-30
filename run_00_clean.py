"""Step 0: construction of the analysis catalog from the raw ISC export.

Columns are mapped by name, timestamps are parsed under a single explicit
convention with parse failures counted, the magnitude-type census is written to
the report, and the study-area and time-window filters are applied with every
intermediate count logged.

Input:   data/Final_.csv
Output:  data/catalog_clean.csv
         outputs/cleaning_report.json
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zm import config as C
from zm import catalog as cat


def main():
    if not os.path.exists(C.RAW_CATALOG):
        sys.exit(f"ERROR: raw catalog not found at {C.RAW_CATALOG}\n"
                 f"Place the raw ISC export there, or set ZM_BASE.")

    rep = {"input_file": C.RAW_CATALOG}

    # ISC exports often carry a preamble; try a few skiprows values.
    df = None
    for skip in (0, 1, 2, 3, 4, 5, 10, 20, 25):
        try:
            trial = pd.read_csv(C.RAW_CATALOG, skiprows=skip, low_memory=False)
        except Exception:
            continue
        trial.columns = [str(c).strip() for c in trial.columns]
        trial, dups = cat.dedupe_columns(trial)
        m, type_diag = cat.resolve_columns(trial)
        if {"LAT", "LON", "MAG"} <= set(m):
            df, colmap, rep["skiprows"] = trial, m, skip
            rep["duplicate_header_labels"] = dups
            rep["type_column_diagnosis"] = type_diag
            break
    if df is None:
        sys.exit("ERROR: could not identify LAT/LON/MAG columns. "
                 "Open the CSV and rename the header row, or edit zm/catalog.py::_ALIASES.")

    rep["n_raw"] = int(len(df))
    rep["detected_columns"] = colmap
    rep["all_columns"] = list(df.columns)
    print("Detected column mapping:")
    for k, v in colmap.items():
        print(f"   {k:12s} <- {v!r}")
    print(f"   rows: {len(df)}")
    if rep.get("duplicate_header_labels"):
        print("   repeated header labels in the raw file (renamed to be unique):")
        for k, v in rep["duplicate_header_labels"].items():
            print(f"      {k!r} -> {v}")
    for c, d in (rep.get("type_column_diagnosis") or {}).items():
        role = ("MAGNITUDE type" if colmap.get("MAG_TYPE") == c else
                "EVENT type" if colmap.get("EVENT_TYPE") == c else "unused")
        print(f"   column {c!r} -> {role}  (mag-code {d['magnitude_code_fraction']:.2f}, "
              f"event-code {d['event_code_fraction']:.2f}, e.g. {d['examples']})")

    out = pd.DataFrame()
    for k in ("LAT", "LON", "DEPTH", "MAG"):
        out[k] = pd.to_numeric(cat.as_series(df, colmap[k]), errors="coerce") if k in colmap else np.nan
    out["MAG_TYPE"] = (cat.as_series(df, colmap["MAG_TYPE"]).astype(str)
                       if "MAG_TYPE" in colmap else "unknown")
    if "AUTHOR" in colmap:
        out["AUTHOR"] = cat.as_series(df, colmap["AUTHOR"]).astype(str)
    if "EVENT_TYPE" in colmap:
        out["EVENT_TYPE"] = cat.as_series(df, colmap["EVENT_TYPE"]).astype(str)

    dt, dtrep = cat.parse_datetime(
        cat.as_series(df, colmap["DATE"]) if "DATE" in colmap else df.iloc[:, 0],
        cat.as_series(df, colmap["TIME"]) if "TIME" in colmap else None)
    out["datetime"] = dt
    rep["datetime_parsing"] = dtrep
    print(f"   datetime format: {dtrep['format_used']} "
          f"({dtrep['parsed']}/{dtrep['total']} parsed, {dtrep['failed']} failed)")

    steps = []

    def _drop(mask, label):
        n0 = len(out)
        kept = out[mask].copy()
        steps.append({"filter": label, "removed": int(n0 - len(kept)), "remaining": int(len(kept))})
        print(f"   {label:38s} -{n0-len(kept):6d}  -> {len(kept)}")
        return kept

    print("\nFiltering:")
    out = _drop(out["datetime"].notna(), "valid datetime")
    out = _drop(out[["LAT", "LON", "MAG"]].notna().all(axis=1), "LAT/LON/MAG present")
    if "EVENT_TYPE" in out.columns:
        # ISC uses two-letter codes, NOT English words, so a word-based filter
        #   ke/se known/suspected earthquake     de damaging   fe felt
        #   ki/si induced   km/sm mining   kx/sx experimental explosion
        #   kn/sn nuclear   kr/sr rockburst   ls landslide   qb quarry blast
        ANTHROPOGENIC = {"ki", "si", "km", "sm", "kx", "sx", "kn", "sn",
                         "kr", "sr", "ls", "qb", "ex"}
        ev = out["EVENT_TYPE"].astype(str).str.strip().str.lower()
        ev_counts = ev.value_counts().to_dict()
        rep["event_type_counts"] = {str(k): int(v) for k, v in ev_counts.items()}
        print("\nEvent types present:")
        for k, v in sorted(ev_counts.items(), key=lambda kv: -kv[1]):
            tag = "ANTHROPOGENIC - removed" if k in ANTHROPOGENIC else "tectonic - kept"
            print(f"   {k:8s} {v:6d}   {tag}")
        bad = ev.isin(ANTHROPOGENIC) | ev.str.contains(
            "explosion|quarry|blast|mining|nuclear|induced|rockburst|landslide",
            na=False)
        rep["n_anthropogenic_removed"] = int(bad.sum())
        out = _drop(~bad, "natural tectonic events only")
    out = _drop((out["LAT"].between(C.LAT_MIN, C.LAT_MAX)) &
                (out["LON"].between(C.LON_MIN, C.LON_MAX)), "inside study area")
    out = _drop((out["datetime"].dt.year >= C.START_YEAR) &
                (out["datetime"] <= pd.Timestamp(C.END_DATE) + pd.Timedelta(days=1)),
                f"{C.START_YEAR} to {C.END_DATE}")
    out = _drop(out["MAG"] > 0, "positive magnitude")
    out = _drop((out["DEPTH"].isna()) | (out["DEPTH"] >= 0), "non-negative depth")

    # ---------------------------------------------------------- magnitude types
    mt = out["MAG_TYPE"].astype(str).str.strip()
    counts = mt.value_counts().to_dict()
    fam = mt.str.lower().value_counts().to_dict()
    rep["magnitude_type_families"] = {str(k): int(v) for k, v in fam.items()}
    _n = max(int(mt.size), 1)
    rep["magnitude_type_dominant_share"] = round(max(fam.values()) / _n, 4)
    rep["magnitude_type_counts_raw"] = {str(k): int(v) for k, v in counts.items()}
    print("\nMagnitude types present (as reported):")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"   {k:12s} {v:6d}")
    print("Magnitude type FAMILIES (case-insensitive):")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"   {k:12s} {v:6d}   ({100.0*v/_n:5.1f} %)")
    if rep["magnitude_type_dominant_share"] < 0.95:
        print("   *** THE CATALOG IS NOT MAGNITUDE-HOMOGENEOUS ***")
        print("   *** The word 'homogeneous' must not be used unless MODE is")
        print("   *** 'sensitivity'/'full', which converts everything to Mw.")

    out["MAG_REPORTED"] = out["MAG"]
    if C.MAG_HOMOGENISATION == "convert":
        mw, conv = cat.to_mw(out["MAG"].to_numpy(), out["MAG_TYPE"].to_numpy())
        # Reported magnitudes sit on a 0.1 grid. A linear conversion whose slope
        # is not 1 moves them off it (0.85*ML+0.60 lands on a 0.085 grid), so the
        # binned frequency-magnitude distribution turns into a comb of spikes and
        # empty bins. That breaks the goodness-of-fit completeness test and drives
        # Mc far too high. Snap back onto the reporting grid.
        mw = np.round(np.asarray(mw, dtype=float) / C.DM) * C.DM
        out["MAG"] = np.round(mw, 4)
        conv["rounded to DM grid"] = float(C.DM)
        rep["homogenisation"] = {"mode": "convert", "conversions": conv,
                                 "note": "Scordilis (2006) mb->Mw and Ms->Mw; ML->Mw 0.85*ML+0.60"}
        print(f"\n   Homogenisation: converted to Mw -> {conv}")
    elif C.MAG_HOMOGENISATION == "preferred":
        keep = out["MAG_TYPE"].str.strip().isin(C.PREFERRED_MAG_TYPES)
        out = _drop(keep, f"magnitude type in {C.PREFERRED_MAG_TYPES}")
        rep["homogenisation"] = {"mode": "preferred", "kept_types": list(C.PREFERRED_MAG_TYPES)}
    else:
        rep["homogenisation"] = {"mode": "none",
                                 "warning": "catalog is NOT magnitude-homogeneous"}
        print("\n   WARNING: no homogenisation applied; the catalog is not "
          "magnitude-homogeneous.")

    # ---------------------------------------------------------- duplicates
    n0 = len(out)
    out = out.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    dt_s = out["datetime"].diff().dt.total_seconds()
    dup = (dt_s < 2.0) & (out["LAT"].diff().abs() < 0.02) & (out["LON"].diff().abs() < 0.02)
    out = out[~dup.fillna(False)].reset_index(drop=True)
    steps.append({"filter": "near-duplicate removal", "removed": int(n0 - len(out)),
                  "remaining": int(len(out))})
    print(f"   {'near-duplicate removal':38s} -{n0-len(out):6d}  -> {len(out)}")

    out = cat.add_time_features(out)

    cols = ["datetime", "LAT", "LON", "DEPTH", "MAG", "MAG_REPORTED", "MAG_TYPE",
            "time_diff", "time_diff_log"]
    out[[c for c in cols if c in out.columns]].to_csv(C.CLEAN_CATALOG, index=False)

    rep["filter_steps"] = steps
    rep["n_clean"] = int(len(out))
    rep["date_min"] = str(out["datetime"].min())
    rep["date_max"] = str(out["datetime"].max())
    rep["mag_min"] = float(out["MAG"].min())
    rep["mag_max"] = float(out["MAG"].max())
    rep["n_depth_missing"] = int(out["DEPTH"].isna().sum())

    with open(os.path.join(C.OUT_DIR, "cleaning_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\nOK  {len(out)} events  ->  {C.CLEAN_CATALOG}")
    print(f"    {rep['date_min']}  ..  {rep['date_max']}")
    print(f"    M {rep['mag_min']:.2f} .. {rep['mag_max']:.2f}")
    print(f"    report -> outputs/cleaning_report.json")


if __name__ == "__main__":
    main()
