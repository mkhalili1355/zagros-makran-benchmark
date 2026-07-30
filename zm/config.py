"""Central configuration for the Zagros-Makran benchmark.

Every constant used in the pipeline is defined here; no script defines its own.

MODE selects the configuration and is read from the environment variable
ZM_MODE (default "minimal").

  minimal      Magnitudes are used as reported, one parameter budget of
               approximately 57,000, two window lengths, four architectures,
               50 epochs, batch 16, patience 8. This is the configuration used
               for the reported results.
  full         Magnitudes are converted to Mw (Scordilis, 2006), a second
               parameter budget of 8,000 is added and training is extended to
               200 epochs.
  sensitivity  As minimal, with magnitude conversion enabled. Intended to be run
               into a separate output tree, for example
               ZM_BASE=/path/to/run_sens python run_00_clean.py

Setting ZM_QUICK=1 reduces the run to 2 seeds and 8 epochs for testing.
"""
from __future__ import annotations
import os

MODE = os.environ.get("ZM_MODE", "minimal")
assert MODE in ("minimal", "full", "sensitivity"), "MODE must be minimal, full or sensitivity"

# ---------------------------------------------------------------- paths
BASE_DIR = os.environ.get("ZM_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUT_DIR, "weights")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

RAW_FILENAME = "Final_.csv"                                   # raw ISC export
RAW_CATALOG = os.path.join(DATA_DIR, RAW_FILENAME)
CLEAN_CATALOG = os.path.join(DATA_DIR, "catalog_clean.csv")   # produced by run_00

for _d in (DATA_DIR, OUT_DIR, MODEL_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- catalog
START_YEAR = 1998
END_DATE = "2023-10-31"

# Study area (deg)
LAT_MIN, LAT_MAX = 25.0, 29.0
LON_MIN, LON_MAX = 54.0, 58.0

# Magnitude-type homogenisation.
# 'convert'    -> convert mb/Ms/ML to Mw with published regressions, keep all
# 'preferred'  -> keep only PREFERRED_MAG_TYPES (smallest catalog)
# run_00 ALWAYS reports the magnitude-type census, whatever this is set to,
MAG_HOMOGENISATION = "none"
PREFERRED_MAG_TYPES = ("mb", "MB")

# Binning of the frequency-magnitude distribution
DM = 0.1

# Completeness: if MC_OVERRIDE is None the value is estimated by run_01 and
# written to outputs/catalog_report.json; downstream scripts read it from there.
# Set MC_OVERRIDE to a float to bypass the automatic estimate.
MC_OVERRIDE = None
MAXC_CORRECTION = 0.2      # Woessner & Wiemer (2005)
GFT_TARGET_R = 90.0        # % goodness of fit (Wiemer & Wyss 2000)
MC_SEARCH = (2.0, 4.5)     # search range for Mc
MC_MIN_EVENTS = 200        # minimum events above Mc for a candidate to be valid
MC_RULE = "maxc"           # 'maxc' = MAXC + correction (standard, and what the
                           # accepted; all three estimates are always reported.

# --- REPORTING completeness, deliberately decoupled from the one above -----
# events, which the sequence models need, and any residual incompleteness hits
# every model and every baseline identically, so the benchmark stays fair.
# The b-value and the ETAS parameters are different: they are physical
# quantities compared against the literature, and they must be estimated where
# b has actually stabilised. MC_REPORTING_RULE controls only those.
MC_REPORTING_RULE = "b_stability"   # or "same" to reuse MC_RULE everywhere
MC_STABILITY_WINDOW = 0.5           # magnitude units, Wiemer & Wyss (2000)

# Gardner-Knopoff (1974) declustering (used for DESCRIPTIVE statistics only;
# the modelling catalog is intentionally NOT declustered)
GK_MODE = "gardner_knopoff"

# ---------------------------------------------------------------- features
FEATURES = ["LAT", "LON", "DEPTH", "MAG", "time_diff_log"]
N_FEAT = len(FEATURES)

# ---------------------------------------------------------------- experiment
WINDOW_SIZES = [20, 50]
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10            # test = remainder

# Parameter budgets. Every architecture is tuned to within PARAM_TOL of the
# budget, and the realised counts are measured and reported.
PARAM_BUDGETS = [57_000]
PARAM_TOL = 0.05           # +/- 5 %

ARCHITECTURES = ["LSTM", "GRU", "TCN", "Transformer"]

# Training settings identical to the original scripts, so that the numbers move
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))
EPOCHS = 50
BATCH = 16
PATIENCE = 8
LR = 1e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-7

# TCN dilation schedule (receptive field = 1 + 2*sum(d) = 63 for kernel 3)
TCN_KERNEL = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]

# ---------------------------------------------------------------- evaluation
THRESHOLDS = [4.0, 4.5]
PRIMARY_THRESHOLD = 4.0    # pre-specified; 4.5 is secondary/exploratory

# Resampling
MBB_BLOCK = 20             # moving-block bootstrap block length (events)
MBB_B = 4_000
RNG_SEED = 42
HOLM_FAMILY_SIZE = None    # computed at run time from the actual comparisons

# Information gain
IG_SIGMA_SOURCE = "validation_mle"   # sigma of the Gaussian predictive density

# Integrated Gradients
IG_N_STEPS = 50
IG_N_BASELINE = 100
IG_BASELINE_SEED = 42

# ---------------------------------------------------------------- ETAS
ETAS_P_FIX = None          # set to a float to fix Omori p
ETAS_RESTARTS = 40         # multi-start global optimisation
ETAS_SEED = 7

# ---------------------------------------------------------------- figures
DPI = 300
FONT_FAMILY = "Times New Roman"
FONT_FALLBACK = "DejaVu Serif"
FONT_SIZE = 12
LINE_WIDTH = 2.0

# ================================================================ MODE overrides
if MODE == "sensitivity":
    MAG_HOMOGENISATION = "convert"

elif MODE == "full":
    MAG_HOMOGENISATION = "convert"
    PARAM_BUDGETS = [8_000, 57_000]
    EPOCHS = 200
    BATCH = 32
    PATIENCE = 20

# ---------------------------------------------------------------- misc
QUICK = bool(int(os.environ.get("ZM_QUICK", "0")))
if QUICK:
    N_SEEDS = 2
    SEEDS = list(range(N_SEEDS))
    EPOCHS = 8
    MBB_B = 200
    ETAS_RESTARTS = 4
    PARAM_BUDGETS = PARAM_BUDGETS[-1:]


def summary() -> str:
    """One-line description of the active configuration, printed by every script."""
    return (
        "MODE={} | QUICK={} | magnitudes={} | budgets={} | windows={} | "
        "seeds={} | epochs={}".format(
            MODE, QUICK, MAG_HOMOGENISATION,
            ["{:,}".format(b) for b in PARAM_BUDGETS],
            WINDOW_SIZES, N_SEEDS, EPOCHS)
    )
