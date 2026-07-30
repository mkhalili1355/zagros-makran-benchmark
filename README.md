# Zagros-Makran next-event magnitude benchmark

Analysis pipeline for a parameter-controlled comparison of four deep sequence
architectures (LSTM, GRU, TCN, Transformer) with statistical and
catalog-derived baselines for next-event magnitude prediction in the
Zagros-Makran transition zone (25-29 N, 54-58 E, 1998-2023).

The analysis is organised as six sequential steps. Every step writes a JSON
report, and the final step consolidates all reported quantities into a single
file, so that the text, tables and figures derive from one source.

## Installation

```
pip install -r requirements.txt
```

The reported results were produced under Python 3.9.25 on CPU with
TensorFlow 2.20 (Keras 3), NumPy 1.26.4, pandas 2.3.1, SciPy 1.13.1,
scikit-learn 1.6.1 and Matplotlib 3.9.4.

## Input data

The input is an ISC Bulletin export for the study region, read from
`data/Final_.csv`. The file name is configurable through `RAW_FILENAME` in
`zm/config.py`. Terms of use and citation requirements for the catalog are
those of the International Seismological Centre.

## Execution

```
python run_00_clean.py      catalog construction and magnitude-type census
python run_01_catalog.py    completeness, b-value, declustering, splits
python run_02_train.py      4 architectures x 2 window lengths x 10 seeds
python run_03_eval.py       baselines, ETAS, MAE, AUC, Molchan, inference
python run_04_ig.py         Integrated Gradients attribution
python run_05_figures.py    figures computed from the data
python run_06_numbers.py    consolidated quantities and tables
```

The same sequence is available in `Zagros_Pipeline.ipynb`.

Step 2 skips any architecture, window length and seed for which a weights file
already exists, so an interrupted run resumes where it stopped.

Wall-clock times measured on a four-core CPU: 265 minutes for the 80 training
runs in step 2, approximately 4 hours for step 3, where the cost is dominated by
the moving-block bootstrap and the exact circular-shift randomisation, and a few
minutes for each remaining step.

The environment variable `ZM_QUICK=1` reduces the configuration to two seeds and
eight epochs. It serves as an installation test and does not reproduce the
reported results.

## Configuration

All constants are defined in `zm/config.py`. The reported results correspond to
the default configuration, `ZM_MODE=minimal`: magnitudes as reported by the
contributing agencies, a single parameter budget of approximately 57,000 with a
5 % tolerance, window lengths of 20 and 50 events, four architectures, 50 epochs,
batch size 16, patience 8, ten seeds, chronological split 80/10/10, moving block
length of 20 windows and 4,000 bootstrap resamples.

Two further configurations are defined. `ZM_MODE=sensitivity` repeats the
analysis with magnitudes converted to Mw following Scordilis (2006).
`ZM_MODE=full` applies the same conversion, adds a second parameter budget of
8,000 and extends training to 200 epochs. Both write to the tree given by
`ZM_BASE`, which keeps them separate from the reported run.

## Outputs

```
outputs/cleaning_report.json        event counts at each filtering stage, magnitude-type census
outputs/catalog_report.json         Mc by four methods, b-values, splits, class balance
outputs/training_report.json        realised parameter counts, epochs, losses, runtimes per run
outputs/predictions.npz             cached validation and test predictions
outputs/results.json                baselines, ETAS, MAE, AUC, Molchan, bootstrap intervals
outputs/ig_report.json              attribution profiles and memory depths
outputs/figures/                    figures in png and pdf
outputs/figures/figure_values.json  every plotted value
outputs/NUMBERS.md                  consolidated quantities
outputs/Table*.csv                  tables in the reported form
```

The figures generated here are those computed from the data: the
frequency-magnitude distribution with its b(Mc) stability panel, the training and
validation loss curves, the Molchan and ROC diagrams at the pre-specified
threshold, the baseline comparison at the secondary threshold, and the
Integrated-Gradients temporal profiles. The seismotectonic map and the
architecture schematic are cartographic and illustrative products and lie
outside the scope of this pipeline.

## Method summary

The chronological split is applied to events before windows are formed, so no
window crosses a partition boundary. The feature scaler and the depth imputation
are estimated on the training partition alone. The regression target is the raw
magnitude in physical units. Architectures are tuned to a common parameter
budget and the realised counts are recorded rather than assumed. Model selection
uses validation loss; the test partition enters no selection decision.

Test windows overlap, so uncertainty is quantified throughout with a
moving-block bootstrap, and the significance of ranking statistics is assessed
with an exact circular-shift randomisation that preserves temporal structure. An
independent-permutation p-value is computed in parallel and retained only as a
measure of the inflation such a test produces on a temporally correlated
catalog; it enters no inference. Multiplicity is controlled by Holm-Bonferroni
over a declared family of eight tests, corresponding to four architectures at
two window lengths.

Completeness is estimated by maximum curvature with correction, the
goodness-of-fit test at the 90 % and 95 % residual levels, the
entire-magnitude-range method, and a b-value stability criterion. The b-value
follows the Aki maximum-likelihood estimator with Shi and Bolt uncertainty.
Declustering by the Gardner-Knopoff windows is computed for descriptive purposes
only; the modelling catalog is not declustered.

The ETAS model is fitted by maximum likelihood on training events above the
completeness threshold, with the background term contributing mu*T to the
integral and the Omori integral evaluated in closed form separately for p = 1 and
p != 1. Its magnitude forecast is the Gutenberg-Richter expectation, which is
constant across events, so it functions as a negative control rather than as a
competing predictor. The Gutenberg-Richter reference b-value entering the
information-gain calculation is estimated on training events only.

Magnitudes are analysed as reported, so the catalog combines mb and ML
determinations. The magnitude-type census is written by step 0 under every
configuration, and conversion or single-scale restriction is available through
`MAG_HOMOGENISATION`.

## Layout

```
run_00_clean.py        catalog construction
run_01_catalog.py      completeness, b-value, declustering, splits
run_02_train.py        training
run_03_eval.py         baselines, ETAS, metrics, resampling inference
run_04_ig.py           Integrated Gradients attribution
run_05_figures.py      figures
run_06_numbers.py      consolidated quantities and tables
zm/config.py           all constants
zm/catalog.py          loading, homogenisation, declustering, windowing, splitting
zm/completeness.py     MAXC, GFT, EMR, Aki b-value, b(Mc) stability
zm/models.py           architectures built to a parameter budget
zm/baselines.py        catalog-derived ranking baselines
zm/etas.py             temporal ETAS log-likelihood and fitting
zm/stats.py            moving-block bootstrap, circular-shift test, Holm, information gain
Zagros_Pipeline.ipynb  notebook wrapper for the six steps
```

Random seeds are fixed and recorded, and software versions are pinned in
`requirements.txt`.

## License

The source code is released under the MIT License (`LICENSE`). The earthquake
catalog is subject to the terms of the ISC Bulletin. Software citation metadata
is given in `CITATION.cff`.
