# Concept-Stage Energy Screening Tool

A genuinely connected implementation of the QM640 capstone study's findings —
the web app reads its coefficients from `model.json`, which is exported
directly from the study's own fitted models. Nothing in `index.html` is a
hand-typed approximation.

## Files

- **`index.html`** — the interface. Fetches `model.json` at load time and runs
  the real regression and classifier in the browser.
- **`model.json`** — the exported model: full concept-stage regression
  coefficients (44 predictors), the RQ4 classifier's scaler and logistic
  regression matrix, the real per-activity tercile cut points, and per-activity
  typical input values, all computed from the actual cleaned CBECS sample.
- **`export_model.py`** — regenerates `model.json` from scratch. Downloads the
  raw CBECS file, reproduces the cleaning and feature-engineering pipeline
  exactly, refits both models, and writes the JSON. No notebook state required.

## How the two models differ from the ones in the report

The report's RQ1 regression and RQ4 classifier are not identical specifications.
RQ1 includes installed heating/cooling system type (`MAINHT`/`MAINCL`); RQ4's
classifier deliberately excludes those two variables, because a concept-stage
tool can't ask about a system that hasn't been chosen yet.

This app needs both a continuous EUI estimate and a band classification, both
usable before system selection, so `export_model.py` fits a **new regression**
using RQ1's exact method (survey-weighted least squares on ln(EUI), weighted
by FINALWT) restricted to the same 44 concept-stage predictors RQ4 already
uses. This is a legitimate sub-model of the reported specification, not an
approximation of it — validated against the notebooks by reproducing RQ1's
full-model R² to four decimal places (.5607) before restricting the predictor
set. It explains less variance than the full RQ1 model (44.5% vs. 56.1%)
because it's missing what system type would have explained — which is itself
a real, disclosed finding, not a limitation of this tool alone.

The classifier is reproduced exactly: same train/test split (seed 42, stratified
on activity), same tercile cut-point algorithm with sparse-activity pooling,
same StandardScaler + multinomial logistic regression. Reproduced performance
(macro F1 = .484, κ = .236) matches the study's reported .485 / .240 to within
normal cross-environment floating-point variation.

## Hosting

Static files, no backend. For GitHub Pages: commit all three files to a folder
in the repo, enable Pages in Settings, done.

**Important:** `index.html` uses `fetch()` to load `model.json`, which browsers
block on `file://` (double-clicking the file). It needs to be served over
HTTP — GitHub Pages does this automatically. To test locally:

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

## Regenerating after retraining

If the cleaning, feature engineering, or either model's specification changes,
rerun the export and replace `model.json` — no changes to `index.html` needed:

```bash
pip install pandas numpy scipy statsmodels scikit-learn --break-system-packages
python3 export_model.py
```
