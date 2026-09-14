# Predicting Building Energy Performance from Early Design Parameters

**A Statistical and Machine Learning Approach Using the 2018 CBECS Microdata**

Data Analytics Capstone (QM 640) — Term 3

| | |
|---|---|
| **Student** | Sharath Mohan |
| **Student ID** | 0293212 |
| **Course** | QM 640: Data Analytics Capstone |
| **Term** | Term 3 |
| **Mentor** | Mr. Rishab Pandey |

---

## Purpose of this repository

This repository provides open access to the complete data and code underlying the capstone
study. It exists so that every figure, table, and reported statistic in the synopsis, interim
report, and final report can be reproduced from a clean clone.

The study asks whether the energy intensity of a commercial building can be predicted from the
parameters that are known at concept design, that is, before any mechanical system is specified
and before an energy model is built. It addresses four research questions spanning statistical
inference, supervised regression, model explanation, and classification.

---

## Data source

All data is drawn from a single authoritative public source. No Kaggle data is used.

**2018 Commercial Buildings Energy Consumption Survey (CBECS), Public Use Microdata**
U.S. Energy Information Administration (EIA)

- Landing page: https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata
- Data file (CSV): https://www.eia.gov/consumption/commercial/data/2018/xls/cbecs2018_final_public.csv
- Codebook (XLSX): https://www.eia.gov/consumption/commercial/data/2018/xls/2018microdata_codebook.xlsx
- User's Guide (PDF): https://www.eia.gov/consumption/commercial/data/2018/pdf/Users%20Guide%20to%20the%202018%20CBECS%20Public%20Use%20Microdata%20File.pdf
- Building Characteristics Flipbook (PDF): https://www.eia.gov/consumption/commercial/data/2018/pdf/CBECS_2018_Building_Characteristics_Flipbook.pdf

The CSV file really does live in the `/xls/` subdirectory on EIA's server — this is not a typo.

**Release history.** The microdata was first released in November 2021 and revised in December
2022. The December 2022 revision appended the consumption and expenditures variables to the
file. The copy committed here is that final revision, which is why building characteristics,
energy consumption, and end-use variables all reside in one file and no joins are required.

**Licence.** CBECS is a U.S. federal government public-use data product. Individual building
identities are confidential and protected by law, so the public-use file has already been
masked by EIA using standard disclosure-avoidance techniques. In that masked form the file is
freely redistributable and is included here for reproducibility.

---

## Dataset at a glance

| Property | Value |
|---|---|
| Records (buildings) in the public file | 6,436 |
| Columns | 1,249 |
| Records excluded (no major-fuel consumption, `MFBTU` missing) | 79 |
| **Analysis sample** | **6,357** |
| Sampling design | Stratified probability sample |
| Weight variable | `FINALWT` |
| Replicate weights | 151 JK2 jackknife replicate weights |

**Response variable.** Site energy use intensity (EUI), computed as annual major-fuel
consumption divided by gross floor area: `EUI = MFBTU / SQFT`, expressed in kBtu per square foot
per year.

---

## Survey design

CBECS is not a simple random sample. Naive analysis will produce biased standard errors. The
analysis therefore uses:

- `FINALWT` for all point estimates.
- The 151 JK2 jackknife replicate weights for variance estimation, following the worked
  examples in EIA's *User's Guide to the 2018 CBECS Public Use Microdata File*.
- Survey-weighted estimation for the inferential research questions (RQ1 and RQ2), because
  these are inferences about the population of U.S. commercial buildings.
- Unweighted estimation for the predictive research questions (RQ3 and RQ4), because
  out-of-sample predictive accuracy on a held-out partition is a property of the model, not a
  population parameter.

**Weight validation.** Summing `FINALWT` over the raw microdata reproduces the published
Table B1 estimates to within a fraction of a percent, both for building count and for total
floorspace. The residual difference is attributable to the disclosure masking applied to the
public file. After the 79 records with missing `MFBTU` are excluded, the analysis sample of
6,357 represents roughly 5.6 million buildings by `FINALWT`. This check is run in the data
preparation step and should pass on any clean clone; the exact figures are printed in the
notebook output.

**Design effect.** The design effect for mean log EUI, computed against the JK2 replicate
weights, is roughly 6, so the effective sample size for inferential tests is about a sixth of
the raw building count, not the full 6,357. The exact value is reported in the EDA notebook.

---

## Coding conventions that require handling

Four conventions in the microdata will silently corrupt an analysis if treated naively.

1. **`NFLOOR` is not a plain count.** Values 1 through 9 are literal floor counts, but 994 and
   995 are category codes denoting banded ranges for taller buildings. These are recoded to an
   ordinal scale rather than treated as numeric magnitudes.
2. **`GLSSPC` is a band, not a percentage.** It is released as a six-level ordinal describing
   the percentage of exterior wall that is glass, and enters the models as an ordered
   categorical predictor.
3. **`FLCEILHT` carries a top-code.** The value 995 is not 995 feet; it is EIA's convention
   for "more than 50 feet." Fourteen records in the analysis sample carry this code (0.22
   percent). Left as a continuous value, these would exert heavy leverage on the regression,
   so they are recoded before RQ1 is fit.
4. **`MFBTU` is missing for 79 records.** These are buildings that consumed no major fuel in
   the reference year. Site EUI is undefined for them, so they are excluded from the analysis
   sample.

`MAINHT` and `MAINCL` also have systematic missingness (429 and 456 records respectively), but
this reflects buildings without a main heating or cooling system rather than a measurement gap,
and is recoded to an explicit "None reported" category rather than dropped or imputed. Both
this and the `FLCEILHT` top-code fall within the single-source data quality taxonomy of Rahm
and Do (2000), and are handled by recoding rather than record deletion so that the sample size
and the JK2 replicate structure stay intact.

Imputation flags (the `Z`-prefixed variables, for example `ZSQFT` for `SQFT`) accompany every
imputed field. Imputation rates for the modelling variables are reported in the data
preparation step.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── cbecs2018_final_public.csv                       # as downloaded from EIA
│       ├── 2018microdata_codebook.xlsx                      # variable and response codebook
│       └── CBECS_2018_Building_Characteristics_Flipbook.pdf # EIA plain-language reference
├── notebooks/
│   ├── cbecs_eda_pipeline.ipynb                             # cleaning, EUI, survey design, full EDA
│   ├── rq1_survey_weighted_regression.ipynb                # RQ1 regression + SHAP driver ranking
│   ├── rq2_vintage_cohort_comparison.ipynb                 # RQ2 design-based Wald test on vintage
│   ├── rq3_model_comparison.ipynb                          # RQ3 OLS vs. RF vs. XGBoost
│   └── rq4_energy_band_classification.ipynb                # RQ4 energy-band classifier
├── output/
│   ├── figures/                                            # generated charts (PNG)
│   └── tables/                                             # generated tables (CSV)
├── app/                                                    # concept-stage screening web tool
│   ├── index.html                                          # the interface
│   ├── model.json                                          # fitted coefficients, exported
│   ├── export_model.py                                     # regenerates model.json from raw data
│   └── README.md                                           # app-specific notes
└── src/                                                    # reserved for shared modules
```

Every figure and table in `output/` is generated by the notebooks, not committed by hand. Each
notebook writes its own outputs and, when run in Colab, commits and pushes them back to the
repository.

---

## Reproducing the analysis

The notebooks read the raw CBECS file directly from this repository, so they can be run in
Google Colab with no local setup, or locally after a clone.

```bash
git clone https://github.com/Shaman2320/QM640_Capstone_BuildingEnergyPerfromance.git
cd QM640_Capstone_BuildingEnergyPerfromance
pip install -r requirements.txt
```

Run the notebooks **in this order**, because each research-question notebook reads the
engineered predictor matrix that the EDA notebook writes to `output/tables/`:

1. `notebooks/cbecs_eda_pipeline.ipynb` — cleaning, survey-weight validation, exploratory
   analysis, and the shared engineered predictor matrix
2. `notebooks/rq1_survey_weighted_regression.ipynb`
3. `notebooks/rq2_vintage_cohort_comparison.ipynb`
4. `notebooks/rq3_model_comparison.ipynb`
5. `notebooks/rq4_energy_band_classification.ipynb`

Each notebook is self-contained and re-executes top to bottom. Random seeds are fixed, so a
clean run reproduces the reported results.

---

## The screening tool (`app/`)

https://shaman2320.github.io/QM640_Capstone_BuildingEnergyPerfromance/app/

The `app/` folder contains a small web application that puts the study's findings into a usable
form: enter the concept-stage parameters of a proposed building and it returns an
activity-relative energy band (low, medium, or high) with a plain-language reliability caveat.

It runs the study's actual fitted models client-side. The regression coefficients and the
classifier are exported to `app/model.json` by `app/export_model.py`, which reproduces the
cleaning pipeline from the raw file and refits both models, so the tool never drifts from the
analysis. To refresh it after any change to the pipeline, rerun `export_model.py` and replace
`model.json`.

Because the page loads `model.json` with a browser `fetch`, it must be served over HTTP rather
than opened directly from disk. Serve it locally with `python3 -m http.server` from inside
`app/`, or host the folder on GitHub Pages. See `app/README.md` for details.

---

## Research questions

| RQ | Question | Primary technique |
|---|---|---|
| RQ1 | Which early-design parameters are associated with site EUI, and which one drives it most? | Survey-weighted multiple regression; SHAP attribution on the RQ3 best model for ranking; partial-correlation confirmation of the top driver |
| RQ2 | Does mean EUI differ across construction era bands? | Design-based Wald test; two-way ANOVA with activity as a robustness check |
| RQ3 | Do ensemble models outperform linear regression at predicting EUI? | OLS vs. random forest vs. XGBoost on 20 percent holdout; corrected resampled *t*-test (Nadeau & Bengio, 2003) on repeated 5-fold cross-validation |
| RQ4 | Can a building be reliably sorted into a low, medium, or high energy band before it is built? | Activity-specific EUI terciles; multinomial logistic regression vs. gradient boosting; macro-F1, Cohen's kappa, McNemar's test against a stratified baseline |

**Sample size.** The binding constraint is RQ4, which requires N = 5,775 to estimate per-class
recall to ±5 percentage points at 95 percent confidence across three balanced classes under an
80/20 split. The analysis sample of 6,357 satisfies this with roughly a 10 percent margin. Full
derivations for all four research questions appear in the synopsis and interim report.

---

## Known limitations

- **Geography is masked to nine census divisions.** No state identifier is available, so no
  claim finer than census division can be made about local climate or code regime.
- **The model describes the U.S. commercial building stock.** Its coefficients encode American
  construction typologies, energy codes, and operating conventions. It is not transferable to
  other national contexts without recalibration; a transfer-learning approach is the intended
  bridge to Indian projects.
- **CBECS captures no orientation, aspect ratio, compactness, façade composition, or shading.**
  These are variables the architectural literature suggests matter, so their absence is a
  ceiling on R² stated in advance rather than a modelling defect.
- **CBECS records what buildings are, not what they were designed to be.** A model fitted to
  the existing stock learns the performance of buildings built under prevailing practice, and
  will regress a genuinely high-performance design toward the stock mean.
- **EUI is an intensity, so floor area is normalised out by construction.** A materially lower
  R² than studies modelling total consumption is expected and is a consequence of the framing
  rather than a deficiency of the model.
- **Subgroup reporting follows EIA's own publication standard**: estimates are suppressed
  where fewer than 20 buildings underlie a calculation or the relative standard error exceeds
  50 percent.

---

## References

Rahm, E., & Do, H. H. (2000). Data cleaning: Problems and current approaches. *IEEE Data
Engineering Bulletin, 23*(4), 3–13.

U.S. Energy Information Administration. (2022). *2018 Commercial Buildings Energy Consumption
Survey: Public use microdata* [Data set]. https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata

U.S. Energy Information Administration. (2023). *User's guide to the 2018 CBECS public use
microdata file.* U.S. Department of Energy. https://www.eia.gov/consumption/commercial/data/2018/pdf/Users%20Guide%20to%20the%202018%20CBECS%20Public%20Use%20Microdata%20File.pdf
