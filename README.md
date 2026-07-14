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
study. It exists so that every figure, table, and reported statistic in the synopsis and final
report can be reproduced from a clean clone.

The study asks whether the energy intensity of a commercial building can be predicted from the
parameters that are known at concept design, that is, before any mechanical system is specified
and before an energy model is built. It addresses five research questions spanning statistical
inference, supervised regression, model explanation, and classification.

---

## Data source

All data is drawn from a single authoritative public source. No Kaggle data is used.

**2018 Commercial Buildings Energy Consumption Survey (CBECS), Public Use Microdata**
U.S. Energy Information Administration (EIA)

- Landing page: https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata
- Data file: https://www.eia.gov/consumption/commercial/data/2018/xls/cbecs2018_final_public.csv
- Codebook: https://www.eia.gov/consumption/commercial/data/2018/xls/2018microdata_codebook.xlsx

**Release history.** The microdata was first released in November 2021 and revised in December
2022. The December 2022 revision appended the consumption and expenditures variables to the
file. The copy committed here is that final revision, which is why building characteristics,
energy consumption, and end-use variables all reside in one file and no joins are required.

**Licence.** CBECS is a U.S. federal government public-use data product and is in the public
domain. It is redistributed here for reproducibility.

---

## Dataset at a glance

| Property | Value |
|---|---|
| Records (buildings) in the public file | 6,436 |
| Columns | 1,249 |
| Records excluded (no major-fuel consumption, MFBTU missing) | 79 |
| **Analysis sample** | **6,357** |
| Population represented (weighted) | 5,918 thousand commercial buildings |
| Floorspace represented (weighted) | 96,527 million sq ft |
| Sampling design | Stratified probability sample |
| Weight variable | `FINALWT` |
| Replicate weights | 150 JK2 jackknife replicate weights |

**Response variable.** Site energy use intensity (EUI), computed as annual major-fuel
consumption divided by gross floor area: `EUI = MFBTU / SQFT`, expressed in kBtu per square foot
per year.

---

## Survey design

CBECS is not a simple random sample. Naive analysis will produce biased standard errors. The
analysis therefore uses:

- `FINALWT` for all point estimates.
- The 150 JK2 jackknife replicate weights for variance estimation, following the worked R
  examples in EIA's *User's Guide to the 2018 CBECS Public Use Microdata File*.
- Survey-weighted estimation for the inferential research questions (RQ1, RQ2, RQ4), because
  these are inferences about the population of U.S. commercial buildings.
- Unweighted estimation for the predictive research questions (RQ3, RQ5), because out-of-sample
  predictive accuracy on a held-out partition is a property of the model, not a population
  parameter.

**Weight validation.** Summing `FINALWT` over the microdata reproduces the published Table B1
estimates to within 0.1 percent (5,918 thousand buildings; 96,527 against 96,423 million sq ft),
the residual difference being attributable to the disclosure masking applied to the public file.
This check is run in the data preparation step and should pass on any clean clone.

---

## Coding conventions that require handling

Three conventions in the microdata will silently corrupt an analysis if treated naively.

1. **`NFLOOR` is not a plain count.** Values 1 through 9 are literal floor counts, but 994 and
   995 are category codes denoting banded ranges for taller buildings. These are recoded to an
   ordinal scale rather than treated as numeric magnitudes.
2. **`GLSSPC` is a band, not a percentage.** It is released as a six-level ordinal describing the
   percentage of exterior wall that is glass, and enters the models as an ordered categorical
   predictor.
3. **`MFBTU` is missing for 79 records.** These are buildings that consumed no major fuel in the
   reference year. Site EUI is undefined for them, so they are excluded.

Imputation flags (the `Z`-prefixed variables, for example `ZSQFT` for `SQFT`) accompany every
imputed field. Imputation rates for the modelling variables are reported in the data preparation
step.

---

## Repository structure

```
.
├── README.md
├── data/
│   ├── raw/
│   │   ├── cbecs2018_final_public.csv      # as downloaded from EIA
│   │   └── 2018microdata_codebook.xlsx     # variable and response codebook
│   └── processed/                          # derived analysis extract (generated)
├── src/
│   ├── 01_prepare.py                       # cleaning, EUI construction, survey design object
│   ├── 02_rq1_regression.py                # survey-weighted multiple regression
│   ├── 03_rq2_vintage.py                   # construction era band comparison
│   ├── 04_rq3_model_comparison.py          # OLS vs. random forest vs. gradient boosting
│   ├── 05_rq4_shap.py                      # SHAP attribution and confirmatory test
│   └── 06_rq5_classification.py            # three-class energy band classification
├── outputs/
│   ├── figures/
│   └── tables/
└── requirements.txt
```

Code is added incrementally as the analysis proceeds. The data and codebook are complete as of
the synopsis submission.

---

## Reproducing the analysis

```bash
git clone <REPOSITORY_URL>
cd cbecs-energy-capstone
pip install -r requirements.txt
python src/01_prepare.py     # writes data/processed/, runs the Table B1 weight validation
python src/02_rq1_regression.py
# ... and so on
```

---

## Research questions

| RQ | Question | Primary technique |
|---|---|---|
| RQ1 | Which early-design parameters are significantly associated with site EUI, and how much variance do they explain? | Survey-weighted multiple linear regression |
| RQ2 | Does mean EUI differ across construction era bands? | Design-based Wald test; ANOVA as unweighted check |
| RQ3 | Do ensemble models outperform linear regression at predicting EUI? | Repeated 5-fold CV; corrected resampled *t*-test (Nadeau & Bengio, 2003) |
| RQ4 | Which design-controllable parameter is the leading driver, and is its partial association significant? | SHAP for identification; partial correlation for confirmation |
| RQ5 | Can a design be reliably assigned to a low, neutral, or high energy band relative to its activity peers? | Multinomial logistic regression vs. gradient boosting; macro-F1, confusion matrix |

**Sample size.** The binding constraint is RQ5, which requires N = 5,775 to estimate per-class
recall to ±5 percentage points at 95 percent confidence across three balanced classes under an
80/20 split. The analysis sample of 6,357 satisfies this. Full derivations for all five research
questions appear in the synopsis.

---

## Known limitations

- **Geography is masked to nine census divisions.** No state identifier is available, so no claim
  finer than census division can be made about local climate or code regime.
- **The model describes the U.S. commercial building stock.** Its coefficients encode American
  construction typologies, energy codes, and operating conventions. It is not transferable to
  other national contexts without recalibration.
- **CBECS records what buildings are, not what they were designed to be.** A model fitted to the
  existing stock learns the performance of buildings built under prevailing practice, and will
  regress a genuinely high-performance design toward the stock mean.
- **EUI is an intensity, so floor area is normalised out by construction.** A materially lower R²
  than studies modelling total consumption is expected and is a consequence of the framing rather
  than a deficiency of the model.
- **Subgroup reporting follows EIA's own publication standard**: estimates are suppressed where
  fewer than 20 buildings underlie a calculation or the relative standard error exceeds 50
  percent.

---

## Reference

U.S. Energy Information Administration. (2022). *2018 Commercial Buildings Energy Consumption
Survey: Public use microdata* [Data set]. https://www.eia.gov/consumption/commercial/data/2018/index.php?view=microdata
