"""
export_model.py — regenerates model.json for the concept-stage energy
screening tool from scratch.

Run this any time the underlying analysis changes (a new CBECS release, a
methodology change in the cleaning or feature-engineering steps, a different
classifier). It reproduces the exact cleaning and matrix-assembly logic from
cbecs_eda_pipeline.ipynb, refits the concept-stage regression and the RQ4
classifier, and writes a single model.json that index.html reads at runtime.

No notebook state is required -- this is self-contained and reads the raw
CBECS file directly from the project's GitHub repository.

Usage:
    pip install pandas numpy scipy statsmodels scikit-learn --break-system-packages
    python3 export_model.py
    # writes ./model.json -- copy or commit it alongside index.html
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

RAW_CSV_URL = ("https://raw.githubusercontent.com/Shaman2320/"
              "QM640_Capstone_BuildingEnergyPerfromance/main/data/raw/cbecs2018_final_public.csv")
SEED, TEST_SIZE, MIN_ACTIVITY_N = 42, 0.20, 60

ID_VARS = ["PUBID"]
CONTINUOUS_VARS = ["SQFT", "FLCEILHT", "WKHRS", "NWKER", "HDD65", "CDD65"]
ORDINAL_VARS = ["NFLOOR", "YRCONC", "GLSSPC"]
NOMINAL_VARS = ["PBA", "WLCNS", "RFCNS", "MAINHT", "MAINCL", "OPEN24"]
WEIGHT_VAR, FUEL_VAR = "FINALWT", "MFBTU"

PBA_LABELS = {1: "Vacant", 2: "Office", 4: "Laboratory", 5: "Nonrefrigerated warehouse",
              6: "Food sales", 7: "Public order and safety", 8: "Outpatient health care",
              11: "Refrigerated warehouse", 12: "Religious worship", 13: "Public assembly",
              14: "Education", 15: "Food service", 16: "Inpatient health care",
              17: "Nursing", 18: "Lodging", 23: "Strip shopping mall",
              24: "Enclosed mall", 25: "Retail (other than mall)", 26: "Service", 91: "Other"}
WLCNS_LABELS = {1: "Masonry (brick, stone, concrete block)", 2: "Curtain wall / metal panel",
                3: "Concrete", 4: "Wood", 5: "Concrete panel / pre-cast",
                6: "Sheet metal panel", 7: "Shingle (siding)", 8: "Other"}
RFCNS_LABELS = {1: "Built-up", 2: "Slate or tile shingles", 3: "Wood shingles/shakes",
                4: "Asphalt/fiberglass shingles", 5: "Metal surfacing",
                6: "Synthetic or rubber", 7: "Concrete", 8: "Other"}


def clean_and_engineer(raw):
    df = raw[raw[FUEL_VAR].notna()].copy()
    df["MAINHT"] = df["MAINHT"].fillna(0)
    df["MAINCL"] = df["MAINCL"].fillna(0)
    for col in ORDINAL_VARS:
        df[col] = pd.Categorical(df[col], categories=sorted(df[col].dropna().unique()), ordered=True)
    for col in NOMINAL_VARS:
        df[col] = df[col].astype("category")
    df["EUI"] = df[FUEL_VAR] / df["SQFT"]
    df["log_EUI"] = np.log(df["EUI"])

    eng = pd.DataFrame(index=df.index)
    eng["ln_SQFT"] = np.log(df["SQFT"])
    eng["FLCEILHT_capped"] = df["FLCEILHT"].clip(upper=50)
    eng["ceiling_over_50ft"] = (df["FLCEILHT"] == 995).astype(int)
    eng["worker_density_per_1000sqft"] = df["NWKER"] / df["SQFT"] * 1000
    eng["WKHRS"] = df["WKHRS"]
    eng["HDD65"] = df["HDD65"]
    eng["CDD65"] = df["CDD65"]

    ordinal_X = df[ORDINAL_VARS].astype("category").apply(lambda s: s.cat.codes)
    nominal_X = pd.get_dummies(df[NOMINAL_VARS].astype("category"), drop_first=True, dtype=int)
    X_rq1 = pd.concat([eng, ordinal_X, nominal_X], axis=1)

    ordinal_ranks = {col: sorted(float(x) for x in df[col].cat.categories) for col in ORDINAL_VARS}
    return df, X_rq1, ordinal_ranks


def fit_concept_regression(X_rq1, y, w_full, concept_cols):
    X_design = sm.add_constant(X_rq1[concept_cols].astype(float))
    fit = sm.WLS(y, X_design, weights=w_full).fit()
    coefs = dict(zip(X_design.columns, fit.params.values))
    return coefs, {
        "n": int(len(y)), "n_predictors": len(concept_cols),
        "r2_weighted": float(fit.rsquared), "adj_r2_weighted": float(fit.rsquared_adj),
        "residual_sd_log": float(np.std(fit.resid)),
    }


def fit_rq4_classifier(X_rq1, df, concept_cols):
    eui = np.exp(df["log_EUI"].to_numpy(float))
    X_concept = X_rq1[concept_cols].to_numpy(float)
    pba_code = df["PBA"].astype(int).to_numpy()

    strata = pba_code.copy()
    counts = pd.Series(strata).value_counts()
    strata = np.where(np.isin(strata, counts[counts < 10].index), -1, strata)
    idx_train, idx_test = train_test_split(
        np.arange(len(X_rq1)), test_size=TEST_SIZE, random_state=SEED, stratify=strata)

    train_counts = pd.Series(pba_code[idx_train]).value_counts()
    pooled = set(train_counts[train_counts < MIN_ACTIVITY_N].index)
    group = np.where(np.isin(pba_code, list(pooled)), -1, pba_code)

    cuts = {}
    for g in np.unique(group):
        vals = eui[idx_train][group[idx_train] == g]
        lo, hi = np.quantile(vals, [1/3, 2/3])
        cuts[int(g)] = (float(lo), float(hi))

    band = np.zeros(len(X_rq1), dtype=int)
    for g, (lo, hi) in cuts.items():
        m = (group == g)
        band[m] = np.where(eui[m] <= lo, 0, np.where(eui[m] <= hi, 1, 2))

    Xc_train, y_train = X_concept[idx_train], band[idx_train]
    Xc_test, y_test = X_concept[idx_test], band[idx_test]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, random_state=SEED))
    clf.fit(Xc_train, y_train)
    pred = clf.predict(Xc_test)
    perf = {"accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "kappa": float(cohen_kappa_score(y_test, pred))}

    scaler, logit = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    return {
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "logit_coef": logit.coef_.tolist(), "logit_intercept": logit.intercept_.tolist(),
        "cutpoints": {str(k): {"low": v[0], "high": v[1]} for k, v in cuts.items()},
        "pooled_activity_codes": sorted(int(g) for g in pooled),
        "reproduced_performance": perf,
    }


def build_per_activity_profiles(df):
    per_activity = {}
    for code, label in PBA_LABELS.items():
        sub = df[df["PBA"].astype(int) == int(code)]
        if len(sub) == 0:
            continue
        per_activity[str(code)] = {
            "label": label, "n": int(len(sub)),
            "sqft_median": float(sub["SQFT"].median()),
            "floors_median": float(pd.to_numeric(sub["NFLOOR"], errors="coerce").clip(upper=9).median()),
            "ceiling_median": float(sub["FLCEILHT"].clip(upper=50).median()),
            "glazing_mode": float(sub["GLSSPC"].astype(float).mode().iloc[0]),
            "hours_median": float(sub["WKHRS"].median()),
            "workers_median": float(sub["NWKER"].median()),
            "wall_mode": int(sub["WLCNS"].astype(int).mode().iloc[0]),
            "roof_mode": int(sub["RFCNS"].astype(int).mode().iloc[0]),
            "hdd_median": float(sub["HDD65"].median()),
            "cdd_median": float(sub["CDD65"].median()),
            "eui_geomean": float(np.exp(np.log(sub["EUI"]).mean())),
        }
    return per_activity


def main():
    print(f"Downloading raw CBECS data from {RAW_CSV_URL} ...")
    raw = pd.read_csv(RAW_CSV_URL)
    df, X_rq1, ordinal_ranks = clean_and_engineer(raw)
    y = df["log_EUI"]
    w_full = df[WEIGHT_VAR].to_numpy(float)
    concept_cols = [c for c in X_rq1.columns if not (c.startswith("MAINHT_") or c.startswith("MAINCL_"))]
    print(f"Cleaned sample: {len(df):,} buildings; concept-stage predictors: {len(concept_cols)}")

    regression_coefficients, reg_summary = fit_concept_regression(X_rq1, y, w_full, concept_cols)
    print(f"Concept-stage regression: weighted R2 = {reg_summary['r2_weighted']:.4f}")

    classifier_export = fit_rq4_classifier(X_rq1, df, concept_cols)
    print(f"RQ4 classifier reproduction: macro F1 = "
          f"{classifier_export['reproduced_performance']['macro_f1']:.4f}, "
          f"kappa = {classifier_export['reproduced_performance']['kappa']:.4f}")

    per_activity = build_per_activity_profiles(df)

    model = {
        "meta": {
            "source": "QM640 capstone: Predicting Commercial Building Energy Performance "
                      "from Early Design Parameters (2018 CBECS)",
            "n": reg_summary["n"],
            "regression_r2_weighted": reg_summary["r2_weighted"],
            "regression_residual_sd_log": reg_summary["residual_sd_log"],
            "classifier_macro_f1": classifier_export["reproduced_performance"]["macro_f1"],
            "classifier_kappa": classifier_export["reproduced_performance"]["kappa"],
            "note": "Both models are refit here restricted to concept-stage predictors "
                   "(installed heating/cooling system type excluded).",
        },
        "concept_cols": concept_cols,
        "regression_coefficients": regression_coefficients,
        "ordinal_ranks": ordinal_ranks,
        "scaler_mean": classifier_export["scaler_mean"],
        "scaler_scale": classifier_export["scaler_scale"],
        "logit_coef": classifier_export["logit_coef"],
        "logit_intercept": classifier_export["logit_intercept"],
        "cutpoints": classifier_export["cutpoints"],
        "pooled_activity_codes": classifier_export["pooled_activity_codes"],
        "pba_labels": {str(k): v for k, v in PBA_LABELS.items()},
        "wlcns_labels": {str(k): v for k, v in WLCNS_LABELS.items()},
        "rfcns_labels": {str(k): v for k, v in RFCNS_LABELS.items()},
        "per_activity": per_activity,
    }

    with open("model.json", "w") as f:
        json.dump(model, f, indent=1)
    print("\nWrote model.json")


if __name__ == "__main__":
    main()
