#!/usr/bin/env python3
"""Audit consistency between the locked four-gene model and OOF workflows.

All expression and clinical inputs are read on the remote server. Only
aggregate metrics and model-level summaries are intended for export.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "model_consistency"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
LOCKED = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]
SEED = 20261020


def event_status(value: object) -> float:
    text = str(value).upper()
    if "DECEASED" in text or "1:" in text:
        return 1.0
    if "LIVING" in text or "0:" in text:
        return 0.0
    return np.nan


def load_data() -> pd.DataFrame:
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    df = expr.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    df["time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["event"] = df["OS_STATUS"].map(event_status)
    df = df.dropna(subset=["time", "event", *GENES]).copy()
    df = df[df["time"] > 0].reset_index(drop=True)
    for gene in GENES:
        df[gene] = np.log2(pd.to_numeric(df[gene], errors="coerce") + 1.0)
    return df[["patient_id", "time", "event", *GENES]].reset_index(drop=True)


def standardize(train: pd.DataFrame, test: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = train[genes].mean()
    sd = train[genes].std(ddof=0).replace(0, np.nan)
    return ((train[genes] - mu) / sd).fillna(0.0), ((test[genes] - mu) / sd).fillna(0.0)


def fit_cox(train_z: pd.DataFrame, genes: list[str]) -> CoxPHFitter:
    fit_data = pd.concat([train[["time", "event"]].reset_index(drop=True), train_z.reset_index(drop=True)], axis=1)
    model = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    model.fit(fit_data, "time", "event")
    return model


def select_genes(train: pd.DataFrame) -> list[str]:
    rows = []
    for gene in GENES:
        keep = train[["time", "event", gene]].dropna()
        if len(keep) < 18 or keep["event"].sum() < 4:
            continue
        try:
            model = CoxPHFitter(penalizer=0.01)
            model.fit(keep, "time", "event")
            row = model.summary.loc[gene]
            rows.append((gene, float(row["p"]), -abs(float(row["z"]))))
        except Exception:
            continue
    rows.sort(key=lambda row: (row[1], row[2]))
    return [row[0] for row in rows[:4]]


def patient_bootstrap(df: pd.DataFrame, n_boot: int = 5000) -> dict:
    valid = df[["time", "event", "risk"]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(SEED + 91)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(valid), len(valid))
        sample = valid.iloc[index]
        if sample["event"].sum() == 0:
            continue
        values.append(float(concordance_index(sample["time"], -sample["risk"], sample["event"])))
    return {
        "n_boot": len(values),
        "median": float(np.median(values)),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
    }


def summarize_predictions(df: pd.DataFrame, prediction_rows: list[dict]) -> tuple[dict, pd.DataFrame]:
    predictions = pd.DataFrame(prediction_rows)
    averaged = predictions.groupby("row_index", as_index=False)["risk"].mean()
    merged = df.reset_index().merge(averaged, left_on="index", right_on="row_index", how="inner")
    result = {
        "n_predictions": int(len(predictions)),
        "unique_cases": int(merged["row_index"].nunique()),
        "events": int(merged["event"].sum()),
        "cindex": float(concordance_index(merged["time"], -merged["risk"], merged["event"])),
        "bootstrap": patient_bootstrap(merged),
    }
    return result, merged[["row_index", "time", "event", "risk"]]


def run_repeated_oof(df: pd.DataFrame, fixed_genes: list[str] | None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    for repeat in range(10):
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            chosen = list(fixed_genes) if fixed_genes is not None else select_genes(train)
            selections.extend({"repeat": repeat + 1, "fold": fold, "gene": gene} for gene in chosen)
            train_z, test_z = standardize(train, test, chosen)
            try:
                model = fit_cox(train, train_z, chosen)
                risk = model.predict_partial_hazard(test_z[chosen]).to_numpy().ravel()
            except Exception:
                continue
            rows.extend({"repeat": repeat + 1, "fold": fold, "row_index": int(idx), "risk": float(value)}
                        for idx, value in zip(test.index, risk))
    summary, patient_predictions = summarize_predictions(df, rows)
    return summary, patient_predictions, pd.DataFrame(selections)


def fit_cox(train: pd.DataFrame, train_z: pd.DataFrame, genes: list[str]) -> CoxPHFitter:
    fit_data = pd.concat([train[["time", "event"]].reset_index(drop=True), train_z[genes].reset_index(drop=True)], axis=1)
    model = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    model.fit(fit_data, "time", "event")
    return model


def main() -> None:
    df = load_data()

    full_z, _ = standardize(df, df, LOCKED)
    full_model = fit_cox(df, full_z, LOCKED)
    apparent_risk = full_model.predict_partial_hazard(full_z[LOCKED]).to_numpy().ravel()
    apparent = df[["time", "event"]].copy()
    apparent["risk"] = apparent_risk
    apparent_summary = {
        "n": int(len(apparent)),
        "events": int(apparent["event"].sum()),
        "cindex": float(concordance_index(apparent["time"], -apparent["risk"], apparent["event"])),
        "interpretation": "apparent discrimination of the final full-data locked four-gene model; not internal validation",
        "coefficients": {gene: float(full_model.params_[gene]) for gene in LOCKED},
    }

    fixed_oof, fixed_patient, fixed_selection = run_repeated_oof(df, LOCKED)
    workflow_oof, workflow_patient, workflow_selection = run_repeated_oof(df, None)
    fixed_selection_counts = fixed_selection["gene"].value_counts().to_dict()
    workflow_selection_counts = workflow_selection["gene"].value_counts().to_dict()

    summary = {
        "cohort": "TCGA-ACC",
        "n": int(len(df)),
        "events": int(df["event"].sum()),
        "locked_genes": LOCKED,
        "full_data_locked_model": apparent_summary,
        "fixed_four_gene_repeated_oof": {
            **fixed_oof,
            "selection": "gene set fixed a priori; coefficients and training-fold standardization refit within each fold",
            "interpretation": "internal patient-level performance of the fixed gene-set workflow; not the apparent full-data locked score",
        },
        "feature_selection_repeated_oof": {
            **workflow_oof,
            "selection": "four genes selected inside each training fold; standardization and penalized Cox fit inside each training fold",
            "interpretation": "internal patient-level performance of the complete feature-selection workflow",
        },
        "fixed_gene_selection_frequency": fixed_selection_counts,
        "feature_selection_frequency": workflow_selection_counts,
        "comparison_note": "The final full-data locked model, the fixed-gene repeated OOF workflow, and the feature-selection repeated OOF workflow are distinct estimands and must not share one C-index label.",
    }
    (OUT / "model_consistency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"model_object": "full-data locked four-gene model", "estimate": apparent_summary["cindex"], "low": np.nan, "high": np.nan, "n": len(df), "events": int(df["event"].sum()), "interpretation": "apparent"},
        {"model_object": "fixed four-gene repeated OOF", "estimate": fixed_oof["cindex"], "low": fixed_oof["bootstrap"]["low_95"], "high": fixed_oof["bootstrap"]["high_95"], "n": fixed_oof["unique_cases"], "events": fixed_oof["events"], "interpretation": "internal validation"},
        {"model_object": "feature-selection repeated OOF", "estimate": workflow_oof["cindex"], "low": workflow_oof["bootstrap"]["low_95"], "high": workflow_oof["bootstrap"]["high_95"], "n": workflow_oof["unique_cases"], "events": workflow_oof["events"], "interpretation": "internal validation"},
    ]).to_csv(OUT / "model_consistency_metrics.tsv", sep="\t", index=False)
    pd.DataFrame([{"gene": gene, "fixed_gene_set_oof_frequency": fixed_selection_counts.get(gene, 0), "feature_selection_oof_frequency": workflow_selection_counts.get(gene, 0)} for gene in GENES]).to_csv(OUT / "model_consistency_selection_frequency.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
