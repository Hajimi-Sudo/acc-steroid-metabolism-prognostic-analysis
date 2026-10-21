#!/usr/bin/env python3
"""Remote-only leakage-free signature and steroid-state robustness analyses."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
FROZEN_GENES = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]


def event_status(value: object) -> float:
    text = str(value).upper()
    if "DECEASED" in text or "1:" in text:
        return 1.0
    if "LIVING" in text or "0:" in text:
        return 0.0
    return np.nan


def fit_single_gene(train: pd.DataFrame, gene: str) -> tuple[float, float, float] | None:
    keep = train[["time", "event", gene]].replace([np.inf, -np.inf], np.nan).dropna()
    keep = keep[keep["time"] > 0]
    if len(keep) < 18 or keep["event"].sum() < 4:
        return None
    try:
        model = CoxPHFitter(penalizer=0.01)
        model.fit(keep, "time", "event")
        row = model.summary.loc[gene]
        return float(row["p"]), abs(float(row["z"])), float(row["coef"])
    except Exception:
        return None


def select_genes(train: pd.DataFrame) -> list[str]:
    rows = []
    for gene in GENES:
        fitted = fit_single_gene(train, gene)
        if fitted is not None:
            rows.append((gene, *fitted))
    rows.sort(key=lambda row: (row[1], -row[2]))
    return [row[0] for row in rows[:4]]


def standardize(train: pd.DataFrame, test: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = train[genes].mean()
    sd = train[genes].std(ddof=0).replace(0, np.nan)
    return ((train[genes] - mu) / sd).fillna(0.0), ((test[genes] - mu) / sd).fillna(0.0)


def leakage_free_oof(df: pd.DataFrame) -> dict:
    rows = []
    selections = []
    for repeat in range(10):
        splitter = KFold(n_splits=5, shuffle=True, random_state=20261020 + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            chosen = select_genes(train)
            selections.extend({"repeat": repeat + 1, "fold": fold, "gene": gene} for gene in chosen)
            if not chosen:
                continue
            train_z, test_z = standardize(train, test, chosen)
            fit_data = pd.concat([train[["time", "event"]].reset_index(drop=True), train_z.reset_index(drop=True)], axis=1)
            test_data = pd.concat([test[["time", "event"]].reset_index(drop=True), test_z.reset_index(drop=True)], axis=1)
            if fit_data["event"].sum() < len(chosen) + 2:
                continue
            try:
                model = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
                model.fit(fit_data, "time", "event")
                risk = model.predict_partial_hazard(test_data[chosen]).to_numpy().ravel()
                rows.extend({"row_index": int(idx), "risk": float(value)} for idx, value in zip(test.index, risk))
            except Exception:
                continue
    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("no leakage-free OOF predictions")
    avg = pred.groupby("row_index", as_index=False)["risk"].mean()
    merged = df.reset_index().merge(avg, left_on="index", right_on="row_index", how="inner")
    values = [
        float(concordance_index(merged["time"], -merged["risk"], merged["event"]))
        for _ in [0]
    ]
    return {
        "n_predictions": int(len(pred)),
        "unique_cases": int(merged["row_index"].nunique()),
        "events_in_oof_cases": int(merged["event"].sum()),
        "cindex": values[0],
        "selection_frequency": pred.assign(gene="unknown").to_dict(orient="records")[:0],
        "selected_gene_frequency": pd.DataFrame(selections)["gene"].value_counts().to_dict(),
    }


def subtype_analysis(df: pd.DataFrame) -> dict:
    features = np.log2(df[GENES].apply(pd.to_numeric, errors="coerce") + 1.0)
    features = features.fillna(features.median())
    z = (features - features.mean()) / features.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    pathway_score = z.mean(axis=1)
    result = {"n": int(len(df)), "events": int(df["event"].sum()), "k": {}}
    for k in (2, 3):
        labels = KMeans(n_clusters=k, n_init=50, random_state=20261020).fit_predict(z)
        means = pd.Series(pathway_score).groupby(labels).mean().sort_values()
        low = int(means.index[0])
        adverse = (labels != low).astype(int) if k == 2 else labels
        frame = pd.DataFrame({"time": df["time"], "event": df["event"], "subtype": adverse}).dropna()
        frame = frame[frame["time"] > 0]
        groups = [frame.loc[frame["subtype"] == value, "time"] for value in sorted(frame["subtype"].unique())]
        group_events = [frame.loc[frame["subtype"] == value, "event"] for value in sorted(frame["subtype"].unique())]
        item = {
            "cluster_sizes": {str(int(value)): int((labels == value).sum()) for value in sorted(set(labels))},
            "mean_pathway_score_by_cluster": {str(int(value)): float(pathway_score[labels == value].mean()) for value in sorted(set(labels))},
            "event_counts_by_cluster": {str(int(value)): int(df.loc[labels == value, "event"].sum()) for value in sorted(set(labels))},
        }
        if len(groups) == 2:
            test = logrank_test(groups[0], groups[1], group_events[0], group_events[1])
            item["logrank_p"] = float(test.p_value)
        if k == 2:
            item["adverse_cluster"] = int(means.index[-1] != low)
            item["adverse_vs_low_logrank_p"] = item.get("logrank_p")
        result["k"][str(k)] = item
    stabilities = []
    reference = KMeans(n_clusters=2, n_init=50, random_state=20261020).fit_predict(z)
    for seed in range(100):
        labels = KMeans(n_clusters=2, n_init=25, random_state=seed).fit_predict(z)
        stabilities.append(float(adjusted_rand_score(reference, labels)))
    result["k2_stability"] = {
        "n_runs": len(stabilities),
        "median_ari": float(np.median(stabilities)),
        "low_ari": float(np.quantile(stabilities, 0.025)),
        "high_ari": float(np.quantile(stabilities, 0.975)),
    }
    return result


def main() -> None:
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    df = expr.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    df["time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["event"] = df["OS_STATUS"].map(event_status)
    df = df.dropna(subset=["time", "event"] + GENES).reset_index(drop=True)
    df = df[df["time"] > 0].reset_index(drop=True)
    oof = leakage_free_oof(df)
    subtype = subtype_analysis(df)
    summary = {
        "cohort": "TCGA-ACC",
        "analysis_n": int(len(df)),
        "events": int(df["event"].sum()),
        "leakage_free_repeated_nested_like_oof": oof,
        "subtype_analysis": subtype,
        "claim_status": "internal robustness and unsupervised state analysis; not independent clinical validation",
    }
    (OUT / "leakage_free_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
