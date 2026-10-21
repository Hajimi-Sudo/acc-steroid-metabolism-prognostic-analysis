#!/usr/bin/env python3
"""Remote-only short signature, internal validation, and interaction audit."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2, norm
from sklearn.model_selection import RepeatedKFold

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
FIRST = ROOT / "results" / "first_wave"
OUT = ROOT / "results" / "signature_interactions"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]


def event_status(value: object) -> float:
    text = str(value).upper()
    if "DECEASED" in text or "1:" in text:
        return 1.0
    if "LIVING" in text or "0:" in text:
        return 0.0
    return np.nan


def fit_cox(train: pd.DataFrame, time_col: str, event_col: str, covars: list[str],
            penalizer: float = 0.1, l1_ratio: float = 0.5) -> CoxPHFitter | None:
    keep = train[[time_col, event_col, *covars]].replace([np.inf, -np.inf], np.nan).dropna()
    keep = keep[keep[time_col] > 0]
    if len(keep) < 18 or keep[event_col].sum() < max(4, len(covars) + 1):
        return None
    try:
        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        model.fit(keep, duration_col=time_col, event_col=event_col)
        return model
    except Exception:
        return None


def standardize(train: pd.DataFrame, test: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = train[genes].mean()
    sd = train[genes].std(ddof=0).replace(0, np.nan)
    return (train[genes] - mu).fillna(0.0) / sd.fillna(1.0), (test[genes] - mu).fillna(0.0) / sd.fillna(1.0)


def select_genes(train: pd.DataFrame, genes: list[str], max_genes: int = 4) -> list[str]:
    rows = []
    for gene in genes:
        model = fit_cox(train[["OS_MONTHS", "OS_event", gene]], "OS_MONTHS", "OS_event", [gene], 0.01, 0.0)
        if model is None or gene not in model.summary.index:
            continue
        row = model.summary.loc[gene]
        rows.append((gene, float(row["p"]), abs(float(row["z"])), float(row["coef"])))
    ranked = sorted(rows, key=lambda x: (x[1], -x[2]))
    return [x[0] for x in ranked[:max_genes]]


def internal_validation(df: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(20261020)
    rows = []
    selections = []
    for repeat in range(10):
        splitter = RepeatedKFold(n_splits=5, n_repeats=1, random_state=20261020 + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            chosen = select_genes(train, genes, max_genes=4)
            selections.extend({"repeat": repeat + 1, "fold": fold, "gene": gene} for gene in chosen)
            if not chosen:
                continue
            train_z, test_z = standardize(train, test, chosen)
            train_fit = pd.concat([train[["OS_MONTHS", "OS_event"]].reset_index(drop=True), train_z.reset_index(drop=True)], axis=1)
            test_fit = pd.concat([test[["OS_MONTHS", "OS_event"]].reset_index(drop=True), test_z.reset_index(drop=True)], axis=1)
            model = fit_cox(train_fit, "OS_MONTHS", "OS_event", chosen, 0.1, 0.5)
            if model is None:
                continue
            risk = model.predict_partial_hazard(test_fit[chosen]).to_numpy().ravel()
            for idx, value in zip(test.index, risk):
                rows.append({"repeat": repeat + 1, "fold": fold, "row_index": int(idx), "risk": float(value), "genes": ",".join(chosen)})
    pred = pd.DataFrame(rows)
    sel = pd.DataFrame(selections)
    return pred, sel


def bootstrap_cindex(df: pd.DataFrame, risk_col: str, n_boot: int = 500) -> dict:
    valid = df[["OS_MONTHS", "OS_event", risk_col]].dropna()
    rng = np.random.RandomState(20261021)
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(valid), len(valid))
        sample = valid.iloc[idx]
        if sample["OS_event"].sum() == 0:
            continue
        values.append(float(concordance_index(sample["OS_MONTHS"], -sample[risk_col], sample["OS_event"])))
    if not values:
        return {"n_boot": 0}
    return {"n_boot": len(values), "cindex_median": float(np.median(values)), "cindex_low": float(np.quantile(values, 0.025)), "cindex_high": float(np.quantile(values, 0.975))}


def interaction_result(df: pd.DataFrame, modifier: str, time_col: str = "OS_MONTHS", event_col: str = "OS_event") -> dict:
    cols = [time_col, event_col, "locked_score", modifier, "score_x_modifier"]
    keep = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    keep = keep[keep[time_col] > 0]
    result = {"modifier": modifier, "n": int(len(keep)), "events": int(keep[event_col].sum()) if len(keep) else 0}
    if len(keep) < 24 or keep[event_col].sum() < 6 or keep[modifier].nunique() < 2:
        result["status"] = "insufficient_complete_cases_or_events"
        return result
    try:
        base = CoxPHFitter(penalizer=0.05)
        full = CoxPHFitter(penalizer=0.05)
        base.fit(keep[[time_col, event_col, "locked_score", modifier]], time_col, event_col)
        full.fit(keep[cols], time_col, event_col)
        llr = 2 * (full.log_likelihood_ - base.log_likelihood_)
        result.update({"status": "ok", "interaction_hr": float(np.exp(full.params_["score_x_modifier"])), "interaction_p_wald": float(full.summary.loc["score_x_modifier", "p"]), "interaction_p_lrt": float(chi2.sf(llr, 1)), "base_loglik": float(base.log_likelihood_), "full_loglik": float(full.log_likelihood_)})
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc)})
    return result


def main() -> None:
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    df = expr.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    numeric = np.log2(df[GENES].apply(pd.to_numeric, errors="coerce") + 1.0)
    z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    df = df[["sample_id", "patient_id", "OS_MONTHS", "OS_STATUS", "WEISS_SCORE_OVERALL", "HISTORY_ADRENAL_HORMONE_EXCESS", *GENES]].copy()
    df["OS_event"] = df["OS_STATUS"].map(event_status)
    for gene in GENES:
        df[gene] = z[gene]

    # Freeze a short score using the full discovery cohort only after OOF runs.
    selected = select_genes(df.dropna(subset=["OS_MONTHS", "OS_event"]), GENES, max_genes=4)
    train_z, _ = standardize(df, df, selected)
    train_fit = pd.concat([df[["OS_MONTHS", "OS_event"]].reset_index(drop=True), train_z.reset_index(drop=True)], axis=1)
    final = fit_cox(train_fit, "OS_MONTHS", "OS_event", selected, 0.1, 0.5)
    if final is None:
        raise RuntimeError("final short signature Cox model failed")
    df["locked_score"] = final.predict_partial_hazard(train_fit[selected]).to_numpy().ravel()
    df["locked_high"] = (df["locked_score"] >= df["locked_score"].median()).astype(int)

    oof, selections = internal_validation(df.dropna(subset=["OS_MONTHS", "OS_event"]), GENES)
    oof.to_csv(OUT / "oof_predictions.tsv", sep="\t", index=False)
    selections.to_csv(OUT / "gene_selection_frequency.tsv", sep="\t", index=False)
    df.to_csv(OUT / "tcga_locked_score_table.tsv", sep="\t", index=False)

    oof_summary = {"n_predictions": int(len(oof)), "unique_rows": int(oof["row_index"].nunique()) if len(oof) else 0}
    if len(oof) and oof["row_index"].nunique() >= 20:
        avg = oof.groupby("row_index", as_index=False)["risk"].mean()
        merged = df.reset_index().merge(avg, left_on="index", right_on="row_index", how="inner")
        oof_summary["cindex"] = float(concordance_index(merged["OS_MONTHS"], -merged["risk"], merged["OS_event"]))
        oof_summary["bootstrap"] = bootstrap_cindex(merged, "risk")
    else:
        oof_summary["status"] = "insufficient_oof_predictions"

    selected_counts = selections["gene"].value_counts().to_dict() if len(selections) else {}
    interaction_df = df.copy()
    interaction_df["WEISS_SCORE_OVERALL"] = pd.to_numeric(interaction_df["WEISS_SCORE_OVERALL"], errors="coerce")
    interaction_df["weiss_high"] = np.where(
        interaction_df["WEISS_SCORE_OVERALL"].notna(),
        (interaction_df["WEISS_SCORE_OVERALL"] >= 6).astype(float),
        np.nan,
    )
    # Preserve missing hormone annotations as missing. Treating an unrecorded
    # value as hormone-negative would inflate the interaction sample and change
    # the estimand.
    def hormone_indicator(value: object) -> float:
        if pd.isna(value) or str(value).strip().lower() in {"", "nan"}:
            return np.nan
        return 0.0 if str(value).strip().lower() == "none" else 1.0

    interaction_df["hormone_any"] = interaction_df["HISTORY_ADRENAL_HORMONE_EXCESS"].map(hormone_indicator)
    interactions = []
    for modifier in ["WEISS_SCORE_OVERALL", "weiss_high", "hormone_any"]:
        interaction_df["score_x_modifier"] = interaction_df["locked_score"] * interaction_df[modifier]
        interactions.append(interaction_result(interaction_df, modifier))

    summary = {
        "n_discovery": int(len(df)),
        "os_events": int(df["OS_event"].sum()),
        "selected_genes": selected,
        "selection_frequency_oof": selected_counts,
        "apparent_model_coefficients": {gene: float(final.params_.get(gene, np.nan)) for gene in selected},
        "oof": oof_summary,
        "interactions": interactions,
        "c1a_c1b_status": "blocked_not_available_in_current_verified_clinical_extract",
        "claim_ceiling": "internal computational association; no independent OS validation or causal/drug efficacy claim",
    }
    (OUT / "signature_interaction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
