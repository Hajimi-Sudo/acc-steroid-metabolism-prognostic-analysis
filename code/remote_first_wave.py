#!/usr/bin/env python
"""Remote-only first-wave ACC steroid score analysis."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import kruskal, spearmanr


ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "first_wave"
OUT.mkdir(parents=True, exist_ok=True)


def status_event(value):
    text = str(value).upper()
    if "1:" in text or "DECEASED" in text or "RECUR" in text or "PROGRESS" in text:
        return 1
    if "0:" in text or "LIVING" in text or "DISEASEFREE" in text or "DISEASE FREE" in text:
        return 0
    return np.nan


def cox(df: pd.DataFrame, time_col: str, event_col: str, covars: list[str]) -> dict:
    keep = df[[time_col, event_col, *covars]].copy()
    keep[time_col] = pd.to_numeric(keep[time_col], errors="coerce")
    keep[event_col] = keep[event_col].map(status_event) if keep[event_col].dtype == object else pd.to_numeric(keep[event_col], errors="coerce")
    for col in covars:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")
    keep = keep.replace([np.inf, -np.inf], np.nan).dropna()
    keep = keep[keep[time_col] > 0]
    result = {"n": int(len(keep)), "events": int(keep[event_col].sum()) if len(keep) else 0}
    if len(keep) < 12 or keep[event_col].sum() < 3 or keep[event_col].sum() >= len(keep):
        result["status"] = "insufficient_events"
        return result
    try:
        model = CoxPHFitter(penalizer=0.01)
        model.fit(keep, duration_col=time_col, event_col=event_col)
        row = model.summary.loc[covars[0]]
        result.update({"status": "ok", "hr": float(np.exp(row["coef"])), "p": float(row["p"]), "ci_low": float(np.exp(row["coef lower 95%"])), "ci_high": float(np.exp(row["coef upper 95%"]))})
    except Exception as exc:
        result.update({"status": "error", "error": repr(exc)})
    return result


def main() -> None:
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    df = expr.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    genes = [x for x in expr.columns if x not in {"sample_id", "patient_id"}]
    log_expr = np.log2(df[genes].apply(pd.to_numeric, errors="coerce") + 1.0)
    z = (log_expr - log_expr.mean()) / log_expr.std(ddof=0).replace(0, np.nan)
    df["steroid_score"] = z.mean(axis=1, skipna=False)
    df["steroid_score_high"] = (df["steroid_score"] >= df["steroid_score"].median()).astype(int)
    df["OS_event"] = df["OS_STATUS"].map(status_event)
    df["DFS_event"] = df["DFS_STATUS"].map(status_event)
    df.to_csv(OUT / "tcga_acc_steroid_score_table.tsv", sep="\t", index=False)

    audit = {
        "n_expression": int(len(expr)),
        "n_merged": int(len(df)),
        "score_nonmissing": int(df["steroid_score"].notna().sum()),
        "os_time_nonmissing": int(pd.to_numeric(df["OS_MONTHS"], errors="coerce").notna().sum()),
        "os_events": int(df["OS_event"].sum(skipna=True)),
        "dfs_time_nonmissing": int(pd.to_numeric(df["DFS_MONTHS"], errors="coerce").notna().sum()),
        "dfs_events": int(df["DFS_event"].sum(skipna=True)),
        "weiss_nonmissing": int(pd.to_numeric(df["WEISS_SCORE_OVERALL"], errors="coerce").notna().sum()),
        "hormone_nonmissing": int(df["HISTORY_ADRENAL_HORMONE_EXCESS"].notna().sum()),
        "C1A_C1B_available": False,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    survival = []
    for endpoint, time_col, event_col in [("OS", "OS_MONTHS", "OS_event"), ("DFS", "DFS_MONTHS", "DFS_event")]:
        survival.append({"endpoint": endpoint, **cox(df, time_col, event_col, ["steroid_score"])})
        valid = df[[time_col, event_col, "steroid_score_high"]].copy().dropna()
        valid = valid[valid[time_col] > 0]
        if valid["steroid_score_high"].nunique() == 2:
            high = valid[valid["steroid_score_high"] == 1]
            low = valid[valid["steroid_score_high"] == 0]
            test = logrank_test(high[time_col], low[time_col], high[event_col], low[event_col])
            survival[-1].update({"high_n": int(len(high)), "low_n": int(len(low)), "logrank_p": float(test.p_value)})
    pd.DataFrame(survival).to_csv(OUT / "survival_results.tsv", sep="\t", index=False)

    gene_rows = []
    for gene in genes:
        gene_rows.append({"gene": gene, "os": cox(df.assign(gene_score=z[gene]), "OS_MONTHS", "OS_event", ["gene_score"]), "dfs": cox(df.assign(gene_score=z[gene]), "DFS_MONTHS", "DFS_event", ["gene_score"])})
    gene_out = []
    for row in gene_rows:
        for endpoint in ["os", "dfs"]:
            item = {"gene": row["gene"], "endpoint": endpoint, **row[endpoint]}
            gene_out.append(item)
    pd.DataFrame(gene_out).to_csv(OUT / "gene_univariate_results.tsv", sep="\t", index=False)

    weiss = df[["steroid_score", "WEISS_SCORE_OVERALL"]].apply(pd.to_numeric, errors="coerce").dropna()
    weiss_result = {"n": int(len(weiss))}
    if len(weiss) >= 5:
        rho, p = spearmanr(weiss["steroid_score"], weiss["WEISS_SCORE_OVERALL"])
        weiss_result.update({"spearman_rho": float(rho), "p": float(p)})
    (OUT / "weiss_association.json").write_text(json.dumps(weiss_result, indent=2), encoding="utf-8")

    hormone = df[["steroid_score", "HISTORY_ADRENAL_HORMONE_EXCESS"]].dropna()
    groups = [g["steroid_score"].to_numpy() for _, g in hormone.groupby("HISTORY_ADRENAL_HORMONE_EXCESS") if len(g) >= 2]
    hormone_result = {"n": int(len(hormone)), "groups": int(len(groups))}
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        hormone_result.update({"kruskal_h": float(stat), "p": float(p)})
    (OUT / "hormone_association.json").write_text(json.dumps(hormone_result, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "survival": survival, "weiss": weiss_result, "hormone": hormone_result}, indent=2))


if __name__ == "__main__":
    main()
