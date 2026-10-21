#!/usr/bin/env python3
"""Remote-only biological association and incremental prognostic audit.

This module consumes the already frozen TCGA expression/clinical inputs and
locked four-gene score. It deliberately does not tune the score or use any
external outcome to select genes. Because the remote project currently stores
the 22-gene matrix rather than a genome-wide matrix, the differential analysis
is explicitly limited to the prespecified steroid program.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import kruskal, mannwhitneyu, spearmanr

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
FIRST = ROOT / "results" / "first_wave"
LOCKED = ROOT / "results" / "signature_interactions" / "tcga_locked_score_table.tsv"
OUT = ROOT / "results" / "biology_associations"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
LOCKED_GENES = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]


def fdr_bh(values: list[float]) -> list[float]:
    order = np.argsort(values)
    out = np.full(len(values), np.nan, dtype=float)
    running = 1.0
    for rank in range(len(values), 0, -1):
        idx = order[rank - 1]
        running = min(running, values[idx] * len(values) / rank)
        out[idx] = running
    return out.tolist()


def bootstrap_ci(values: np.ndarray, statistic, seed: int = 20261020, n_boot: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        try:
            result = statistic(sample)
            if np.isfinite(result):
                estimates.append(float(result))
        except Exception:
            continue
    if not estimates:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.median(estimates)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)))


def bootstrap_spearman(frame: pd.DataFrame, x: str, y: str, seed: int = 20261020, n_boot: int = 5000) -> dict:
    keep = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sample = keep[rng.integers(0, len(keep), len(keep))]
        if len(np.unique(sample[:, 0])) < 3 or len(np.unique(sample[:, 1])) < 3:
            continue
        rho, _ = spearmanr(sample[:, 0], sample[:, 1])
        if np.isfinite(rho):
            estimates.append(float(rho))
    return {"median": float(np.median(estimates)), "low_95": float(np.quantile(estimates, 0.025)), "high_95": float(np.quantile(estimates, 0.975)), "n_boot": len(estimates)}


def score_group_effect(values: pd.Series, groups: pd.Series) -> dict:
    keep = pd.DataFrame({"score": values, "group": groups}).dropna()
    arrays = [g["score"].to_numpy() for _, g in keep.groupby("group") if len(g) >= 2]
    result = {"n": int(len(keep)), "groups": int(len(arrays)), "group_sizes": {str(k): int(len(v)) for k, v in keep.groupby("group")}}
    if len(arrays) < 2:
        result["status"] = "insufficient_groups"
        return result
    stat, p = kruskal(*arrays)
    result.update({"status": "ok", "kruskal_h": float(stat), "p": float(p),
                   "median_by_group": {str(k): float(v["score"].median()) for k, v in keep.groupby("group")},
                   "mean_by_group": {str(k): float(v["score"].mean()) for k, v in keep.groupby("group")}})
    if len(arrays) == 2:
        u, up = mannwhitneyu(*arrays, alternative="two-sided")
        result["mann_whitney_u"] = float(u)
        result["mann_whitney_p"] = float(up)
        a, b = arrays
        result["rank_biserial"] = float(2 * u / (len(a) * len(b)) - 1)
    return result


def parse_stage(value: object) -> float:
    match = re.search(r"([1-4])", str(value))
    if match:
        return float(match.group(1))
    roman = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
    text = str(value).upper()
    for label, number in roman.items():
        if re.search(rf"\b{label}\b", text):
            return number
    return np.nan


def cox_fit(frame: pd.DataFrame, covars: list[str]) -> dict:
    keep = frame[["OS_MONTHS", "OS_event", *covars]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    keep = keep[keep["OS_MONTHS"] > 0]
    out = {"covariates": covars, "n": int(len(keep)), "events": int(keep["OS_event"].sum()) if len(keep) else 0}
    if len(keep) < 18 or out["events"] < max(5, len(covars) + 2) or out["events"] >= len(keep):
        out["status"] = "insufficient_events"
        return out
    try:
        model = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
        model.fit(keep, duration_col="OS_MONTHS", event_col="OS_event")
        out["status"] = "ok"
        out["log_likelihood"] = float(model.log_likelihood_)
        out["aic_partial"] = float(model.AIC_partial_)
        out["coefficients"] = {c: float(model.params_[c]) for c in covars}
        out["hazard_ratios"] = {c: float(np.exp(model.params_[c])) for c in covars}
        out["p_values"] = {c: float(model.summary.loc[c, "p"]) for c in covars}
        upper = "coef upper 95%" if "coef upper 95%" in model.summary.columns else "coef upper 95% "
        lower = "coef lower 95%" if "coef lower 95%" in model.summary.columns else "coef lower 95% "
        out["ci95"] = {c: [float(np.exp(model.summary.loc[c, lower])), float(np.exp(model.summary.loc[c, upper]))] for c in covars}
        risk = model.predict_partial_hazard(keep).to_numpy().ravel()
        out["cindex"] = float(concordance_index(keep["OS_MONTHS"], -risk, keep["OS_event"]))
    except Exception as exc:
        out.update({"status": "error", "error": repr(exc)})
    return out


def main() -> None:
    locked = pd.read_csv(LOCKED, sep="\t")
    score = pd.to_numeric(locked["locked_score"], errors="coerce")
    locked["score"] = score
    locked["stage_num"] = locked["AJCC_PATHOLOGIC_TUMOR_STAGE"] if "AJCC_PATHOLOGIC_TUMOR_STAGE" in locked else np.nan
    if locked["stage_num"].isna().all():
        clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
        locked = locked.merge(clinical[["patient_id", "AJCC_PATHOLOGIC_TUMOR_STAGE"]], on="patient_id", how="left", suffixes=("", "_clinical"))
        locked["stage_num"] = locked["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(parse_stage)
    else:
        locked["stage_num"] = locked["stage_num"].map(parse_stage)
    locked["OS_MONTHS"] = pd.to_numeric(locked["OS_MONTHS"], errors="coerce")
    locked["OS_event"] = pd.to_numeric(locked["OS_event"], errors="coerce")

    hormone = score_group_effect(locked["score"], locked["HISTORY_ADRENAL_HORMONE_EXCESS"])
    weiss = locked[["score", "WEISS_SCORE_OVERALL"]].apply(pd.to_numeric, errors="coerce").dropna()
    rho, rho_p = spearmanr(weiss["score"], weiss["WEISS_SCORE_OVERALL"])
    weiss_result = {"n": int(len(weiss)), "spearman_rho": float(rho), "p": float(rho_p), "median_score_by_weiss": {str(k): float(v["score"].median()) for k, v in weiss.groupby("WEISS_SCORE_OVERALL")}}
    stage = locked[["score", "stage_num"]].dropna()
    stage_group = score_group_effect(stage["score"], stage["stage_num"])
    stage_rho, stage_p = spearmanr(stage["score"], stage["stage_num"])
    stage_group.update({"spearman_rho": float(stage_rho), "spearman_p": float(stage_p), "spearman_bootstrap": bootstrap_spearman(stage, "score", "stage_num")})
    hormone["group_median_bootstrap_95"] = {}
    rng = np.random.default_rng(20261020)
    hormone_frame = pd.DataFrame({"score": locked["score"], "group": locked["HISTORY_ADRENAL_HORMONE_EXCESS"]}).dropna()
    for group, values in hormone_frame.groupby("group")["score"]:
        arr = values.to_numpy()
        draws = [float(np.median(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(5000)]
        hormone["group_median_bootstrap_95"][str(group)] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    weiss_result["spearman_bootstrap"] = bootstrap_spearman(weiss, "score", "WEISS_SCORE_OVERALL")

    # Program-level differential analysis: this is the strongest analysis
    # supported by the frozen 22-gene input currently stored remotely.
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    meta = locked[["patient_id", "score"]].merge(expr[["patient_id", *GENES]], on="patient_id", how="inner")
    cutoff = meta["score"].median()
    meta["state"] = np.where(meta["score"] >= cutoff, "Higher-score", "Lower-score")
    diff = []
    log_expr = np.log2(meta[GENES].apply(pd.to_numeric, errors="coerce") + 1.0)
    for gene in GENES:
        high = log_expr.loc[meta["state"] == "Higher-score", gene].dropna()
        low = log_expr.loc[meta["state"] == "Lower-score", gene].dropna()
        u, p = mannwhitneyu(high, low, alternative="two-sided")
        diff.append({"gene": gene, "higher_n": int(len(high)), "lower_n": int(len(low)), "higher_median_log2": float(high.median()), "lower_median_log2": float(low.median()), "median_difference_log2": float(high.median() - low.median()), "u": float(u), "p": float(p)})
    diff_df = pd.DataFrame(diff)
    diff_df["fdr_bh"] = fdr_bh(diff_df["p"].tolist())
    diff_df = diff_df.sort_values(["fdr_bh", "p", "gene"])
    diff_df.to_csv(OUT / "program_gene_differential.tsv", sep="\t", index=False)

    pathways = {
        "cholesterol_transport_and_import": ["SCARB1", "STAR"],
        "steroidogenic_acute_regulation": ["STAR", "MC2R", "MRAP", "NR5A1", "NR0B1"],
        "steroidogenic_enzymes": [g for g in GENES if g not in {"LHCGR", "MC2R", "MRAP", "NR5A1", "NR0B1", "STAR"}],
        "androgen_and_reductase_branch": ["AKR1C3", "HSD17B1", "HSD17B3", "HSD11B1", "HSD11B2"],
    }
    path_rows = []
    z = np.log2(meta[GENES].apply(pd.to_numeric, errors="coerce") + 1.0)
    z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
    for name, genes in pathways.items():
        available = [g for g in genes if g in z]
        vals = z[available].mean(axis=1)
        high = vals[meta["state"] == "Higher-score"].dropna(); low = vals[meta["state"] == "Lower-score"].dropna()
        u, p = mannwhitneyu(high, low, alternative="two-sided")
        path_rows.append({"pathway": name, "genes": ";".join(available), "higher_n": int(len(high)), "lower_n": int(len(low)), "higher_median": float(high.median()), "lower_median": float(low.median()), "median_difference": float(high.median() - low.median()), "u": float(u), "p": float(p)})
    path_df = pd.DataFrame(path_rows); path_df["fdr_bh"] = fdr_bh(path_df["p"].tolist()); path_df.to_csv(OUT / "program_pathway_scores.tsv", sep="\t", index=False)

    baseline = {
        "stage_only": cox_fit(locked.assign(stage=locked["stage_num"]), ["stage"]),
        "score_only": cox_fit(locked, ["score"]),
        "stage_plus_score": cox_fit(locked.assign(stage=locked["stage_num"]), ["stage", "score"]),
        "weiss_only": cox_fit(locked.assign(weiss=pd.to_numeric(locked["WEISS_SCORE_OVERALL"], errors="coerce")), ["weiss"]),
        "weiss_plus_score": cox_fit(locked.assign(weiss=pd.to_numeric(locked["WEISS_SCORE_OVERALL"], errors="coerce")), ["weiss", "score"]),
    }
    # The nested likelihood-ratio comparison quantifies the incremental score
    # contribution without adding more covariates than the event count supports.
    from scipy.stats import chi2
    for base_name, full_name in [("stage_only", "stage_plus_score"), ("weiss_only", "weiss_plus_score")]:
        base, full = baseline[base_name], baseline[full_name]
        if base.get("status") == "ok" and full.get("status") == "ok":
            delta = 2.0 * (full["log_likelihood"] - base["log_likelihood"])
            baseline[f"incremental_{base_name}_plus_score"] = {"delta_log_likelihood": delta, "lrt_chi2": delta, "df": 1, "p": float(chi2.sf(delta, 1)), "delta_aic_partial": full["aic_partial"] - base["aic_partial"]}
    summary = {
        "analysis": "biological association and incremental prognostic audit",
        "cohort": "TCGA-ACC",
        "score_definition": "frozen full-data locked four-gene score; no external outcome tuning",
        "hormone_association": hormone,
        "weiss_association": weiss_result,
        "stage_association": stage_group,
        "program_differential_boundary": "22-gene steroid program only; genome-wide differential expression/pathway enrichment was not performed because no genome-wide matrix was available in the remote frozen input",
        "baseline_models": baseline,
        "c1a_c1b": "not available as verified TCGA sample-level labels; not inferred",
    }
    (OUT / "biology_association_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    pd.DataFrame([
        {"analysis": "Hormone secretion group", "estimate": hormone.get("kruskal_h"), "p": hormone.get("p"), "n": hormone.get("n"), "detail": "Kruskal-Wallis across annotated groups"},
        {"analysis": "Weiss score", "estimate": weiss_result["spearman_rho"], "p": weiss_result["p"], "n": weiss_result["n"], "detail": "Spearman rho"},
        {"analysis": "AJCC stage", "estimate": stage_group.get("spearman_rho"), "p": stage_group.get("spearman_p"), "n": stage_group.get("n"), "detail": "Spearman rho with ordinal stage"},
    ]).to_csv(OUT / "clinical_association_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
