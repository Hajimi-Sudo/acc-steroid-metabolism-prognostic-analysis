#!/usr/bin/env python3
"""Remote-only robustness analysis for the 22-gene steroid states.

The output is aggregate evidence only. Patient-level expression, sample IDs,
and clinical rows remain on the remote server.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
OUT = RES / "state_robustness"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
SEED = 20261020


def event_status(value: object) -> float:
    text = str(value).upper()
    if "DECEASED" in text or "1:" in text:
        return 1.0
    if "LIVING" in text or "0:" in text:
        return 0.0
    return np.nan


def tcga_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    df = expr.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    df["time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["event"] = df["OS_STATUS"].map(event_status)
    df = df.dropna(subset=["time", "event"] + GENES).reset_index(drop=True)
    df = df[df["time"] > 0].reset_index(drop=True)
    values = np.log2(df[GENES].apply(pd.to_numeric, errors="coerce") + 1.0)
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0).to_numpy(dtype=float)
    program = z.mean(axis=1)
    return df, z, program, np.arange(len(df))


def orient_labels(labels: np.ndarray, program: np.ndarray) -> np.ndarray:
    means = pd.Series(program).groupby(labels).mean()
    high = int(means.idxmax())
    return (labels == high).astype(int)


def initialization_stability(z: np.ndarray, program: np.ndarray) -> dict:
    reference = orient_labels(
        KMeans(n_clusters=2, n_init=50, random_state=SEED).fit_predict(z), program
    )
    aris = []
    for seed in range(100):
        labels = KMeans(n_clusters=2, n_init=25, random_state=seed).fit_predict(z)
        aris.append(float(adjusted_rand_score(reference, orient_labels(labels, program))))
    return {
        "n_runs": len(aris),
        "median_ari": float(np.median(aris)),
        "q25_ari": float(np.quantile(aris, 0.25)),
        "q75_ari": float(np.quantile(aris, 0.75)),
        "low_ari": float(np.quantile(aris, 0.025)),
        "high_ari": float(np.quantile(aris, 0.975)),
        "interpretation": "initialization stability only",
    }


def patient_resampling_stability(z: np.ndarray, program: np.ndarray, n_resamples: int = 1000) -> dict:
    rng = np.random.default_rng(SEED)
    n = z.shape[0]
    subset_n = int(round(0.8 * n))
    reference = orient_labels(
        KMeans(n_clusters=2, n_init=50, random_state=SEED).fit_predict(z), program
    )
    aris: list[float] = []
    pair_coassigned = np.zeros((n, n), dtype=float)
    pair_observed = np.zeros((n, n), dtype=float)
    for rep in range(n_resamples):
        subset = np.sort(rng.choice(n, size=subset_n, replace=False))
        labels = KMeans(n_clusters=2, n_init=25, random_state=SEED + rep).fit_predict(z[subset])
        labels = orient_labels(labels, program[subset])
        aris.append(float(adjusted_rand_score(reference[subset], labels)))
        observed = np.zeros((n, n), dtype=bool)
        observed[np.ix_(subset, subset)] = True
        pair_observed += observed
        pair_coassigned[np.ix_(subset, subset)] += labels[:, None] == labels[None, :]
    valid = pair_observed > 0
    np.fill_diagonal(valid, False)
    probabilities = pair_coassigned[valid] / pair_observed[valid]
    return {
        "n_resamples": n_resamples,
        "sampling_scheme": "without replacement; 80% of patients per resample",
        "subset_n": subset_n,
        "median_ari_vs_full_solution": float(np.median(aris)),
        "q25_ari": float(np.quantile(aris, 0.25)),
        "q75_ari": float(np.quantile(aris, 0.75)),
        "low_ari": float(np.quantile(aris, 0.025)),
        "high_ari": float(np.quantile(aris, 0.975)),
        "mean_ari": float(np.mean(aris)),
        "sd_ari": float(np.std(aris, ddof=1)),
        "fraction_ari_at_least_0_8": float(np.mean(np.asarray(aris) >= 0.8)),
        "pairwise_consensus": {
            "n_pairs_observed": int(probabilities.size),
            "mean_coassignment": float(np.mean(probabilities)),
            "median_coassignment": float(np.median(probabilities)),
            "low_95": float(np.quantile(probabilities, 0.025)),
            "high_95": float(np.quantile(probabilities, 0.975)),
            "fraction_at_least_0_8": float(np.mean(probabilities >= 0.8)),
        },
        "interpretation": "patient-resampling stability within TCGA-ACC; not independent external validation",
    }


def within_dispersion(data: np.ndarray, labels: np.ndarray) -> float:
    model = KMeans(n_clusters=int(len(np.unique(labels))), n_init=25, random_state=SEED)
    model.fit(data)
    return float(max(model.inertia_, np.finfo(float).tiny))


def k_selection(z: np.ndarray, max_k: int = 5, null_replicates: int = 100) -> dict:
    rng = np.random.default_rng(SEED + 17)
    minimum = z.min(axis=0)
    maximum = z.max(axis=0)
    rows = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=50, random_state=SEED)
        labels = model.fit_predict(z)
        silhouette = float(silhouette_score(z, labels)) if k >= 2 else None
        null_logs = []
        for _ in range(null_replicates):
            null = rng.uniform(minimum, maximum, size=z.shape)
            null_model = KMeans(n_clusters=k, n_init=15, random_state=SEED + k)
            null_model.fit(null)
            null_logs.append(float(np.log(max(null_model.inertia_, np.finfo(float).tiny))))
        observed_log = float(np.log(max(model.inertia_, np.finfo(float).tiny)))
        null_logs = np.asarray(null_logs)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": silhouette,
            "gap": float(null_logs.mean() - observed_log),
            "gap_se": float(np.sqrt(1.0 + 1.0 / null_replicates) * null_logs.std(ddof=1)),
            "cluster_sizes": ";".join(str(int(x)) for x in sorted(np.bincount(labels))),
        })
    table = pd.DataFrame(rows)
    silhouette_k = int(table.loc[table["silhouette"].idxmax(), "k"])
    gap_k = max_k
    for index in range(max_k - 1):
        current = table.iloc[index]
        following = table.iloc[index + 1]
        if current["gap"] >= following["gap"] - following["gap_se"]:
            gap_k = int(current["k"])
            break
    return {
        "candidate_k": list(range(1, max_k + 1)),
        "null_replicates": null_replicates,
        "table": table.to_dict(orient="records"),
        "best_silhouette_k": silhouette_k,
        "gap_one_se_k": gap_k,
        "interpretation": "k=2 is supported as the prespecified primary solution only when the selection metrics are reported alongside it",
    }


def external_partial_program() -> dict:
    scores = pd.read_csv(RES / "external_validation" / "GSE49278_steroid_scores.tsv", sep="\t")
    mapped = json.loads((RES / "external_validation" / "mapping_audit.json").read_text())
    genes = [gene for gene, count in mapped["gene_probe_coverage"].items() if count > 0]
    expr_cols = [f"{gene}_expr" for gene in genes if f"{gene}_expr" in scores.columns]
    values = scores[expr_cols].apply(pd.to_numeric, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0).to_numpy(dtype=float)
    program = z.mean(axis=1)
    raw_labels = KMeans(n_clusters=2, n_init=50, random_state=SEED).fit_predict(z)
    state = orient_labels(raw_labels, program)
    state_sizes = {"lower_score": int((state == 0).sum()), "higher_score": int((state == 1).sum())}

    audit = pd.read_excel(RES / "gse49278_survival_audit" / "MOESM14_source_fig5.xls", sheet_name="Fig. 5b", header=None)
    audit = audit.iloc[2:, :4].copy()
    audit.columns = ["acc_label", "os_event", "os_delay", "molecular_group"]
    audit["acc_label"] = audit["acc_label"].astype(str).str.strip()
    audit = audit[audit["acc_label"].str.match(r"^ACC\d+$", na=False)]
    bridge = pd.read_csv(PROC / "GSE49278_sample_metadata.tsv", sep="\t").rename(columns={"title": "acc_label"})
    merged = bridge.merge(pd.DataFrame({"sample_id": scores["sample_id"], "state": state}), on="sample_id", validate="one_to_one")
    merged = merged.merge(audit[["acc_label", "molecular_group"]], on="acc_label", how="left")
    groups = merged.dropna(subset=["molecular_group"])
    contingency = pd.crosstab(groups["state"], groups["molecular_group"])
    chi2, p_value, _, _ = chi2_contingency(contingency.to_numpy())
    phi2 = chi2 / max(len(groups), 1)
    r, c = contingency.shape
    cramers_v = float(np.sqrt(phi2 / max(min(r - 1, c - 1), 1)))
    age = pd.to_numeric(merged["age (years)"], errors="coerce")
    age_p = float(mannwhitneyu(age[state == 0], age[state == 1], alternative="two-sided").pvalue)
    gender = merged["gender"].astype(str)
    gender_table = pd.crosstab(state, gender)
    gender_p = None
    if gender_table.shape == (2, 2):
        gender_p = float(fisher_exact(gender_table.to_numpy())[1])
    rho, age_corr_p = spearmanr(program, age, nan_policy="omit")
    return {
        "cohort": "GSE49278",
        "platform": "GPL16686",
        "analysis": "external partial-program state analysis",
        "mapped_gene_count": len(expr_cols),
        "mapped_genes": [column[:-5] for column in expr_cols],
        "n_expression_samples": int(len(scores)),
        "state_sizes": state_sizes,
        "molecular_group_n": int(len(groups)),
        "molecular_group_levels": sorted(groups["molecular_group"].unique().tolist()),
        "state_by_molecular_group_counts": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in contingency.iterrows()
        },
        "molecular_group_chi_square_p": float(p_value),
        "molecular_group_cramers_v": cramers_v,
        "age_median_lower_state": float(np.nanmedian(age[state == 0])),
        "age_median_higher_state": float(np.nanmedian(age[state == 1])),
        "age_state_mann_whitney_p": age_p,
        "age_score_spearman_rho": float(rho),
        "age_score_spearman_p": float(age_corr_p),
        "gender_state_fisher_p": gender_p,
        "claim_status": "external partial-program state transportability only; not validation of the complete 22-gene state",
    }


def main() -> None:
    df, z, program, _ = tcga_matrix()
    reference = orient_labels(KMeans(n_clusters=2, n_init=50, random_state=SEED).fit_predict(z), program)
    tcga = {
        "n": int(len(df)),
        "events": int(df["event"].sum()),
        "reference_state_sizes": {"lower_score": int((reference == 0).sum()), "higher_score": int((reference == 1).sum())},
        "initialization_stability": initialization_stability(z, program),
        "patient_resampling_stability": patient_resampling_stability(z, program),
        "k_selection": k_selection(z),
    }
    summary = {
        "cohort": "TCGA-ACC",
        "tcga": tcga,
        "external_gse49278": external_partial_program(),
        "claim_status": "patient-resampling and k-selection evidence strengthen internal state robustness; GSE49278 remains a partial-program external exploration",
    }
    (OUT / "state_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(tcga["k_selection"]["table"]).to_csv(OUT / "k_selection_metrics.tsv", sep="\t", index=False)
    pd.DataFrame([tcga["patient_resampling_stability"]]).drop(columns=["pairwise_consensus"]).to_csv(OUT / "patient_resampling_summary.tsv", sep="\t", index=False)
    pd.DataFrame([
        {
            "analysis": "random initialization",
            "n_runs": tcga["initialization_stability"]["n_runs"],
            "q025": tcga["initialization_stability"]["low_ari"],
            "q25": tcga["initialization_stability"]["q25_ari"],
            "median": tcga["initialization_stability"]["median_ari"],
            "q75": tcga["initialization_stability"]["q75_ari"],
            "q975": tcga["initialization_stability"]["high_ari"],
            "fraction_ari_at_least_0_8": None,
        },
        {
            "analysis": "80% patient resampling",
            "n_runs": tcga["patient_resampling_stability"]["n_resamples"],
            "q025": tcga["patient_resampling_stability"]["low_ari"],
            "q25": tcga["patient_resampling_stability"]["q25_ari"],
            "median": tcga["patient_resampling_stability"]["median_ari_vs_full_solution"],
            "q75": tcga["patient_resampling_stability"]["q75_ari"],
            "q975": tcga["patient_resampling_stability"]["high_ari"],
            "fraction_ari_at_least_0_8": tcga["patient_resampling_stability"]["fraction_ari_at_least_0_8"],
        },
    ]).to_csv(OUT / "resampling_distribution_summary.tsv", sep="\t", index=False)
    pd.DataFrame([tcga["patient_resampling_stability"]["pairwise_consensus"]]).to_csv(OUT / "consensus_summary.tsv", sep="\t", index=False)
    pd.DataFrame([summary["external_gse49278"]]).drop(columns=["state_by_molecular_group_counts", "mapped_genes", "molecular_group_levels"]).to_csv(OUT / "external_state_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
