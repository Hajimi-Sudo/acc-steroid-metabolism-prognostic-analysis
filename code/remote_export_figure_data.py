#!/usr/bin/env python3
"""Export aggregate, manuscript-facing figure data on the remote server.

The export intentionally excludes raw matrices and patient-level rows. It is
the source-data bundle for plotting and visual review, not an analysis input
bundle.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path("/root/autodl-tmp/acc-steroid-metabolism")
RES = ROOT / "results"
OUT = RES / "figure_data_export"
OUT.mkdir(parents=True, exist_ok=True)


def copy_result(relative: str, output_name: str | None = None) -> Path:
    source = RES / relative
    target = OUT / (output_name or source.name)
    shutil.copy2(source, target)
    return target


def write_text(name: str, text: str) -> Path:
    target = OUT / name
    target.write_text(text)
    return target


def main() -> None:
    for path in OUT.iterdir():
        if path.is_file() and ("_v2" in path.name or "_v3" in path.name or path.name == "Figure3_gene_selection_frequency.tsv"):
            path.unlink()
    copied = []

    # Figure 1 and cohort-level evidence layers.
    for source, target in [
        ("manuscript_inputs/table1_cohorts.tsv", "table1_cohorts.tsv"),
        ("manuscript_inputs/table3_interactions.tsv", "Figure4_interactions.tsv"),
        ("external_validation/mapping_audit.json", "Figure5_mapping_audit.json"),
        ("external_validation/GSE33371_overlap_audit.json", "GSE33371_overlap_audit.json"),
        ("external_validation/GSE33371_locked_signature_validation.json", "GSE33371_locked_signature_validation.json"),
        ("state_robustness/state_robustness_summary.json", "Figure2_state_robustness.json"),
        ("model_consistency/model_consistency_summary.json", "Figure3_model_consistency.json"),
        ("model_consistency/model_consistency_metrics.tsv", "Figure3_model_consistency.tsv"),
        ("model_consistency/model_consistency_selection_frequency.tsv", "Figure3_model_consistency_selection_frequency.tsv"),
        ("external_validation/GPL16686_refseq_probe_mapping.tsv", "Figure5_probe_mapping.tsv"),
        ("druggability/candidate_target_prioritization.tsv", "Figure5_candidate_targets.tsv"),
        ("biology_associations/biology_association_summary.json", "Figure6_biology_association_summary.json"),
    ]:
        copied.append(copy_result(source, target))

    # Figure 1/2 source files are written by the publication figure generator.
    for name in [
        "Figure1_cohort_roles.tsv",
        "Figure1_state_summary.tsv",
        "Figure1_state_km_curve_coordinates.tsv",
        "Figure1_model_performance.tsv",
        "Figure1_gene_selection_frequency.tsv",
        "Figure2_km_curve_coordinates.tsv",
        "Figure2_state_gene_means.tsv",
        "Figure2_state_summary.tsv",
        "Figure2_state_robustness.tsv",
        "Figure2_k_selection_metrics.tsv",
        "Figure2_external_state_by_molecular_group.tsv",
        "Figure2_external_state_summary.tsv",
        "Figure3_oof_risk_histogram_bins.tsv",
        "Figure5_probe_coverage.tsv",
    ]:
        path = OUT / name
        if path.exists() and path not in copied:
            copied.append(path)

    # Figure 2: state-level summaries only; no patient-level heatmap rows leave
    # the server.
    score = pd.read_csv(RES / "first_wave" / "tcga_acc_steroid_score_table.tsv", sep="\t")
    genes = [
        "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
        "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
        "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
        "POR", "SCARB1", "STAR",
    ]
    numeric = score[genes].apply(pd.to_numeric, errors="coerce")
    log_values = np.log2(numeric + 1.0)
    z_values = (log_values - log_values.mean()) / log_values.std(ddof=0)
    z_values = z_values.fillna(0.0)
    labels = KMeans(n_clusters=2, n_init=50, random_state=20261020).fit_predict(z_values)
    program = z_values.mean(axis=1)
    high_cluster = int(pd.Series(program).groupby(labels).mean().idxmax())
    state_labels = np.where(labels == high_cluster, "Higher-score", "Lower-score")
    state = score.assign(
        state=state_labels,
        event=pd.to_numeric(score["OS_event"], errors="coerce"),
        score_value=program,
    )
    state_summary = (
        state.groupby("state", dropna=False)
        .agg(n=("state", "size"), events=("event", "sum"),
             mean_score=("score_value", "mean"), median_score=("score_value", "median"))
        .reset_index()
    )
    state_summary.to_csv(OUT / "Figure2_state_summary.tsv", sep="\t", index=False)
    copied.append(OUT / "Figure2_state_summary.tsv")
    z_values["state"] = state["state"].to_numpy()
    gene_means = z_values.groupby("state")[genes].mean().T.reset_index().rename(columns={"index": "gene"})
    gene_means.to_csv(OUT / "Figure2_state_gene_means.tsv", sep="\t", index=False)
    copied.append(OUT / "Figure2_state_gene_means.tsv")

    # Figure 3: all values needed for the stability/performance panels.
    copied.append(copy_result("signature_interactions/gene_selection_frequency.tsv", "Figure3_selection_frequency.tsv"))
    robust = json.loads((RES / "robustness" / "leakage_free_robustness_summary.json").read_text())
    consistency = json.loads((RES / "model_consistency" / "model_consistency_summary.json").read_text())
    external = json.loads((RES / "external_validation" / "GSE19750_locked_signature_validation.json").read_text())
    followup = json.loads((RES / "external_validation" / "GSE33371_locked_signature_validation.json").read_text())
    sensitivity = json.loads((RES / "external_validation" / "GSE19750_locked_signature_sensitivity.json").read_text())
    interactions = json.loads((RES / "signature_interactions" / "signature_interaction_summary.json").read_text())
    external_boot = external["cindex_bootstrap"]
    performance = pd.DataFrame([
        {"layer": "TCGA feature-selection repeated OOF", "estimate": consistency["feature_selection_repeated_oof"]["cindex"],
         "low": consistency["feature_selection_repeated_oof"]["bootstrap"]["low_95"], "high": consistency["feature_selection_repeated_oof"]["bootstrap"]["high_95"], "n": 79, "events": 28},
        {"layer": "TCGA fixed four-gene repeated OOF", "estimate": consistency["fixed_four_gene_repeated_oof"]["cindex"],
         "low": consistency["fixed_four_gene_repeated_oof"]["bootstrap"]["low_95"], "high": consistency["fixed_four_gene_repeated_oof"]["bootstrap"]["high_95"], "n": 79, "events": 28},
        {"layer": "GSE19750 OS test", "estimate": external["cindex"],
         "low": external_boot["low_95"], "high": external_boot["high_95"],
         "n": external["n_acc_complete_os"], "events": external["events"]},
        {"layer": "GSE33371 follow-up", "estimate": followup["cindex"],
         "low": followup["cindex_bootstrap"]["low_95"], "high": followup["cindex_bootstrap"]["high_95"],
         "n": followup["n_acc_complete_positive_os"], "events": followup["events"]},
    ])
    performance.to_csv(OUT / "Figure3_performance.tsv", sep="\t", index=False)
    copied.append(OUT / "Figure3_performance.tsv")
    pd.DataFrame([
        {"gene": gene, "coefficient": coefficient}
        for gene, coefficient in interactions["apparent_model_coefficients"].items()
    ]).to_csv(OUT / "Figure3_locked_coefficients.tsv", sep="\t", index=False)
    copied.append(OUT / "Figure3_locked_coefficients.tsv")
    pd.DataFrame(sensitivity["variants"]).drop(columns=["bootstrap"], errors="ignore").to_csv(OUT / "Figure3_score_sensitivity.tsv", sep="\t", index=False)
    copied.append(OUT / "Figure3_score_sensitivity.tsv")

    manifest = {
        "project": "acc-steroid-metabolism",
        "source": "remote frozen result artifacts",
        "privacy_boundary": "aggregate figure data only; no raw matrix or patient-level rows",
        "files": [],
    }
    for path in sorted(set(copied)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"].append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    write_text("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    write_text("README.txt", "Aggregate source data for Figures 1-5. Generated on the remote server from frozen result artifacts.\n"
               "GSE33371 is a same-biological-sample follow-up layer of GSE10927, not an independent expression cohort.\n"
               "No raw expression matrix or patient-level clinical rows are included.\n")
    print(json.dumps({"output": str(OUT), "files": [x["file"] for x in manifest["files"]]}, indent=2))


if __name__ == "__main__":
    main()
