#!/usr/bin/env python3
"""Build aggregate manuscript tables and figure inputs on the remote server."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "results" / "manuscript_inputs"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    robustness = read_json("results/robustness/leakage_free_robustness_summary.json")
    consistency = read_json("results/model_consistency/model_consistency_summary.json")
    external = read_json("results/external_validation/GSE19750_locked_signature_validation.json")
    followup = read_json("results/external_validation/GSE33371_locked_signature_validation.json")
    sensitivity = read_json("results/external_validation/GSE19750_locked_signature_sensitivity.json")
    mapping = read_json("results/external_validation/mapping_audit.json")
    interactions = read_json("results/signature_interactions/signature_interaction_summary.json")
    audit_geo = read_json("manifests/GSE19750_survival_audit.json")
    rows = [
        {"cohort": "TCGA-ACC", "role": "discovery/internal validation", "endpoint": "OS", "n": 79, "events": 28, "platform": "RNA-seq", "status": "primary"},
        {"cohort": "GSE19750", "role": "adult external test", "endpoint": "OS", "n": external["n_acc_complete_os"], "events": external["events"], "platform": "GPL570", "status": "small directional test"},
        {"cohort": "GSE33371", "role": "same-biological-sample follow-up layer of GSE10927", "endpoint": "OS", "n": followup["n_acc_complete_positive_os"], "events": followup["events"], "platform": "GPL570", "status": "not independent from GSE10927"},
        {"cohort": "GSE49278", "role": "expression-layer transportability", "endpoint": "not available", "n": 44, "events": "NA", "platform": "GPL16686", "status": "partial 10/22-gene coverage"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "table1_cohorts.tsv", sep="\t", index=False)
    model_rows = [
        {"analysis": "Feature-selection repeated 5-fold OOF", "metric": "C-index", "estimate": consistency["feature_selection_repeated_oof"]["cindex"], "n": consistency["feature_selection_repeated_oof"]["unique_cases"], "events": consistency["feature_selection_repeated_oof"]["events"]},
        {"analysis": "Fixed four-gene repeated 5-fold OOF", "metric": "C-index", "estimate": consistency["fixed_four_gene_repeated_oof"]["cindex"], "n": consistency["fixed_four_gene_repeated_oof"]["unique_cases"], "events": consistency["fixed_four_gene_repeated_oof"]["events"]},
        {"analysis": "GSE19750 locked score", "metric": "C-index", "estimate": external["cindex"], "n": external["n_acc_complete_os"], "events": external["events"]},
        {"analysis": "GSE19750 locked score", "metric": "Cox HR", "estimate": external["cox_hr_per_score_unit"], "n": external["n_acc_complete_os"], "events": external["events"]},
        {"analysis": "GSE33371 locked score follow-up layer", "metric": "C-index", "estimate": followup["cindex"], "n": followup["n_acc_complete_positive_os"], "events": followup["events"]},
        {"analysis": "GSE33371 locked score follow-up layer", "metric": "Cox HR", "estimate": followup["cox_hr_per_score_unit"], "n": followup["n_acc_complete_positive_os"], "events": followup["events"]},
    ]
    pd.DataFrame(model_rows).to_csv(OUT / "table2_primary_results.tsv", sep="\t", index=False)
    interaction_rows = []
    for item in interactions["interactions"]:
        interaction_rows.append({"modifier": item["modifier"], "n": item["n"], "events": item["events"], "interaction_hr": item.get("interaction_hr"), "p_wald": item.get("interaction_p_wald"), "p_lrt": item.get("interaction_p_lrt")})
    pd.DataFrame(interaction_rows).to_csv(OUT / "table3_interactions.tsv", sep="\t", index=False)
    pd.DataFrame(sensitivity["variants"]).drop(columns=["bootstrap"], errors="ignore").to_csv(OUT / "table4_external_sensitivity.tsv", sep="\t", index=False)
    pd.DataFrame([{"gene": gene, "probe_count": count, "mapped": count > 0} for gene, count in mapping["gene_probe_coverage"].items()]).to_csv(OUT / "figure_external_mapping.tsv", sep="\t", index=False)
    pd.DataFrame([{"state": "lower-score", "n": 23, "events": 3}, {"state": "higher-score", "n": 56, "events": 25}]).to_csv(OUT / "figure_steroid_state_survival.tsv", sep="\t", index=False)
    summary = {
        "cohort_rows": len(rows),
        "primary_oof_cindex": consistency["feature_selection_repeated_oof"]["cindex"],
        "fixed_gene_oof_cindex": consistency["fixed_four_gene_repeated_oof"]["cindex"],
        "full_data_locked_apparent_cindex": consistency["full_data_locked_model"]["cindex"],
        "external_cindex": external["cindex"],
        "external_bootstrap": external["cindex_bootstrap"],
        "external_hr": external["cox_hr_per_score_unit"],
        "external_hr_ci": [external["cox_ci_low"], external["cox_ci_high"]],
        "external_p": external["cox_p"],
        "gse33371_cindex": followup["cindex"],
        "gse33371_bootstrap": followup["cindex_bootstrap"],
        "gse33371_hr": followup["cox_hr_per_score_unit"],
        "gse33371_hr_ci": [followup["cox_ci_low"], followup["cox_ci_high"]],
        "gse33371_p": followup["cox_p"],
        "state_logrank_p": robustness["subtype_analysis"]["k"]["2"]["logrank_p"],
        "weiss_interaction_p": interactions["interactions"][0]["interaction_p_lrt"],
        "hormone_interaction_p": interactions["interactions"][2].get("interaction_p_lrt"),
        "gse49278_mapped_gene_count": sum(count > 0 for count in mapping["gene_probe_coverage"].values()),
        "gse49278_n_samples": 44,
        "claim_boundary": "stability/state study with small directional GSE19750 OS test, same-biological-sample GSE33371 follow-up layer, and independent partial-program GSE49278 OS evidence",
    }
    (OUT / "manuscript_result_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
