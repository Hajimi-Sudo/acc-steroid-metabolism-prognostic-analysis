#!/usr/bin/env python3
"""Reanalyze GSE49278 after the official Assie et al. OS-table audit.

The sample bridge is taken from GEO !Sample_title (ACC labels), not inferred
from column order. The locked four-gene score is reported only if all four
genes are covered by native GPL16686 probes. Otherwise, only a clearly labelled
partial 22-gene program analysis is computed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
EXT = ROOT / "results" / "external_validation"
AUDIT = ROOT / "results" / "gse49278_survival_audit"
OUT = ROOT / "results" / "gse49278_survival_reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

LOCKED = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]
COEFFICIENTS = {
    "STAR": -0.315164785566944,
    "NR0B1": -0.14768321386120697,
    "HSD11B2": 0.489792020819889,
    "AKR1C3": -0.16957743987510482,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_os_table() -> pd.DataFrame:
    path = AUDIT / "MOESM14_source_fig5.xls"
    frame = pd.read_excel(path, sheet_name="Fig. 5b", header=None)
    frame = frame.iloc[2:, :4].copy()
    frame.columns = ["acc_label", "os_event", "os_delay", "molecular_group"]
    frame["acc_label"] = frame["acc_label"].astype(str).str.strip()
    frame["os_event"] = pd.to_numeric(frame["os_event"], errors="coerce")
    frame["os_delay"] = pd.to_numeric(frame["os_delay"], errors="coerce")
    frame = frame[frame["acc_label"].str.match(r"^ACC\d+$", na=False)].copy()
    frame["os_event"] = frame["os_event"].astype("Int64")
    return frame


def parse_geo_bridge() -> pd.DataFrame:
    metadata = pd.read_csv(PROC / "GSE49278_sample_metadata.tsv", sep="\t")
    metadata = metadata.rename(columns={"title": "acc_label"})
    metadata["acc_label"] = metadata["acc_label"].astype(str).str.strip()
    return metadata[["sample_id", "acc_label", "age (years)", "gender", "cell type", "platform"]]


def load_scores() -> pd.DataFrame:
    score = pd.read_csv(EXT / "GSE49278_steroid_scores.tsv", sep="\t")
    return score


def cox_summary(frame: pd.DataFrame, score_col: str) -> dict:
    valid = frame[["os_delay", "os_event", score_col]].copy()
    valid["os_delay"] = pd.to_numeric(valid["os_delay"], errors="coerce")
    valid["os_event"] = pd.to_numeric(valid["os_event"], errors="coerce")
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[valid["os_delay"] > 0].copy()
    result = {"n": int(len(valid)), "events": int(valid["os_event"].sum()) if len(valid) else 0}
    if len(valid) < 10 or result["events"] < 3 or result["events"] >= len(valid):
        result["status"] = "insufficient_events"
        return result
    result["cindex"] = float(concordance_index(valid["os_delay"], -valid[score_col], valid["os_event"]))
    try:
        model = CoxPHFitter(penalizer=0.01)
        model.fit(valid[["os_delay", "os_event", score_col]], duration_col="os_delay", event_col="os_event")
        row = model.summary.loc[score_col]
        result.update({
            "status": "ok",
            "hr_per_score_unit": float(np.exp(row["coef"])),
            "ci_low": float(np.exp(row["coef lower 95%"])),
            "ci_high": float(np.exp(row["coef upper 95%"])),
            "p": float(row["p"]),
        })
    except Exception as exc:
        result.update({"status": "cox_error", "error": repr(exc)})
    return result


def main() -> None:
    bridge = parse_geo_bridge()
    os_table = parse_os_table()
    scores = load_scores()
    merged = bridge.merge(scores, on="sample_id", how="left", validate="one_to_one")
    merged = merged.merge(os_table, on="acc_label", how="left", validate="one_to_one")
    merged["os_match"] = merged["os_delay"].notna() & merged["os_event"].notna()
    merged.to_csv(OUT / "GSE49278_GSM_ACC_OS_mapping.tsv", sep="\t", index=False)

    coverage = json.loads((EXT / "mapping_audit.json").read_text())
    mapped_genes = [gene for gene, count in coverage["gene_probe_coverage"].items() if count > 0]
    locked_missing = [gene for gene in LOCKED if gene not in mapped_genes]
    valid = merged[merged["os_match"] & (merged["os_delay"] > 0)].copy()
    partial_summary = cox_summary(merged, "steroid_score")

    # Recompute a score from the already audited mapped program genes, with no
    # survival-based selection or threshold optimization.
    gene_columns = [f"{gene}_expr" for gene in mapped_genes if f"{gene}_expr" in merged.columns]
    partial_gene_score = np.nan
    if gene_columns:
        z = merged[gene_columns].apply(pd.to_numeric, errors="coerce")
        z = (z - z.mean()) / z.std(ddof=0).replace(0, np.nan)
        merged["mapped_program_score_recomputed"] = z.mean(axis=1, skipna=False)
        partial_gene_score = cox_summary(merged, "mapped_program_score_recomputed")

    summary = {
        "dataset": "GSE49278",
        "parent_series": "GSE49280",
        "source_paper": "Assie et al. 2014, Nat Genet, doi:10.1038/ng.2953",
        "sample_bridge": "GEO !Sample_title ACC labels joined to MOESM14 Fig. 5b Tumor ID",
        "n_geo_samples": int(len(bridge)),
        "n_os_table_rows": int(len(os_table)),
        "n_gsm_acc_matches": int(merged["acc_label"].notna().sum()),
        "n_os_matched": int(merged["os_match"].sum()),
        "n_os_matched_positive_time": int(len(valid)),
        "os_events_matched_positive_time": int(valid["os_event"].sum()),
        "os_table_unmatched_acc_labels": sorted(set(os_table["acc_label"]) - set(bridge["acc_label"])),
        "geo_unmatched_acc_labels": sorted(set(bridge["acc_label"]) - set(os_table["acc_label"])),
        "mapped_program_genes": mapped_genes,
        "mapped_program_gene_count": int(len(mapped_genes)),
        "locked_genes": LOCKED,
        "locked_genes_missing_on_gpl16686": locked_missing,
        "complete_locked_score_validation": len(locked_missing) == 0,
        "partial_program_os_association": partial_summary,
        "recomputed_mapped_program_os_association": partial_gene_score,
        "claim_status": (
            "complete_locked_four_gene_external_OS_validation_allowed"
            if not locked_missing else
            "locked_four_gene_external_OS_validation_blocked; partial_program_OS_association_only"
        ),
        "source_files": {
            "geo_series_matrix_sha256": (ROOT / "manifests" / "GSE49278_series_matrix.sha256").read_text().strip(),
            "os_table_sha256": sha256(AUDIT / "MOESM14_source_fig5.xls"),
            "mapping_audit_sha256": sha256(EXT / "mapping_audit.json"),
        },
    }
    (OUT / "GSE49278_survival_reanalysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
