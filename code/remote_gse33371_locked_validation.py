#!/usr/bin/env python3
"""Audit and test the frozen four-gene score in GSE33371 on the server.

GSE33371 is explicitly treated as the same expression cohort as GSE10927,
with additional follow-up annotations. It must not be counted as independent
from GSE10927. No feature selection, coefficient tuning, or threshold tuning
uses the GSE33371 outcomes.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RAW = ROOT / "data" / "raw" / "geo_gse33371"
RAW_19750 = ROOT / "data" / "raw" / "geo_gse19750"
OUT = ROOT / "results" / "external_validation"
MAN = ROOT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

FROZEN_GENES = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]
FROZEN_COEFFICIENTS = {
    "STAR": -0.315164785566944,
    "NR0B1": -0.14768321386120697,
    "HSD11B2": 0.489792020819889,
    "AKR1C3": -0.16957743987510482,
}


def parse_platform() -> dict[str, set[str]]:
    path = RAW_19750 / "GPL570.annot.gz"
    mapping: dict[str, set[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = None
        for line in handle:
            if line.startswith("!platform_table_begin"):
                header = None
                continue
            if line.startswith("!platform_table_end"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = [x.strip().lower() for x in fields]
                continue
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            probe = row.get("id", "").strip()
            symbols = row.get("gene symbol", "").strip()
            if not probe or not symbols or symbols in {"---", "NA", "N/A"}:
                continue
            for symbol in re.split(r"[ ///;,]+", symbols.upper()):
                if symbol in FROZEN_GENES:
                    mapping.setdefault(probe, set()).add(symbol)
    if not mapping:
        raise RuntimeError("GPL570 annotation did not yield frozen gene mappings")
    return mapping


def parse_matrix(path: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    metadata: dict[str, list[str]] = {}
    rows: list[list[float | str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_"):
                fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
                metadata[fields[0]] = [x.strip().strip('"') for x in fields[1:]]
            if line.startswith("!series_matrix_table_begin"):
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                sample_ids = [x.strip().strip('"') for x in header[1:]]
                for record in reader:
                    if not record or record[0].startswith("!series_matrix_table_end"):
                        break
                    try:
                        values = [float(x) if x not in {"", "NA", "null", "NaN"} else np.nan for x in record[1:]]
                    except ValueError:
                        continue
                    rows.append([record[0].strip().strip('"'), *values])
                break
    expression = pd.DataFrame(rows, columns=["probe", *sample_ids])
    return sample_ids, metadata, expression


def normalize_acc_label(title: str) -> str | None:
    """Normalize GEO paper labels for a conservative cross-series audit."""
    match = re.search(r"ACC[- ]?0*(\d+)", title, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"Carcinoma[ ,]+0*(\d+)", title, flags=re.IGNORECASE)
    return f"ACC-{int(match.group(1))}" if match else None


def compare_expression_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_sample_ids: list[str],
    right_sample_ids: list[str],
) -> dict:
    """Compare matrices after aligning probes, without changing any score."""
    left_index = left.set_index("probe")
    right_index = right.set_index("probe")
    common_probes = left_index.index.intersection(right_index.index)
    if not len(common_probes):
        return {"common_probes": 0, "same_expression_values": False}
    left_values = left_index.loc[common_probes, left_sample_ids].to_numpy(dtype=float)
    right_values = right_index.loc[common_probes, right_sample_ids].to_numpy(dtype=float)
    if left_values.shape != right_values.shape:
        return {
            "common_probes": int(len(common_probes)),
            "left_shape": list(left_values.shape),
            "right_shape": list(right_values.shape),
            "same_expression_values": False,
        }
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    differences = np.abs(left_values[finite] - right_values[finite])
    return {
        "common_probes": int(len(common_probes)),
        "left_shape": list(left_values.shape),
        "right_shape": list(right_values.shape),
        "finite_values_compared": int(finite.sum()),
        "max_abs_difference": float(differences.max()) if len(differences) else None,
        "same_expression_values": bool(np.array_equal(left_values, right_values, equal_nan=True)),
    }


def compare_by_normalized_titles(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_titles: list[str],
    right_titles: list[str],
) -> dict:
    """Compare corresponding biological samples after title normalization."""
    def normalize_title(title: str) -> str:
        title = re.sub(r"\[(?:nc|aca|acc|affymetrix)\]", "", title, flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", "", title.lower())

    left_map = {normalize_title(title): index for index, title in enumerate(left_titles)}
    right_map = {normalize_title(title): index for index, title in enumerate(right_titles)}
    common = sorted(set(left_map) & set(right_map))
    left_index = left.set_index("probe")
    right_index = right.set_index("probe")
    common_probes = left_index.index.intersection(right_index.index)
    rows = []
    for key in common:
        left_position = left_map[key]
        right_position = right_map[key]
        left_values = left_index.loc[common_probes, left.columns[left_position + 1]].to_numpy(dtype=float)
        right_values = right_index.loc[common_probes, right.columns[right_position + 1]].to_numpy(dtype=float)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        correlation = float(np.corrcoef(left_values[finite], right_values[finite])[0, 1]) if finite.sum() > 2 else None
        rmse = float(np.sqrt(np.mean((left_values[finite] - right_values[finite]) ** 2))) if finite.any() else None
        rows.append({
            "left_title": left_titles[left_position],
            "right_title": right_titles[right_position],
            "correlation": correlation,
            "rmse": rmse,
            "exact_probe_values": bool(np.array_equal(left_values, right_values, equal_nan=True)),
        })
    return {
        "title_matches": len(common),
        "left_samples": len(left_titles),
        "right_samples": len(right_titles),
        "common_probes": int(len(common_probes)),
        "n_exact_expression_matches": sum(row["exact_probe_values"] for row in rows),
        "median_sample_expression_correlation": float(np.median([row["correlation"] for row in rows])) if rows else None,
        "min_sample_expression_correlation": float(min(row["correlation"] for row in rows)) if rows else None,
        "max_sample_expression_correlation": float(max(row["correlation"] for row in rows)) if rows else None,
        "median_sample_expression_rmse": float(np.median([row["rmse"] for row in rows])) if rows else None,
        "sample_comparisons": rows,
        "interpretation": "same biological sample labels are present in both series; values were compared after title-based alignment",
    }


def audit_gse19750_overlap(
    gse33371_sample_ids: list[str],
    gse33371_metadata: dict[str, list[str]],
    gse33371_expression: pd.DataFrame,
) -> dict:
    other_sample_ids, other_metadata, other_expression = parse_matrix(
        RAW_19750 / "GSE19750_series_matrix.txt.gz"
    )
    left_titles = gse33371_metadata["!Sample_title"]
    right_titles = other_metadata["!Sample_title"]
    left = {
        normalize_acc_label(title): (index, title)
        for index, title in enumerate(left_titles)
        if normalize_acc_label(title) is not None and "carcinoma" in title.lower()
    }
    right = {
        normalize_acc_label(title): (index, title)
        for index, title in enumerate(right_titles)
        if normalize_acc_label(title) is not None and "carcinoma" in title.lower()
    }
    common_labels = sorted(set(left) & set(right))
    comparisons = []
    left_index = gse33371_expression.set_index("probe")
    right_index = other_expression.set_index("probe")
    common_probes = left_index.index.intersection(right_index.index)
    for label in common_labels:
        left_position, left_title = left[label]
        right_position, right_title = right[label]
        left_values = left_index.loc[common_probes, gse33371_sample_ids[left_position]].to_numpy(dtype=float)
        right_values = right_index.loc[common_probes, other_sample_ids[right_position]].to_numpy(dtype=float)
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if finite.sum() > 2:
            correlation = float(np.corrcoef(left_values[finite], right_values[finite])[0, 1])
            rmse = float(np.sqrt(np.mean((left_values[finite] - right_values[finite]) ** 2)))
        else:
            correlation, rmse = None, None
        comparisons.append({
            "label": label,
            "gse33371_title": left_title,
            "gse19750_title": right_title,
            "common_probes": int(len(common_probes)),
            "correlation": correlation,
            "rmse": rmse,
            "exact_probe_values": bool(np.array_equal(left_values, right_values, equal_nan=True)),
        })
    exact_matches = [row for row in comparisons if row["exact_probe_values"]]
    high_concordance = [row for row in comparisons if row["correlation"] is not None and row["correlation"] >= 0.99]
    return {
        "gse33371_acc_labels": sorted(left),
        "gse19750_acc_labels": sorted(right),
        "common_normalized_labels": common_labels,
        "n_common_normalized_labels": len(common_labels),
        "comparisons": comparisons,
        "n_exact_expression_matches": len(exact_matches),
        "n_high_concordance_expression_matches": len(high_concordance),
        "interpretation": (
            "patient-level overlap is supported for labels with near-identical GPL570 profiles"
            if high_concordance else
            "shared labels do not have expression-level confirmation of duplicated samples"
        ),
    }


def parse_endpoint(metadata: dict[str, list[str]], sample_ids: list[str]) -> pd.DataFrame:
    titles = metadata["!Sample_title"]
    characteristics = metadata.get("!Sample_characteristics_ch1", [])
    # GEO stores multiple characteristic rows under the same key in some
    # series; the matrix parser retains only the last row, so parse the source
    # file again for all rows and select the endpoint labels explicitly.
    values_by_key: dict[str, list[list[str]]] = {}
    path = RAW / "GSE33371_series_matrix.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_characteristics_ch1"):
                continue
            fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            values_by_key.setdefault(fields[0], []).append([x.strip().strip('"') for x in fields[1:]])
    rows = []
    for index, sample_id in enumerate(sample_ids):
        title = titles[index]
        record = {"sample_id": sample_id, "title": title, "is_acc": "carcinoma" in title.lower()}
        for row in values_by_key.get("!Sample_characteristics_ch1", []):
            value = row[index] if index < len(row) else ""
            lower = value.lower()
            if "years to last followup:" in lower:
                record["survival_years"] = value.split(":", 1)[1].strip()
            elif "dead or alive at last followup:" in lower:
                record["status"] = value.split(":", 1)[1].strip().lower()
            elif "our sample name" in lower:
                record["paper_sample"] = value
        rows.append(record)
    result = pd.DataFrame(rows)
    result["survival_years"] = pd.to_numeric(result["survival_years"].replace({"unknown": np.nan, "Unknown": np.nan}), errors="coerce")
    result["event"] = result["status"].map({"dead": 1.0, "alive": 0.0})
    result["complete_positive_os"] = result["is_acc"] & result["survival_years"].gt(0) & result["event"].notna()
    return result


def score(expression: pd.DataFrame, mapping: dict[str, set[str]], sample_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    probe_set = set(expression["probe"].astype(str))
    gene_values: dict[str, pd.Series] = {}
    probe_counts = {}
    for gene in FROZEN_GENES:
        probes = [probe for probe, genes in mapping.items() if gene in genes and probe in probe_set]
        probe_counts[gene] = len(probes)
        if probes:
            values = expression[expression["probe"].isin(probes)].set_index("probe").iloc[:, 0:]
            gene_values[gene] = values.mean(axis=0).reindex(sample_ids)
    gene_df = pd.DataFrame(gene_values, index=sample_ids)
    missing = sorted(set(FROZEN_GENES) - set(gene_df.columns))
    if missing:
        raise RuntimeError(f"missing frozen genes on GPL570: {missing}")
    z = (gene_df - gene_df.mean()) / gene_df.std(ddof=0).replace(0, np.nan)
    risk = sum(FROZEN_COEFFICIENTS[g] * z[g] for g in FROZEN_GENES).rename("risk")
    return pd.DataFrame(gene_df).join(risk), {"probe_counts": probe_counts, "missing_genes": missing}


def bootstrap_cindex(valid: pd.DataFrame, n_boot: int = 5000) -> dict:
    rng = np.random.RandomState(20261022)
    values = []
    for _ in range(n_boot):
        sample = valid.iloc[rng.randint(0, len(valid), len(valid))]
        if sample["event"].sum() == 0:
            continue
        values.append(float(concordance_index(sample["survival_years"], -sample["risk"], sample["event"])))
    return {
        "n_boot": len(values),
        "median": float(np.median(values)),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    mapping = parse_platform()
    sample_ids, metadata, expression = parse_matrix(RAW / "GSE33371_series_matrix.txt.gz")
    gse10927_sample_ids, gse10927_metadata, gse10927_expression = parse_matrix(
        RAW / "GSE10927_series_matrix.txt.gz"
    )
    gse10927_comparison = compare_expression_matrices(
        expression, gse10927_expression, sample_ids, gse10927_sample_ids
    )
    gse10927_title_comparison = compare_by_normalized_titles(
        expression,
        gse10927_expression,
        metadata["!Sample_title"],
        gse10927_metadata["!Sample_title"],
    )
    gse19750_overlap = audit_gse19750_overlap(sample_ids, metadata, expression)
    endpoint = parse_endpoint(metadata, sample_ids)
    score_df, mapping_info = score(expression, mapping, sample_ids)
    result = endpoint.set_index("sample_id").join(score_df).reset_index()
    valid = result[result["complete_positive_os"]].copy()
    valid["risk_high"] = (valid["risk"] >= valid["risk"].median()).astype(int)
    cindex = float(concordance_index(valid["survival_years"], -valid["risk"], valid["event"]))
    model = CoxPHFitter(penalizer=0.01)
    model.fit(valid[["survival_years", "event", "risk"]], "survival_years", "event")
    row = model.summary.loc["risk"]
    summary = {
        "cohort": "GSE33371",
        "platform": "GPL570",
        "parent_expression_series": "GSE10927",
        "same_expression_as_GSE10927": True,
        "expression_overlap_rule": "GSE33371 and GSE10927 contain the same 65 biological sample labels after title normalization; sample order and processed values differ, so GSE33371 is treated as a reprocessed/annotated companion series rather than an independent cohort",
        "gse10927_expression_comparison": gse10927_comparison,
        "gse10927_title_aligned_comparison": gse10927_title_comparison,
        "n_series_samples": len(sample_ids),
        "n_acc_samples": int(endpoint["is_acc"].sum()),
        "n_acc_complete_positive_os": int(len(valid)),
        "events": int(valid["event"].sum()),
        "genes_requested": FROZEN_GENES,
        "genes_mapped": sorted(score_df.columns.difference(["risk"]).tolist()),
        "probe_counts": mapping_info["probe_counts"],
        "score_definition": "TCGA-locked coefficients applied to within-cohort GPL570 gene z-scores; no external tuning",
        "cindex": cindex,
        "cindex_bootstrap": bootstrap_cindex(valid),
        "cox_hr_per_score_unit": float(np.exp(row["coef"])),
        "cox_p": float(row["p"]),
        "cox_ci_low": float(np.exp(row["coef lower 95%"])),
        "cox_ci_high": float(np.exp(row["coef upper 95%"])),
        "claim_status": "same-expression-cohort follow-up validation; not independent from GSE10927; exploratory external OS test",
    }
    result.to_csv(OUT / "GSE33371_locked_signature_scores.tsv", sep="\t", index=False)
    (OUT / "GSE33371_locked_signature_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    overlap = {
        "GSE33371": {"n_samples": len(sample_ids), "n_acc": int(endpoint["is_acc"].sum()), "sample_ids": sample_ids},
        "GSE10927": {
            "n_samples": len(gse10927_sample_ids),
            "comparison": gse10927_comparison,
            "title_aligned_comparison": gse10927_title_comparison,
            "same_biological_sample_labels": True,
        },
        "GSE19750": {
            "sample_identifier_namespace": "normalized ACC-<number> labels",
            "comparison": gse19750_overlap,
            "overlap_status": gse19750_overlap["interpretation"],
        },
        "GSE49278": {"sample_identifier_namespace": "GEO GSM/ACC labels from a different series", "overlap_status": "no verified patient-level overlap established"},
        "independence_decision": "GSE33371 is not independent from GSE10927; it is a supplementary follow-up test. GSE19750 overlap is reported at the normalized-label and expression-concordance levels rather than counted as a separate independent cohort.",
    }
    (OUT / "GSE33371_overlap_audit.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    manifest = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "GSE33371_matrix_sha256": hashlib.sha256((RAW / "GSE33371_series_matrix.txt.gz").read_bytes()).hexdigest(),
        "GSE10927_matrix_sha256": hashlib.sha256((RAW / "GSE10927_series_matrix.txt.gz").read_bytes()).hexdigest(),
        "supplement_sha256": hashlib.sha256((RAW / "GSE33371_GEO_Adrenal_Bcat_supplement.xls.gz").read_bytes()).hexdigest(),
        "summary": summary,
        "overlap": overlap,
    }
    (MAN / "gse33371_locked_signature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
