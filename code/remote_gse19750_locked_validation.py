#!/usr/bin/env python3
"""Apply the frozen TCGA steroid score to GSE19750 on the remote server.

The script never selects genes or tunes parameters on GSE19750. It downloads
only the official GPL570 annotation if it is absent, maps probes to the four
frozen genes, computes a within-cohort standardized score, and reports aggregate
transport statistics with bootstrap uncertainty.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RAW = ROOT / "data" / "raw" / "geo_gse19750"
OUT = ROOT / "results" / "external_validation"
MAN = ROOT / "manifests"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_GENES = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]
FROZEN_COEFFICIENTS = {
    "STAR": -0.315164785566944,
    "NR0B1": -0.14768321386120697,
    "HSD11B2": 0.489792020819889,
    "AKR1C3": -0.16957743987510482,
}
GPL_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
GPL_PATH = RAW / "GPL570.annot.gz"


def download_platform() -> None:
    if GPL_PATH.exists() and GPL_PATH.stat().st_size > 100_000:
        return
    with urllib.request.urlopen(GPL_URL, timeout=120) as response, GPL_PATH.open("wb") as handle:
        handle.write(response.read())


def parse_platform() -> dict[str, set[str]]:
    """Return Affymetrix probe -> gene symbol mappings from NCBI GPL570."""
    mapping: dict[str, set[str]] = {}
    with gzip.open(GPL_PATH, "rt", encoding="utf-8", errors="replace") as handle:
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


def parse_expression() -> tuple[list[str], pd.DataFrame]:
    source = RAW / "GSE19750_series_matrix.txt.gz"
    rows: list[list[float | str]] = []
    sample_ids: list[str] = []
    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!series_matrix_table_begin"):
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                matrix_ids = [x.strip().strip('"') for x in header[1:]]
                if sample_ids != matrix_ids:
                    raise RuntimeError("GSE19750 sample IDs differ between metadata and matrix")
                for record in reader:
                    if not record or record[0].startswith("!series_matrix_table_end"):
                        break
                    try:
                        values = [float(x) if x not in {"", "NA", "null", "NaN"} else np.nan for x in record[1:]]
                    except ValueError:
                        continue
                    rows.append([record[0].strip().strip('"'), *values])
                break
    return sample_ids, pd.DataFrame(rows, columns=["probe", *sample_ids])


def parse_survival() -> pd.DataFrame:
    rows: list[tuple[str, list[str]]] = []
    with gzip.open(RAW / "GSE19750_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_"):
                continue
            fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            rows.append((fields[0], [x.strip().strip('"') for x in fields[1:]]))

    def one(key: str) -> list[str]:
        return next(values for name, values in rows if name == key)

    sample_ids = one("!Sample_geo_accession")
    titles = one("!Sample_title")
    source = one("!Sample_source_name_ch1")
    survival = next(values for name, values in rows if "survival in years:" in " ".join(values).lower())
    status = next(values for name, values in rows if "survival status:" in " ".join(values).lower())
    records = []
    for sample_id, title, src, surv, stat in zip(sample_ids, titles, source, survival, status):
        def value(text: str) -> str:
            return text.split(":", 1)[1].strip() if ":" in text else text.strip()
        records.append({
            "sample_id": sample_id,
            "title": title,
            "source": src,
            "survival_years": pd.to_numeric(value(surv), errors="coerce"),
            "event": {"dead": 1.0, "alive": 0.0}.get(value(stat).lower(), np.nan),
            "is_acc": "adrenocortical carcinoma" in title.lower(),
        })
    return pd.DataFrame(records)


def bootstrap_cindex(valid: pd.DataFrame, n_boot: int = 5000) -> dict:
    rng = np.random.RandomState(20261021)
    values = []
    for _ in range(n_boot):
        sample = valid.iloc[rng.randint(0, len(valid), len(valid))]
        if sample["event"].sum() == 0:
            continue
        values.append(float(concordance_index(sample["survival_years"], -sample["risk"], sample["event"])))
    if not values:
        return {"n_boot": 0}
    return {
        "n_boot": len(values),
        "median": float(np.median(values)),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    download_platform()
    mapping = parse_platform()
    sample_ids, expression = parse_expression()
    survival = parse_survival()
    gene_values: dict[str, pd.Series] = {}
    probe_set = set(expression["probe"].astype(str))
    for gene in FROZEN_GENES:
        probes = [probe for probe, genes in mapping.items() if gene in genes and probe in probe_set]
        if not probes:
            continue
        values = expression[expression["probe"].isin(probes)].set_index("probe").iloc[:, 0:]
        gene_values[gene] = values.mean(axis=0).reindex(sample_ids)
    gene_df = pd.DataFrame(gene_values, index=sample_ids)
    z = (gene_df - gene_df.mean()) / gene_df.std(ddof=0).replace(0, np.nan)
    score = sum(FROZEN_COEFFICIENTS[g] * z[g] for g in FROZEN_GENES if g in z.columns)
    score = score.rename("risk")
    result = survival.set_index("sample_id").join(score).reset_index()
    valid = result[result["is_acc"] & result["survival_years"].notna() & result["event"].notna() & result["risk"].notna()].copy()
    valid = valid[valid["survival_years"] > 0]
    if len(valid) < 5 or valid["event"].sum() < 2:
        raise RuntimeError(f"insufficient external OS cases: n={len(valid)} events={valid['event'].sum()}")
    cindex = float(concordance_index(valid["survival_years"], -valid["risk"], valid["event"]))
    valid["risk_high"] = (valid["risk"] >= valid["risk"].median()).astype(int)
    model = CoxPHFitter(penalizer=0.01)
    model.fit(valid[["survival_years", "event", "risk"]], "survival_years", "event")
    row = model.summary.loc["risk"]
    summary = {
        "cohort": "GSE19750",
        "platform": "GPL570",
        "endpoint": "OS",
        "n_series_samples": int(len(sample_ids)),
        "n_acc_complete_os": int(len(valid)),
        "events": int(valid["event"].sum()),
        "genes_requested": FROZEN_GENES,
        "genes_mapped": sorted(gene_df.columns.tolist()),
        "probe_counts": {g: int(sum(g in genes and p in probe_set for p, genes in mapping.items())) for g in FROZEN_GENES},
        "mapping_basis": "NCBI GPL570 annotation gene symbols",
        "score_definition": "TCGA-locked coefficients applied to within-GSE19750 gene z-scores; no external tuning",
        "cindex": cindex,
        "cindex_bootstrap": bootstrap_cindex(valid),
        "cox_hr_per_score_unit": float(np.exp(row["coef"])),
        "cox_p": float(row["p"]),
        "cox_ci_low": float(np.exp(row["coef lower 95%"])),
        "cox_ci_high": float(np.exp(row["coef upper 95%"])),
        "median_split_n_high": int(valid["risk_high"].sum()),
        "median_split_n_low": int((valid["risk_high"] == 0).sum()),
        "claim_status": "external_adult_OS_test; small sample and annotation-derived endpoints; not definitive validation",
    }
    (OUT / "GSE19750_locked_signature_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    valid[["sample_id", "survival_years", "event", "risk", "risk_high"]].to_csv(OUT / "GSE19750_locked_signature_scores.tsv", sep="\t", index=False)
    manifest = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform_sha256": hashlib.sha256(GPL_PATH.read_bytes()).hexdigest(),
        "summary": summary,
    }
    (MAN / "gse19750_locked_signature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
