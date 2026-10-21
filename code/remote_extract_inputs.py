#!/usr/bin/env python
"""Extract only the frozen ACC inputs on the remote server.

The script does not contact the local workspace and writes all patient-level
tables below the remote project root.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MAN = ROOT / "manifests"
PROC.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
CBIO = "https://www.cbioportal.org/api"


def get(path: str, params: dict | None = None, timeout: int = 180):
    last_error = None
    for attempt in range(1, 6):
        try:
            response = requests.get(CBIO + path, params=params, headers={"Accept": "application/json"}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise last_error


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def extract_geo() -> None:
    source = RAW / "geo" / "GSE49278_series_matrix.txt.gz"
    metadata = {}
    sample_ids = []
    rows = []
    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                metadata["title"] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                values = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                key = values[0].split(":", 1)[0] if values and ":" in values[0] else "characteristics"
                metadata[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in values]
            elif line.startswith("!Sample_platform_id"):
                metadata["platform"] = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                break
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        value_sample_ids = [x.strip().strip('"') for x in header[1:]]
        if sample_ids != value_sample_ids:
            raise ValueError("GEO sample IDs differ between metadata and expression table")
        for record in reader:
            if not record or record[0].startswith("!series_matrix_table_end"):
                break
            probe = record[0].strip().strip('"')
            try:
                values = [float(x) if x not in ("", "null", "NA", "NaN") else np.nan for x in record[1:]]
            except ValueError:
                continue
            rows.append([probe, *values])
    expr = pd.DataFrame(rows, columns=["probe", *sample_ids])
    expr.to_csv(PROC / "GSE49278_probe_expression.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame({"sample_id": sample_ids, **{k: v for k, v in metadata.items()}}).to_csv(
        PROC / "GSE49278_sample_metadata.tsv", sep="\t", index=False
    )
    write_json(MAN / "gse49278_parse_audit.json", {
        "n_samples": len(sample_ids), "n_probe_rows": len(expr), "platform": sorted(set(metadata.get("platform", []))),
        "sample_titles": metadata.get("title", []), "source": str(source),
    })


def extract_cbio() -> None:
    sample_ids = get("/sample-lists/acc_tcga_rna_seq_v2_mrna/sample-ids")
    patient_clinical = get("/studies/acc_tcga/clinical-data", {"clinicalDataType": "PATIENT", "projection": "DETAILED", "pageSize": 100000})
    sample_clinical = get("/studies/acc_tcga/clinical-data", {"clinicalDataType": "SAMPLE", "projection": "DETAILED", "pageSize": 100000})
    all_clinical = patient_clinical + sample_clinical
    try:
        attrs = get("/studies/acc_tcga/clinical-attributes", {"projection": "DETAILED"})
    except requests.RequestException:
        attrs = []
    write_json(MAN / "cbioportal_clinical_attributes.json", attrs)
    target_ids = {
        "OS_MONTHS", "OS_STATUS", "DSS_MONTHS", "DSS_STATUS", "DFS_MONTHS", "DFS_STATUS",
        "DAYS_LAST_FOLLOWUP", "WEISS_SCORE_OVERALL", "WEISS_SCORE", "WEISS", "HISTORY_ADRENAL_HORMONE_EXCESS",
        "C1A_C1B", "MOLECULAR_SUBTYPE", "SOMATIC_STATUS", "AJCC_PATHOLOGIC_TUMOR_STAGE",
    }
    selected = [row for row in all_clinical if row.get("clinicalAttributeId") in target_ids]
    clinical = defaultdict(dict)
    for row in selected:
        key = row.get("patientId") or row.get("sampleId")
        clinical[key][row["clinicalAttributeId"]] = row.get("value")
    clinical_rows = []
    for patient_id, values in sorted(clinical.items()):
        clinical_rows.append({"patient_id": patient_id, **values})
    pd.DataFrame(clinical_rows).to_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t", index=False)

    def fetch_gene(symbol: str) -> dict | None:
        gene = get(f"/genes/{symbol}")
        entrez = gene.get("entrezGeneId") if isinstance(gene, dict) else None
        if not entrez:
            return None
        values = get(
            "/molecular-profiles/acc_tcga_rna_seq_v2_mrna/molecular-data",
            {"sampleListId": "acc_tcga_rna_seq_v2_mrna", "entrezGeneId": entrez, "projection": "DETAILED"},
        )
        by_sample = {row["sampleId"]: row.get("value") for row in values}
        return {"symbol": symbol, "entrez_gene_id": entrez, "values": by_sample}

    gene_rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_gene, symbol): symbol for symbol in GENES}
        for future in as_completed(futures):
            symbol = futures[future]
            row = future.result()
            if row:
                gene_rows.append(row)
            print(f"gene {symbol} complete", flush=True)
    gene_rows.sort(key=lambda row: GENES.index(row["symbol"]))
    matrix = pd.DataFrame({row["symbol"]: [row["values"].get(sample) for sample in sample_ids] for row in gene_rows}, index=sample_ids)
    matrix.index.name = "sample_id"
    matrix.to_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t", compression="gzip")
    write_json(MAN / "tcga_acc_cbio_audit.json", {
        "n_expression_samples": len(sample_ids), "n_patient_clinical_rows": len(patient_clinical),
        "n_sample_clinical_rows": len(sample_clinical), "n_clinical_rows": len(all_clinical),
        "n_selected_clinical_rows": len(selected), "n_selected_patients": len(clinical_rows),
        "genes_requested": GENES, "genes_retrieved": [row["symbol"] for row in gene_rows],
        "study": "acc_tcga", "expression_profile": "acc_tcga_rna_seq_v2_mrna",
        "source_version": "cBioPortal TCGA Firehose Legacy",
    })


def main() -> None:
    extract_geo()
    extract_cbio()
    print("remote input extraction complete")


if __name__ == "__main__":
    main()
