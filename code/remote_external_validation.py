#!/usr/bin/env python
"""Remote-only GPL16686 probe mapping and GSE49278 score validation."""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "geo"
OUT = ROOT / "results" / "external_validation"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]


def refseqs(symbol: str) -> set[str]:
    response = requests.get(
        "https://mygene.info/v3/query",
        params={"q": symbol, "species": "human", "fields": "symbol,refseq.rna,entrezgene", "size": 5},
        timeout=60,
    )
    response.raise_for_status()
    hits = response.json().get("hits", [])
    for hit in hits:
        if hit.get("symbol") != symbol:
            continue
        values = hit.get("refseq", {}).get("rna", [])
        if isinstance(values, str):
            values = [values]
        return {str(value).split(".", 1)[0] for value in values}
    return set()


def parse_platform() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    root = ET.parse(RAW / "GPL16686_full.xml").getroot()
    namespace = {"g": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
    table = root.find(".//g:Internal-Data", namespace)
    if table is None or not table.text:
        raise RuntimeError("GPL16686 annotation table missing")
    mapping: dict[str, set[str]] = {}
    refseq_by_gene = {gene: refseqs(gene) for gene in GENES}
    accession_to_gene = {
        accession: gene
        for gene, accessions in refseq_by_gene.items()
        for accession in accessions
    }
    for line in table.text.splitlines():
        fields = line.split("\t")
        if len(fields) < 6:
            continue
        probe = fields[0].strip()
        accessions = {
            re.split(r"[. ]", token.strip())[0]
            for token in re.split(r"[;,| ]+", fields[5])
            if token.strip()
        }
        for accession in accessions:
            gene = accession_to_gene.get(accession)
            if gene:
                mapping.setdefault(probe, set()).add(gene)

    # Keep only identifiers from the GPL16686 table. GPL25483/GPL27949 are
    # official alternative CDFs, but their first-column identifiers are not
    # GPL16686 transcript-cluster IDs and cannot be joined to this matrix.
    write_mapping = pd.DataFrame(
        [(probe, gene) for probe, genes in mapping.items() for gene in sorted(genes)],
        columns=["probe", "gene"],
    )
    write_mapping.to_csv(OUT / "GPL16686_refseq_probe_mapping.tsv", sep="\t", index=False)
    return mapping, refseq_by_gene


def parse_geo_metadata() -> pd.DataFrame:
    return pd.read_csv(PROC / "GSE49278_sample_metadata.tsv", sep="\t")


def score_geo(mapping: dict[str, set[str]]) -> pd.DataFrame:
    expression = pd.read_csv(PROC / "GSE49278_probe_expression.tsv.gz", sep="\t")
    sample_ids = list(expression.columns[1:])
    result = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    per_gene = {}
    for gene in GENES:
        probes = [probe for probe, genes in mapping.items() if gene in genes and probe in set(expression["probe"].astype(str))]
        if not probes:
            continue
        values = expression[expression["probe"].astype(str).isin(probes)].set_index("probe").iloc[:, 0:]
        # Probe rows are already log2-scale; use the mean when multiple probes map.
        per_gene[gene] = values.mean(axis=0).reindex(result.index).to_numpy()
    gene_df = pd.DataFrame(per_gene, index=result.index)
    z = (gene_df - gene_df.mean()) / gene_df.std(ddof=0).replace(0, np.nan)
    result = result.join(z.mean(axis=1, skipna=False).rename("steroid_score"))
    result = result.join(gene_df.add_suffix("_expr"))
    result["steroid_score_high"] = (result["steroid_score"] >= result["steroid_score"].median()).astype(int)
    result.reset_index().to_csv(OUT / "GSE49278_steroid_scores.tsv", sep="\t", index=False)
    return result


def main() -> None:
    mapping, refseq_by_gene = parse_platform()
    scores = score_geo(mapping)
    coverage = {gene: sum(gene in genes for genes in mapping.values()) for gene in GENES}
    (OUT / "mapping_audit.json").write_text(json.dumps({
        "genes_requested": GENES,
        "genes_with_refseq": {gene: sorted(values) for gene, values in refseq_by_gene.items()},
        "probe_count": len(mapping),
        "gene_probe_coverage": coverage,
        "n_score_nonmissing": int(scores["steroid_score"].notna().sum()),
        "mapping_basis": "GPL16686 native probe IDs matched to RefSeq accessions",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"mapping_probes": len(mapping), "coverage": coverage, "score_nonmissing": int(scores["steroid_score"].notna().sum())}, indent=2))


if __name__ == "__main__":
    main()
