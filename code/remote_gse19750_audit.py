#!/usr/bin/env python3
"""Audit GSE19750 sample-level survival fields on the remote server."""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RAW = ROOT / "data" / "raw" / "geo_gse19750"
OUT = ROOT / "manifests"


def parse_series() -> pd.DataFrame:
    rows: list[tuple[str, list[str]]] = []
    with gzip.open(RAW / "GSE19750_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_"):
                continue
            fields = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            rows.append((fields[0], [x.strip().strip('"') for x in fields[1:]]))
    def one(key: str) -> list[str]:
        return next(v for k, v in rows if k == key)
    sample_ids = one("!Sample_geo_accession")
    titles = one("!Sample_title")
    source = one("!Sample_source_name_ch1")
    survival = next(v for k, v in rows if "survival in years:" in " ".join(v).lower())
    status = next(v for k, v in rows if "survival status:" in " ".join(v).lower())
    stage = next(v for k, v in rows if "stage:" in " ".join(v).lower())
    grade = next(v for k, v in rows if "tumor grade:" in " ".join(v).lower())
    functional = next(v for k, v in rows if "functional:" in " ".join(v).lower())
    records = []
    for sample_id, title, src, surv, stat, stg, grd, func in zip(sample_ids, titles, source, survival, status, stage, grade, functional):
        def val(x: str) -> str:
            return x.split(":", 1)[1].strip() if ":" in x else x.strip()
        records.append({
            "sample_id": sample_id,
            "title": title,
            "sample_label": re.sub(r".*?ACC[- ]?", "ACC ", title, flags=re.I).replace(" [Affymetrix]", "").strip(),
            "source": src,
            "survival_years_raw": val(surv),
            "survival_status_raw": val(stat).lower(),
            "stage": val(stg),
            "grade": val(grd),
            "functional": val(func),
        })
    return pd.DataFrame(records)


def parse_table2() -> pd.DataFrame:
    text = (OUT / "gse19750_pmc_tables.txt").read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("Sample*\tGender"))
    tokens = [x.strip() for x in line.split("\t") if x.strip()]
    header = ["sample_label", "gender", "age_years", "survival_years", "tumor_size_cm", "tumor_weight_g", "stage", "grade"]
    rows = []
    for i in range(0, len(tokens), 8):
        chunk = tokens[i:i + 8]
        if len(chunk) == 8 and chunk[0].startswith(("ACC", "MPI")):
            rows.append(dict(zip(header, chunk)))
    return pd.DataFrame(rows)


def main() -> None:
    geo = parse_series()
    table = parse_table2()
    geo["sample_label"] = geo["sample_label"].str.replace(r"\*", "", regex=True).str.strip()
    geo["is_acc"] = geo["title"].str.contains("Adrenocortical Carcinoma", case=False, na=False)
    geo["survival_years"] = pd.to_numeric(geo["survival_years_raw"].replace({"NA": None, "Unknown": None, "unknown": None, "": None}), errors="coerce")
    geo["event"] = geo["survival_status_raw"].map({"dead": 1, "alive": 0})
    geo["complete_os"] = geo["is_acc"] & geo["survival_years"].notna() & geo["event"].notna()
    geo_labels = set(geo.loc[geo["is_acc"], "sample_label"])
    table["sample_label"] = table["sample_label"].str.replace(r"\*", "", regex=True).str.strip()
    table_labels = set(table["sample_label"])
    result = {
        "n_series_samples": int(len(geo)),
        "n_acc_samples": int(geo["is_acc"].sum()),
        "n_acc_complete_os": int(geo["complete_os"].sum()),
        "n_acc_events": int(geo.loc[geo["complete_os"], "event"].sum()),
        "n_acc_censored": int((geo.loc[geo["complete_os"], "event"] == 0).sum()),
        "n_paper_table_rows": int(len(table)),
        "n_labels_in_both": int(len(geo_labels & table_labels)),
        "geo_only_labels": sorted(geo_labels - table_labels),
        "paper_only_labels": sorted(table_labels - geo_labels),
        "duplicate_or_related_labels": sorted([x for x in geo_labels if x in {"ACC 149", "ACC 150"} or "MPI" in x]),
        "platform": "GPL570",
        "claim_status": "candidate_external_os_validation_pending_probe_mapping_and_overlap_audit",
    }
    geo.to_csv(OUT / "GSE19750_sample_survival_audit.tsv", sep="\t", index=False)
    table.to_csv(OUT / "GSE19750_paper_table2_survival.tsv", sep="\t", index=False)
    (OUT / "GSE19750_survival_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
