#!/usr/bin/env python
"""Remote-only Gate 1 inventory for TCGA-ACC and GSE49278.

This script writes metadata and clinical audit artifacts to the remote project
root. It does not download expression matrices or patient-level files.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "manifests"
LOG = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_json(url: str, params: dict | None = None, timeout: int = 120) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    gdc_base = "https://api.gdc.cancer.gov"
    file_filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": [{"field": "cases.project.project_id", "value": ["TCGA-ACC"]}]},
            {"op": "in", "content": [{"field": "data_type", "value": ["Gene Expression Quantification"]}]},
            {"op": "in", "content": [{"field": "analysis.workflow_type", "value": ["STAR - Counts"]}]},
        ],
    }
    file_fields = ",".join([
        "file_id", "file_name", "file_size", "data_format", "data_type",
        "experimental_strategy", "analysis.workflow_type", "cases.case_id",
        "cases.submitter_id", "cases.samples.sample_type", "cases.samples.submitter_id",
    ])
    file_payload = {
        "filters": file_filters,
        "fields": file_fields,
        "format": "JSON",
        "size": "10000",
    }
    files = post_json(f"{gdc_base}/files", file_payload)
    write_json(OUT / "tcga_acc_star_counts_inventory.json", files)

    case_fields = ",".join([
        "case_id", "submitter_id", "project.project_id", "demographic",
        "diagnoses", "samples.submitter_id", "samples.sample_type",
        "samples.is_ffpe", "samples.portions.analytes.aliquots.submitter_id",
    ])
    case_payload = {
        "filters": {"op": "in", "content": [{"field": "project.project_id", "value": ["TCGA-ACC"]}]},
        "fields": case_fields,
        "format": "JSON",
        "size": "10000",
    }
    cases = post_json(f"{gdc_base}/cases", case_payload)
    write_json(OUT / "tcga_acc_cases_inventory.json", cases)

    project = get_json(f"{gdc_base}/projects/TCGA-ACC", {"format": "json"})
    write_json(OUT / "tcga_acc_project.json", project)

    geo_summary = get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        {"db": "gds", "id": "200049278", "retmode": "json"},
    )
    write_json(OUT / "gse49278_summary.json", geo_summary)

    geo_dir = requests.get(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE49nnn/GSE49278/matrix/",
        timeout=120,
    )
    geo_dir.raise_for_status()
    (OUT / "gse49278_matrix_directory.html").write_text(geo_dir.text, encoding="utf-8")

    env = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    write_json(LOG / "gate1_environment.json", env)

    manifest_lines = []
    for path in sorted(OUT.glob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_lines.append(f"{digest}  {path.name}")
    (OUT / "gate1_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "files_hits": files.get("data", {}).get("pagination", {}),
        "cases_hits": cases.get("data", {}).get("pagination", {}),
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
