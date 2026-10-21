#!/usr/bin/env python
"""Remote-only concurrent acquisition for the frozen ACC cohort contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "manifests"
LOGS = ROOT / "logs"
GDC = "https://api.gdc.cancer.gov"


def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def query_files() -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": [{"field": "cases.project.project_id", "value": ["TCGA-ACC"]}]},
            {"op": "in", "content": [{"field": "data_type", "value": ["Gene Expression Quantification"]}]},
            {"op": "in", "content": [{"field": "analysis.workflow_type", "value": ["STAR - Counts"]}]},
            {"op": "in", "content": [{"field": "access", "value": ["open"]}]},
        ],
    }
    payload = {
        "filters": filters,
        "fields": ",".join([
            "file_id", "file_name", "file_size", "data_format", "data_type",
            "experimental_strategy", "analysis.workflow_type", "access",
            "cases.case_id", "cases.submitter_id", "cases.samples.submitter_id",
            "cases.samples.sample_type",
        ]),
        "format": "JSON",
        "size": "10000",
    }
    data = post_json(f"{GDC}/files", payload)
    records = data.get("data", {}).get("hits", [])
    (MANIFESTS / "tcga_acc_star_counts_inventory.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    return records


def query_cases() -> dict:
    payload = {
        "filters": {"op": "in", "content": [{"field": "project.project_id", "value": ["TCGA-ACC"]}]},
        "fields": ",".join([
            "case_id", "submitter_id", "project.project_id", "demographic",
            "diagnoses", "samples.submitter_id", "samples.sample_type",
            "samples.is_ffpe",
        ]),
        "format": "JSON",
        "size": "10000",
    }
    data = post_json(f"{GDC}/cases", payload)
    (MANIFESTS / "tcga_acc_cases_inventory.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    return data


def download_one(rec: dict, retries: int = 4) -> dict:
    file_id = rec["file_id"]
    filename = Path(rec.get("file_name") or f"{file_id}.dat").name
    dest = RAW / "tcga_star_counts" / f"{file_id}__{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(rec.get("file_size") or 0)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            existing = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing and expected and existing < expected else {}
            if expected and existing > expected:
                dest.unlink()
                existing = 0
                headers = {}
            with requests.get(f"{GDC}/data/{file_id}", headers=headers, stream=True, timeout=(30, 300)) as response:
                if response.status_code == 416 and expected and existing == expected:
                    pass
                else:
                    response.raise_for_status()
                    mode = "ab" if existing and response.status_code == 206 else "wb"
                    with dest.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
            size = dest.stat().st_size
            if expected and size != expected:
                raise IOError(f"size mismatch {size} != {expected}")
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            return {"file_id": file_id, "file_name": filename, "path": str(dest), "bytes": size, "sha256": digest, "status": "ok", "attempts": attempt}
        except Exception as exc:  # retry transient API/network failures
            last_error = repr(exc)
            time.sleep(min(2 ** attempt, 20))
    return {"file_id": file_id, "file_name": filename, "path": str(dest), "status": "failed", "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    for directory in (RAW, MANIFESTS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    records = query_files()
    query_cases()
    print(f"GDC open STAR-count files: {len(records)}", flush=True)
    if args.inventory_only:
        return
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download_one, rec) for rec in records]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{index}/{len(futures)}] {result.get('status')} {result.get('file_name')} {result.get('bytes', 0)}", flush=True)
    results.sort(key=lambda item: item["file_id"])
    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "n_requested": len(records),
        "n_ok": sum(item.get("status") == "ok" for item in results),
        "n_failed": sum(item.get("status") != "ok" for item in results),
        "files": results,
    }
    (MANIFESTS / "tcga_star_counts_download.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["n_failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
