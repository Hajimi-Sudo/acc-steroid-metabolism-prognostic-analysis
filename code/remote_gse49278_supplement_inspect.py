#!/usr/bin/env python3
"""Inspect small official Assie et al. 2014 supplementary workbooks remotely."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "results" / "gse49278_survival_audit"
OUT.mkdir(parents=True, exist_ok=True)
UA = "ACC-steroid-metabolism-audit/1.0 (research provenance)"

URLS = {
    "MOESM10_source_fig1.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fng.2953/MediaObjects/41588_2014_BFng2953_MOESM10_ESM.xlsx",
    "MOESM11_source_fig2.xls": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fng.2953/MediaObjects/41588_2014_BFng2953_MOESM11_ESM.xls",
    "MOESM12_source_fig3.xls": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fng.2953/MediaObjects/41588_2014_BFng2953_MOESM12_ESM.xls",
    "MOESM13_source_fig4.xls": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fng.2953/MediaObjects/41588_2014_BFng2953_MOESM13_ESM.xls",
    "MOESM14_source_fig5.xls": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fng.2953/MediaObjects/41588_2014_BFng2953_MOESM14_ESM.xls",
    "MOESM16_supp_table4.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fng.2953/MediaObjects/41588_2014_BFng2953_MOESM16_ESM.xlsx",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, url: str) -> Path:
    path = OUT / name
    if not path.exists() or path.stat().st_size < 100:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def frame_summary(frame: pd.DataFrame) -> dict:
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    text = "\n".join(str(x) for x in frame.astype(object).fillna("").to_numpy().ravel())
    lower = text.lower()
    terms = {
        term: len(re.findall(re.escape(term), lower))
        for term in ["gsm", "acc", "survival", "overall survival", "dead", "alive", "death", "status", "sample", "patient", "weiss"]
    }
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(x) for x in frame.columns.tolist()[:100]],
        "first_rows": frame.head(12).astype(object).where(pd.notna(frame.head(12)), None).to_dict(orient="records"),
        "term_counts": terms,
    }


def inspect(path: Path) -> dict:
    result = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    try:
        sheets = pd.ExcelFile(path).sheet_names
        result["sheets"] = sheets
        result["sheet_summaries"] = {}
        for sheet in sheets:
            frame = pd.read_excel(path, sheet_name=sheet, header=None)
            result["sheet_summaries"][sheet] = frame_summary(frame)
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> None:
    results = []
    for name, url in URLS.items():
        try:
            path = download(name, url)
            result = inspect(path)
            result["url"] = url
            result["http_source"] = "Springer Nature official supplementary asset"
        except Exception as exc:
            result = {"file": name, "url": url, "error": repr(exc)}
        results.append(result)
    output = {
        "source_paper": "Assie et al. Integrated genomic characterization of adrenocortical carcinoma, Nat Genet 2014, doi:10.1038/ng.2953",
        "files": results,
        "interpretation": "pending_sample_level_survival_match_audit",
    }
    (OUT / "supplement_workbook_inspection.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
