#!/usr/bin/env python3
"""Audit official GSE49278 clinical/survival sources on the remote server.

This script downloads only metadata and small supplementary files. It does not
download the 330 MB GEO CEL archive or copy patient-level data locally.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "results" / "gse49278_survival_audit"
MAN = ROOT / "manifests"
OUT.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

UA = "ACC-steroid-metabolism-audit/1.0 (research provenance)"
GEO_XML = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49278&targ=self&form=xml&view=full"
GEO_SUPPL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE49nnn/GSE49278/suppl/"
NATURE = "https://www.nature.com/articles/ng.2953"
EUROPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href") or ""
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            self.links.append((" ".join("".join(self._text).split()), self._href))
            self._href = ""
            self._text = []


def get(url: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": UA}
    headers.update(kwargs.pop("headers", {}))
    last = None
    for attempt in range(1, 5):
        try:
            response = requests.get(url, headers=headers, timeout=90, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 15))
    raise last  # pragma: no cover


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_geo_xml(data: bytes) -> dict:
    root = ET.fromstring(data)
    ns = {"g": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
    series = root.find(".//g:Series", ns)
    samples = root.findall(".//g:Sample", ns)
    series_fields: dict[str, list[str]] = {}
    if series is not None:
        for child in list(series):
            key = child.tag.split("}")[-1]
            if child.text and key in {"Title", "Accession", "Summary", "Overall-Design", "Pubmed-ID", "Supplementary-Data"}:
                series_fields.setdefault(key, []).append(child.text.strip())
    sample_rows = []
    for sample in samples:
        row = {"sample_id": sample.attrib.get("iid")}
        for child in sample.iter():
            key = child.tag.split("}")[-1]
            if child.text and key in {"Title", "Source", "Characteristics", "Accession"}:
                row.setdefault(key, []).append(child.text.strip())
        sample_rows.append(row)
    return {"series": series_fields, "samples": sample_rows}


def parse_europepmc() -> dict:
    response = get(EUROPMC, params={"query": "EXT_ID:24747642", "format": "json"})
    return response.json()


def main() -> None:
    audit: dict = {
        "project": "acc-steroid-metabolism",
        "dataset": "GSE49278",
        "sources": [],
        "claim_status": "pending",
    }

    geo_response = get(GEO_XML)
    (OUT / "GSE49278_geo_full.xml").write_bytes(geo_response.content)
    geo = parse_geo_xml(geo_response.content)
    pd.DataFrame(geo["samples"]).to_json(OUT / "GSE49278_geo_samples.json", orient="records", indent=2)
    audit["sources"].append({"url": GEO_XML, "status": geo_response.status_code, "bytes": len(geo_response.content), "sha256": sha256_bytes(geo_response.content)})
    audit["geo_series"] = geo["series"]
    audit["n_geo_samples"] = len(geo["samples"])

    ftp_response = get(GEO_SUPPL)
    (OUT / "GSE49278_geo_supp_index.html").write_bytes(ftp_response.content)
    parser = LinkParser()
    parser.feed(ftp_response.text)
    ftp_links = [{"text": text, "url": urljoin(GEO_SUPPL, href)} for text, href in parser.links if href and href not in {"../", "/"}]
    audit["geo_supplementary_index"] = ftp_links
    audit["sources"].append({"url": GEO_SUPPL, "status": ftp_response.status_code, "bytes": len(ftp_response.content), "sha256": sha256_bytes(ftp_response.content)})

    nature_response = get(NATURE)
    (OUT / "nature_ng2953.html").write_bytes(nature_response.content)
    parser = LinkParser()
    parser.feed(nature_response.text)
    links = []
    for text, href in parser.links:
        absolute = urljoin(NATURE, href)
        if any(token in (text + " " + absolute).lower() for token in ["supplement", "source data", "moesm", "table s", "extended data"]):
            links.append({"text": text, "url": absolute})
    # Some Nature assets are present only in JSON/HTML attributes, not anchor text.
    raw_candidates = sorted(set(re.findall(r"https?[^\\\"' ]+", nature_response.text)))
    for candidate in raw_candidates:
        if any(token in candidate.lower() for token in ["moesm", "supplement", "source-data", "source_data"]):
            links.append({"text": "raw-html-candidate", "url": candidate.replace("\\u0026", "&")})
    dedup = {(item["text"], item["url"]): item for item in links}
    audit["nature_supplement_candidates"] = list(dedup.values())
    audit["sources"].append({"url": NATURE, "status": nature_response.status_code, "bytes": len(nature_response.content), "sha256": sha256_bytes(nature_response.content)})

    europepmc = parse_europepmc()
    (OUT / "europepmc_assie_2014.json").write_text(json.dumps(europepmc, indent=2), encoding="utf-8")
    audit["europepmc"] = europepmc.get("resultList", {}).get("result", [])

    audit["sample_id_patterns"] = {
        "geo_accessions": sorted({row.get("sample_id") for row in geo["samples"]}),
        "sample_titles": sorted({x for row in geo["samples"] for x in row.get("Title", [])}),
        "characteristic_values": sorted({x for row in geo["samples"] for x in row.get("Characteristics", [])}),
    }
    audit["claim_status"] = "official_GEO_has_expression_and_CEL_only_pending_supplementary_clinical_source_review"
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "n_geo_samples": audit["n_geo_samples"],
        "pubmed_ids": geo["series"].get("Pubmed-ID", []),
        "geo_supplement_files": ftp_links,
        "nature_candidate_count": len(audit["nature_supplement_candidates"]),
        "out": str(OUT / "audit.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
