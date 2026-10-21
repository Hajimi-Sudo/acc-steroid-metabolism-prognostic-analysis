#!/usr/bin/env python3
"""Remote-only candidate target lookup; hypothesis generation, not efficacy evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "results" / "druggability"
OUT.mkdir(parents=True, exist_ok=True)
GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]


def mygene(symbol: str) -> dict:
    response = requests.get(
        "https://mygene.info/v3/query",
        params={"q": symbol, "species": "human", "fields": "symbol,ensembl.gene", "size": 10},
        timeout=60,
    )
    response.raise_for_status()
    hits = response.json().get("hits", [])
    for hit in hits:
        if hit.get("symbol") != symbol:
            continue
        ens = hit.get("ensembl", {})
        if isinstance(ens, dict):
            ens = [ens]
        ids = sorted({item.get("gene") for item in ens if item.get("gene")})
        return {"symbol": symbol, "ensembl_ids": ids, "mygene_status": "matched"}
    return {"symbol": symbol, "ensembl_ids": [], "mygene_status": "not_found"}


def opentarget_search(symbol: str) -> dict:
    query = "query Search($q: String!) { search(queryString: $q) { hits { id entity } } }"
    response = requests.post(
        "https://api.platform.opentargets.org/api/v4/graphql",
        json={"query": query, "variables": {"q": symbol}},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("data", {}).get("search", {}).get("hits", [])
    target_ids = [item["id"] for item in hits if item.get("entity") == "target" and item.get("id", "").startswith("ENSG")]
    return {"opentargets_target_ids": target_ids[:10], "opentargets_status": "search_hit" if target_ids else "no_target_hit"}


def main() -> None:
    summary = json.loads((ROOT / "results" / "signature_interactions" / "signature_interaction_summary.json").read_text())
    freq = summary.get("selection_frequency_oof", {})
    univ = pd.read_csv(ROOT / "results" / "first_wave" / "gene_univariate_results.tsv", sep="\t")
    os_rows = univ[univ["endpoint"] == "os"].set_index("gene").to_dict("index")
    dfs_rows = univ[univ["endpoint"] == "dfs"].set_index("gene").to_dict("index")
    rows = []
    for gene in GENES:
        item = {"gene": gene, "oof_selection_count": int(freq.get(gene, 0)), "selected_locked": gene in summary.get("selected_genes", [])}
        item.update({"os_hr": os_rows.get(gene, {}).get("hr"), "os_p": os_rows.get(gene, {}).get("p"), "dfs_hr": dfs_rows.get(gene, {}).get("hr"), "dfs_p": dfs_rows.get(gene, {}).get("p")})
        try:
            item.update(mygene(gene))
            item["opentargets_target_ids"] = []
            item["opentargets_status"] = "not_queried_no_ensembl" if not item["ensembl_ids"] else "pending"
            for target_id in item["ensembl_ids"][:1]:
                item.update(opentarget_search(gene))
                break
        except Exception as exc:
            item.update({"ensembl_ids": [], "mygene_status": "error", "opentargets_target_ids": [], "opentargets_status": "error", "error": repr(exc)})
        item["claim_ceiling"] = "candidate prioritization only; no drug sensitivity, efficacy, mechanism, or clinical utility"
        rows.append(item)
    result = pd.DataFrame(rows)
    result["priority_score"] = result["oof_selection_count"].fillna(0) + (result["selected_locked"].astype(int) * 10) + ((-result["os_p"].fillna(1).clip(lower=1e-300).map(lambda x: __import__("math").log10(x))).clip(lower=0))
    result.sort_values(["priority_score", "oof_selection_count"], ascending=False).to_csv(OUT / "candidate_target_prioritization.tsv", sep="\t", index=False)
    (OUT / "candidate_target_prioritization.json").write_text(json.dumps({"n_genes": len(result), "sources": ["MyGene.info", "Open Targets Platform search"], "claim_ceiling": "hypothesis generation only", "rows": rows}, indent=2), encoding="utf-8")
    print(result[["gene", "selected_locked", "oof_selection_count", "os_p", "ensembl_ids", "opentargets_target_ids", "opentargets_status"]].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
