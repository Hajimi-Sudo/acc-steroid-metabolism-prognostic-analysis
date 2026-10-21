#!/usr/bin/env python3
"""Remote-only sensitivity analysis for the frozen GSE19750 score.

All variants are descriptive robustness checks defined without using external
survival labels for model selection. No variant is promoted to the primary
score.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from remote_gse19750_locked_validation import (
    FROZEN_COEFFICIENTS,
    FROZEN_GENES,
    parse_expression,
    parse_platform,
    parse_survival,
)

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
OUT = ROOT / "results" / "external_validation"


def bootstrap(valid: pd.DataFrame, score_col: str, n_boot: int = 2000) -> dict:
    rng = np.random.RandomState(20261022)
    values = []
    for _ in range(n_boot):
        sample = valid.iloc[rng.randint(0, len(valid), len(valid))]
        if sample["event"].sum() == 0:
            continue
        values.append(float(concordance_index(sample["survival_years"], -sample[score_col], sample["event"])))
    return {
        "n_boot": len(values),
        "median": float(np.median(values)),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    mapping = parse_platform()
    sample_ids, expression = parse_expression()
    survival = parse_survival().set_index("sample_id")
    probe_set = set(expression["probe"].astype(str))
    gene_values = {}
    probe_counts = {}
    for gene in FROZEN_GENES:
        probes = [probe for probe, genes in mapping.items() if gene in genes and probe in probe_set]
        probe_counts[gene] = len(probes)
        values = expression[expression["probe"].isin(probes)].set_index("probe").iloc[:, 0:]
        gene_values[gene] = values.mean(axis=0).reindex(sample_ids)
    genes = pd.DataFrame(gene_values, index=sample_ids)
    z = (genes - genes.mean()) / genes.std(ddof=0).replace(0, np.nan)
    valid = survival.join(z).reset_index()
    valid = valid[valid["is_acc"] & valid["survival_years"].notna() & valid["event"].notna()]
    valid = valid[valid["survival_years"] > 0].copy()

    variants = {}
    variants["frozen_weighted_mean_z"] = sum(FROZEN_COEFFICIENTS[g] * valid[g] for g in FROZEN_GENES)
    variants["equal_weight_mean_z"] = valid[FROZEN_GENES].mean(axis=1)
    variants["rank_mean"] = valid[FROZEN_GENES].rank(axis=0, pct=True).mean(axis=1)
    for omitted in FROZEN_GENES:
        kept = [g for g in FROZEN_GENES if g != omitted]
        variants[f"omit_{omitted}"] = sum(FROZEN_COEFFICIENTS[g] * valid[g] for g in kept)
    rows = []
    for name, values in variants.items():
        frame = valid[["survival_years", "event"]].copy()
        frame["score"] = values
        rows.append({
            "variant": name,
            "n": int(len(frame)),
            "events": int(frame["event"].sum()),
            "cindex": float(concordance_index(frame["survival_years"], -frame["score"], frame["event"])),
            "bootstrap": bootstrap(frame, "score"),
        })
    summary = {
        "cohort": "GSE19750",
        "endpoint": "OS",
        "analysis_n": int(len(valid)),
        "events": int(valid["event"].sum()),
        "probe_counts": probe_counts,
        "normalization": "within-cohort z-score; descriptive platform sensitivity",
        "variants": rows,
        "interpretation": "Primary frozen weighted score remains unchanged; sensitivity variants are not external-label-tuned.",
    }
    (OUT / "GSE19750_locked_signature_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
