#!/usr/bin/env python3
"""Generate aggregate manuscript figures on the remote server."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    robustness = json.loads((RES / "robustness" / "leakage_free_robustness_summary.json").read_text())
    external = json.loads((RES / "external_validation" / "GSE19750_locked_signature_validation.json").read_text())
    mapping = json.loads((RES / "external_validation" / "mapping_audit.json").read_text())
    interactions = json.loads((RES / "signature_interactions" / "signature_interaction_summary.json").read_text())

    # Figure 1: cohort and endpoint roles.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    boxes = [
        (0.05, 0.58, 0.26, 0.22, "TCGA-ACC\nRNA-seq\n79 cases / 28 OS events\nDiscovery + internal validation", "#d9edf7"),
        (0.37, 0.58, 0.26, 0.22, "GSE19750\nGPL570\n21 positive-time OS cases / 17 events\nSmall external OS test", "#e8f5e9"),
        (0.69, 0.58, 0.26, 0.22, "GSE49278\nGPL16686\n44 ACC expression samples\nExpression-only transportability", "#fff3cd"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#3c4856", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    ax.annotate("prespecified 22-gene steroid program\nand four-gene score", xy=(0.33, 0.58), xytext=(0.33, 0.38), ha="center", arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.annotate("locked score; no external-label tuning", xy=(0.63, 0.58), xytext=(0.63, 0.38), ha="center", arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.5, 0.11, "C1A/C1B sample-level labels were not verified and were excluded from interaction analysis.", ha="center", va="center", fontsize=9, color="#8a1c1c")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "Figure1_cohort_roles")

    # Figure 2: state structure and stability.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.5))
    states = ["Lower-score\nstate", "Higher-score\nstate"]
    n = [23, 56]
    events = [3, 25]
    x = np.arange(2)
    ax1.bar(x, n, color=["#9ecae1", "#de2d26"], width=0.62)
    for i, (ni, ei) in enumerate(zip(n, events)):
        ax1.text(i, ni + 1.5, f"n={ni}\n{ei} events", ha="center", fontsize=9)
    ax1.set_xticks(x, states); ax1.set_ylabel("TCGA-ACC cases")
    ax1.set_title("Steroid-program states")
    ax2.bar(["k=2 stability"], [1.0], color="#31a354", width=0.45)
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("Adjusted Rand index")
    ax2.set_title("100 random initializations")
    ax2.text(0, 1.02, "median = 1.00", ha="center", va="bottom")
    save(fig, "Figure2_steroid_state_stability")

    # Figure 3: internal and external discrimination.
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = ["TCGA OOF\ninternal", "GSE19750\nadult OS test"]
    estimates = [robustness["leakage_free_repeated_nested_like_oof"]["cindex"], external["cindex"]]
    low = [np.nan, external["cindex_bootstrap"]["low_95"]]
    high = [np.nan, external["cindex_bootstrap"]["high_95"]]
    ax.errorbar([1], [estimates[1]], yerr=[[estimates[1] - low[1]], [high[1] - estimates[1]]], fmt="o", color="#238b45", capsize=5, label="Bootstrap 95% interval")
    ax.scatter([0], [estimates[0]], color="#2171b5", s=55, label="Point estimate")
    ax.axhline(0.5, color="#777", linestyle="--", linewidth=0.9)
    ax.set_xticks([0, 1], labels); ax.set_ylim(0.2, 0.9); ax.set_ylabel("Concordance index")
    ax.set_title("Locked-score discrimination by evidence layer")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "Figure3_internal_external_performance")

    # Figure 4: exploratory interaction effects.
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    items = [x for x in interactions["interactions"] if x.get("interaction_hr") is not None]
    names = ["Weiss score", "Weiss >=6", "Hormone status"]
    hrs = [x["interaction_hr"] for x in items]
    ps = [x["interaction_p_lrt"] for x in items]
    y = np.arange(len(items))
    ax.scatter(hrs, y, color="#756bb1", s=55)
    for yi, hr, p in zip(y, hrs, ps):
        ax.text(hr + 0.012, yi, f"P={p:.3f}", va="center", fontsize=8)
    ax.axvline(1.0, color="#777", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, names); ax.set_xlabel("Interaction hazard ratio")
    ax.set_title("Exploratory score-by-clinical-feature interactions")
    save(fig, "Figure4_interaction_audit")

    # Figure 5: external platform coverage.
    coverage = mapping["gene_probe_coverage"]
    names = list(coverage.keys())
    vals = [coverage[name] for name in names]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(np.arange(len(names)), vals, color=["#31a354" if value else "#bdbdbd" for value in vals])
    ax.set_xticks(np.arange(len(names)), names, rotation=75, ha="right")
    ax.set_ylabel("Mapped GPL16686 probes")
    ax.set_title("GSE49278 steroid-program coverage")
    ax.text(0.99, 0.95, "10/22 genes covered; no survival table", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    save(fig, "Figure5_GSE49278_mapping")

    print(json.dumps({"figure_dir": str(OUT), "figures": sorted(path.name for path in OUT.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
