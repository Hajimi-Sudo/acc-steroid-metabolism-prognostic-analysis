#!/usr/bin/env python3
"""Create aggregate GSE49278 update tables and publication figures remotely."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
REAN = ROOT / "results" / "gse49278_survival_reanalysis"
EXT = ROOT / "results" / "external_validation"
FIG = ROOT / "results" / "figures"
FD = ROOT / "results" / "figure_data_export"
MI = ROOT / "results" / "manuscript_inputs"
for path in [FIG, FD, MI]:
    path.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
BLUE = "#2F6B8A"
TEAL = "#2D8C85"
RED = "#C8554D"
GOLD = "#D39B3D"
MUTED = "#667085"
GRID = "#D9E2EC"
PALE = "#F3F7FA"


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax, title: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=NAVY, fontweight="bold", pad=12)


def write(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False)


def figure1(summary: dict, gse33371: dict) -> None:
    cohort = pd.DataFrame([
        {"cohort": "TCGA-ACC", "role": "Discovery / internal OOF", "n": 79, "events": 28, "endpoint": "OS"},
        {"cohort": "GSE19750", "role": "Frozen four-gene OS test", "n": 21, "events": 17, "endpoint": "OS"},
        {"cohort": "GSE33371", "role": "Follow-up annotated GPL570 test", "n": gse33371["n_acc_complete_positive_os"], "events": gse33371["events"], "endpoint": "OS"},
        {"cohort": "GSE49278", "role": "Partial program OS test", "n": 42, "events": 19, "endpoint": "OS"},
        {"cohort": "GSE49278", "role": "Expression mapping", "n": 44, "events": 0, "endpoint": "Expression"},
    ])
    write(FD / "Figure1_cohort_roles_v3.tsv", cohort)
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.52, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    labels = ["TCGA-ACC", "GSE19750", "GSE33371\nOS", "GSE49278\nOS-matched", "GSE49278\nall expression"]
    ns = [79, 21, gse33371["n_acc_complete_positive_os"], 42, 44]
    colors = [BLUE, GOLD, "#8A6BBE", RED, TEAL]
    bars = ax.bar(labels, ns, color=colors, width=0.68)
    ax.set_ylabel("Samples")
    ax.set_ylim(0, 88)
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, n + 2, str(n), ha="center", color=NAVY, fontweight="bold")
    style(ax, "Cohorts were assigned distinct evidence roles")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    events = [28, 17, gse33371["events"], 19]
    total = [79, 21, gse33371["n_acc_complete_positive_os"], 42]
    ax.bar(x, total, color="#DCE7EF", width=0.62, label="Censored / no event")
    ax.bar(x, events, color=[BLUE, GOLD, RED], width=0.62, label="OS event")
    ax.set_xticks(x, ["TCGA-ACC", "GSE19750", "GSE33371", "GSE49278"])
    ax.set_ylabel("Cases")
    ax.set_ylim(0, 88)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for i, (n, e) in enumerate(zip(total, events)):
        ax.text(i, n + 2, f"{e}/{n} events", ha="center", fontsize=8, color=NAVY)
    style(ax, "Matched OS evidence was available in two external layers")

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "GSE49278 identity bridge", fontsize=12, color=NAVY, fontweight="bold")
    steps = [("44", "GEO samples", BLUE), ("44", "ACC labels", TEAL), ("42", "OS-matched", RED), ("19", "OS events", GOLD)]
    for i, (n, label, color) in enumerate(steps):
        x = 0.07 + i * 0.30
        ax.add_patch(plt.Circle((x, 0.55), 0.095, color=color, alpha=0.95))
        ax.text(x, 0.55, n, ha="center", va="center", color="white", fontsize=16, fontweight="bold")
        ax.text(x, 0.30, label, ha="center", va="center", color=NAVY, fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 0.20, 0.55), xytext=(x + 0.10, 0.55), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 2})
    ax.text(0.02, 0.08, "GSE49278 bridge: GEO !Sample_title (ACC label) → Assié et al. MOESM14 Tumor ID", fontsize=8, color=MUTED)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "What the external layer can support", fontsize=12, color=NAVY, fontweight="bold")
    rows = [("23/33", "GSE33371 OS cases", "purple"), ("42/44", "GSE49278 matched OS", "red"), ("10/22", "GSE49278 genes mapped", "teal"), ("0", "new feature selection", "blue")]
    for i, (value, label, color_name) in enumerate(rows):
        y = 0.72 - i * 0.18
        color = {"red": RED, "teal": TEAL, "gold": GOLD, "blue": BLUE, "purple": "#8A6BBE"}[color_name]
        ax.text(0.03, y, value, color=color, fontsize=14, fontweight="bold", va="center")
        ax.text(0.26, y, label, color=NAVY, fontsize=9, va="center")
    ax.text(0.02, 0.08, "Result: partial program OS association; complete four-gene validation remains blocked.", fontsize=8, color=MUTED)
    savefig(fig, "Figure1_cohort_roles_v3")


def figure5(summary: dict) -> None:
    mapping = json.loads((EXT / "mapping_audit.json").read_text())
    coverage = pd.DataFrame({"gene": list(mapping["gene_probe_coverage"]), "probes": list(mapping["gene_probe_coverage"].values())})
    coverage["locked"] = coverage["gene"].isin(["STAR", "NR0B1", "HSD11B2", "AKR1C3"])
    coverage["mapped"] = coverage["probes"] > 0
    write(FD / "Figure5_probe_coverage_v3.tsv", coverage)
    association = pd.DataFrame([
        {"analysis": "GSE19750 frozen four-gene score", "n": 21, "events": 17, "cindex": 0.676, "hr": 1.337, "ci_low": 0.779, "ci_high": 2.296, "p": 0.292, "status": "locked score; small test"},
        {"analysis": "GSE33371 follow-up annotated four-gene score", "n": gse33371["n_acc_complete_positive_os"], "events": gse33371["events"], "cindex": gse33371["cindex"], "hr": gse33371["cox_hr_per_score_unit"], "ci_low": gse33371["cox_ci_low"], "ci_high": gse33371["cox_ci_high"], "p": gse33371["cox_p"], "status": "same expression as GSE10927; exploratory"},
        {"analysis": "GSE49278 mapped 10/22-gene program", "n": summary["n_os_matched_positive_time"], "events": summary["os_events_matched_positive_time"], "cindex": summary["partial_program_os_association"]["cindex"], "hr": summary["partial_program_os_association"]["hr_per_score_unit"], "ci_low": summary["partial_program_os_association"]["ci_low"], "ci_high": summary["partial_program_os_association"]["ci_high"], "p": summary["partial_program_os_association"]["p"], "status": "partial program; exploratory"},
    ])
    write(FD / "Figure5_external_os_association_v3.tsv", association)
    fig = plt.figure(figsize=(13.2, 8.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1], hspace=0.62, wspace=0.34)

    ax = fig.add_subplot(gs[0, :])
    order = coverage.sort_values(["locked", "probes", "gene"], ascending=[False, False, True]).reset_index(drop=True)
    colors = [RED if locked and mapped else (GOLD if locked else (TEAL if mapped else "#DCE3EB")) for locked, mapped in zip(order["locked"], order["mapped"])]
    bars = ax.bar(np.arange(len(order)), order["probes"], color=colors, width=0.82)
    ax.set_xticks(np.arange(len(order)), order["gene"], rotation=58, ha="right", fontsize=8)
    ax.set_ylabel("Native GPL16686 probes")
    ax.set_ylim(0, max(order["probes"].max() + 2, 4))
    ax.set_title("Native-platform coverage determines what can be validated", loc="left", color=NAVY, fontweight="bold", pad=12)
    ax.text(0.01, 0.96, "red = locked and mapped   gold = locked but absent   teal = other mapped genes", transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    style(ax)

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(association))[::-1]
    for yi, row in zip(y, association.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=RED if "GSE49278" in row.analysis else GOLD, lw=3, solid_capstyle="round")
        ax.scatter([row.hr], [yi], s=80, color=RED if "GSE49278" in row.analysis else GOLD, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(17.8, yi, f"HR {row.hr:.2f}  P={row.p:.3f}\nn={row.n}, E={row.events}", va="center", fontsize=8, color=NAVY)
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.55, 32)
    ax.set_yticks(y, ["GSE49278\n10/22 program", "GSE33371\nfollow-up four-gene", "GSE19750\nfrozen four-gene"])
    ax.set_xlabel("Cox hazard ratio per score unit (log scale)")
    style(ax, "External OS associations are not equivalent score validations")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "Claim boundary", fontsize=12, color=NAVY, fontweight="bold")
    items = [("Supported", "42 matched OS cases; 10/22-gene program association", TEAL), ("Blocked", "NR0B1 and AKR1C3 absent on GPL16686", RED), ("Not done", "No outcome-driven gene or threshold re-selection", BLUE)]
    for i, (head, text, color) in enumerate(items):
        y0 = 0.70 - i * 0.24
        ax.add_patch(plt.Rectangle((0.03, y0 - 0.07), 0.035, 0.14, color=color, alpha=0.95))
        ax.text(0.10, y0 + 0.025, head, fontsize=9, color=color, fontweight="bold")
        ax.text(0.10, y0 - 0.025, text, fontsize=8, color=NAVY, wrap=True)
    ax.text(0.02, 0.08, "This panel supports independent partial-program evidence, not definitive four-gene validation.", fontsize=8, color=MUTED)
    savefig(fig, "Figure5_GSE49278_audit_v3")


def main() -> None:
    summary = json.loads((REAN / "GSE49278_survival_reanalysis.json").read_text())
    gse33371 = json.loads((EXT / "GSE33371_locked_signature_validation.json").read_text())
    figure1(summary, gse33371)
    figure5(summary, gse33371)
    rows = pd.DataFrame([
        {"cohort": "TCGA-ACC", "role": "discovery/internal validation", "endpoint": "OS", "n": 79, "events": 28, "platform": "RNA-seq", "status": "primary"},
        {"cohort": "GSE19750", "role": "adult external test", "endpoint": "OS", "n": 21, "events": 17, "platform": "GPL570", "status": "small directional test"},
        {"cohort": "GSE33371", "role": "follow-up annotated four-gene test", "endpoint": "OS", "n": gse33371["n_acc_complete_positive_os"], "events": gse33371["events"], "platform": "GPL570", "status": "same expression as GSE10927; exploratory"},
        {"cohort": "GSE49278", "role": "partial program external test", "endpoint": "OS", "n": summary["n_os_matched_positive_time"], "events": summary["os_events_matched_positive_time"], "platform": "GPL16686", "status": "10/22 genes; locked score incomplete"},
        {"cohort": "GSE49278", "role": "expression transportability", "endpoint": "expression", "n": 44, "events": "NA", "platform": "GPL16686", "status": "10/22 genes mapped"},
    ])
    write(MI / "table1_cohorts_v3.tsv", rows)
    manifest = {
        "version": "gse49278_update_v3",
        "summary_source": str(REAN / "GSE49278_survival_reanalysis.json"),
        "figures": ["Figure1_cohort_roles_v3", "Figure5_GSE49278_audit_v3"],
        "aggregate_tables": ["Figure1_cohort_roles_v3.tsv", "Figure5_probe_coverage_v3.tsv", "Figure5_external_os_association_v3.tsv", "table1_cohorts_v3.tsv"],
        "claim_boundary": summary["claim_status"],
    }
    (FD / "GSE49278_update_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
