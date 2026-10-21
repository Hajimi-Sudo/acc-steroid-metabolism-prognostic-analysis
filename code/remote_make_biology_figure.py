#!/usr/bin/env python3
"""Render the biological-association figure from remote aggregate results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RES = ROOT / "results"
FIG = RES / "figures"
SRC = RES / "figure_data_export"
BIO = RES / "biology_associations"
PAPER = ROOT / "paper"
FIGURE_TAG = os.environ.get("ACC_BIOLOGY_FIGURE_TAG", "Figure6")
FIG.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2468a8", "teal": "#168c8c", "red": "#c84d4d", "gold": "#d49b26", "purple": "#7251a3", "navy": "#243447", "muted": "#75859b", "grid": "#d9e2ec", "low": "#63a8c6"}
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.0, "axes.titlesize": 9.5, "axes.labelsize": 8.0, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def panel(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=C["navy"], va="top")


def write(df, name):
    df.to_csv(SRC / name, sep="\t", index=False)


def main():
    summary = json.loads((BIO / "biology_association_summary.json").read_text())
    table = pd.read_csv(RES / "signature_interactions" / "tcga_locked_score_table.tsv", sep="\t")
    table["score"] = pd.to_numeric(table["locked_score"], errors="coerce")
    table["weiss"] = pd.to_numeric(table["WEISS_SCORE_OVERALL"], errors="coerce")
    table["stage"] = table["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.extract(r"(IV|III|II|I)")[0].map({"I": 1, "II": 2, "III": 3, "IV": 4}) if "AJCC_PATHOLOGIC_TUMOR_STAGE" in table else np.nan
    if table["stage"].isna().all():
        clinical = pd.read_csv(ROOT / "data" / "processed" / "TCGA_ACC_clinical_selected.tsv", sep="\t")
        table = table.merge(clinical[["patient_id", "AJCC_PATHOLOGIC_TUMOR_STAGE"]], on="patient_id", how="left")
        table["stage"] = table["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.extract(r"(IV|III|II|I)")[0].map({"I": 1, "II": 2, "III": 3, "IV": 4})

    hormone = []
    ci = summary["hormone_association"].get("group_median_bootstrap_95", {})
    for group, part in table.dropna(subset=["score", "HISTORY_ADRENAL_HORMONE_EXCESS"]).groupby("HISTORY_ADRENAL_HORMONE_EXCESS"):
        hormone.append({"group": group, "n": len(part), "median_score": part["score"].median(), "mean_score": part["score"].mean(), "ci_low": ci.get(group, [np.nan, np.nan])[0], "ci_high": ci.get(group, [np.nan, np.nan])[1]})
    hormone = pd.DataFrame(hormone).sort_values("median_score")
    write(hormone, f"{FIGURE_TAG}_hormone_score_summary.tsv")

    weiss = table.dropna(subset=["score", "weiss"]).groupby("weiss").agg(n=("score", "size"), median_score=("score", "median"), q1=("score", lambda x: x.quantile(.25)), q3=("score", lambda x: x.quantile(.75))).reset_index()
    write(weiss, f"{FIGURE_TAG}_weiss_score_summary.tsv")
    stage = table.dropna(subset=["score", "stage"]).groupby("stage").agg(n=("score", "size"), median_score=("score", "median"), q1=("score", lambda x: x.quantile(.25)), q3=("score", lambda x: x.quantile(.75))).reset_index()
    write(stage, f"{FIGURE_TAG}_stage_score_summary.tsv")

    diff = pd.read_csv(BIO / "program_gene_differential.tsv", sep="\t")
    path = pd.read_csv(BIO / "program_pathway_scores.tsv", sep="\t")
    write(diff, f"{FIGURE_TAG}_program_gene_differential.tsv")
    write(path, f"{FIGURE_TAG}_program_pathway_scores.tsv")
    models = []
    baseline = summary["baseline_models"]
    for name, model in [("Stage only", baseline["stage_only"]), ("Stage + score", baseline["stage_plus_score"]), ("Weiss only", baseline["weiss_only"]), ("Weiss + score", baseline["weiss_plus_score"])]:
        if model.get("status") == "ok":
            key = "score" if "score" in name else list(model["hazard_ratios"])[0]
            models.append({"model": name, "term": key, "hr": model["hazard_ratios"][key], "p": model["p_values"][key], "cindex": model["cindex"], "n": model["n"], "events": model["events"]})
    inc = []
    for name in ["incremental_stage_only_plus_score", "incremental_weiss_only_plus_score"]:
        item = baseline[name]
        inc.append({"comparison": name.replace("incremental_", "").replace("_plus_score", " + score"), "lrt_chi2": item["lrt_chi2"], "p": item["p"], "delta_aic_partial": item["delta_aic_partial"]})
    write(pd.DataFrame(models), f"{FIGURE_TAG}_nested_cox_models.tsv")
    write(pd.DataFrame(inc), f"{FIGURE_TAG}_incremental_lrt.tsv")

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(hormone))
    ax.errorbar(hormone["median_score"], y, xerr=[hormone["median_score"] - hormone["ci_low"], hormone["ci_high"] - hormone["median_score"]], fmt="o", color=C["purple"], ecolor=C["purple"], capsize=3, lw=1.4)
    ax.set_yticks(y, [f"{g}\n(n={n})" for g, n in zip(hormone["group"], hormone["n"])], fontsize=7)
    ax.set_xlabel("Locked score; median and bootstrap 95% CI"); ax.set_title("Hormone secretion group")
    ax.text(.02, .98, f"Kruskal--Wallis P={summary['hormone_association']['p']:.3f}", transform=ax.transAxes, fontsize=7, color=C["muted"], va="top"); panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(weiss["weiss"], weiss["median_score"], yerr=[weiss["median_score"] - weiss["q1"], weiss["q3"] - weiss["median_score"]], fmt="o-", color=C["teal"], capsize=3, lw=1.3)
    ax.set_xlabel("Weiss score"); ax.set_ylabel("Locked score")
    ax.set_title(f"Weiss association\nSpearman $\\rho$={summary['weiss_association']['spearman_rho']:.3f}, P={summary['weiss_association']['p']:.3f}"); panel(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    ax.errorbar(stage["stage"], stage["median_score"], yerr=[stage["median_score"] - stage["q1"], stage["q3"] - stage["median_score"]], fmt="o-", color=C["red"], capsize=3, lw=1.4)
    ax.set_xticks([1, 2, 3, 4], ["I", "II", "III", "IV"]); ax.set_xlabel("AJCC stage"); ax.set_ylabel("Locked score")
    ax.set_title(f"Stage association\nSpearman $\\rho$={summary['stage_association']['spearman_rho']:.3f}, P<0.001"); panel(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    show = diff.sort_values("median_difference_log2"); colors = [C["red"] if p < .05 else C["muted"] for p in show["fdr_bh"]]
    ax.barh(np.arange(len(show)), show["median_difference_log2"], color=colors, height=.72)
    ax.set_yticks(np.arange(len(show)), show["gene"], fontsize=6.8); ax.set_xlabel("Higher-score minus lower-score median (log2)"); ax.set_title("Prespecified 22-gene program"); panel(ax, "D")
    ax.text(.02, .03, "Mann--Whitney tests; FDR shown in source data", transform=ax.transAxes, fontsize=6.5, color=C["muted"])

    ax = fig.add_subplot(gs[1, 1])
    show = path.sort_values("median_difference"); ax.barh(np.arange(len(show)), show["median_difference"], color=[C["gold"], C["teal"], C["purple"], C["blue"]])
    ax.set_yticks(np.arange(len(show)), [x.replace("_", " ") for x in show["pathway"]], fontsize=6.4); ax.set_xlabel("Median program-score difference"); ax.set_title("Predefined pathway modules"); panel(ax, "E")
    ax.text(.02, .03, "22-gene module scores; no genome-wide enrichment", transform=ax.transAxes, fontsize=6.0, color=C["muted"])

    ax = fig.add_subplot(gs[1, 2])
    show = pd.DataFrame(models); positions = np.arange(len(show));
    ax.axvline(1, color=C["muted"], ls="--", lw=.8)
    for i, row in show.iterrows():
        ax.scatter(row["hr"], i, s=48, color=C["red"] if "score" in row["term"] else C["blue"], zorder=3)
        ax.text(max(row["hr"] + .04, 1.03), i, f"HR {row['hr']:.2f}; P={row['p']:.3g}", va="center", fontsize=6.8)
    ax.set_yticks(positions, show["model"], fontsize=7); ax.set_xlabel("Hazard ratio for displayed term"); ax.set_xlim(.75, 2.6); ax.set_title("Low-complexity prognostic comparison"); panel(ax, "F")
    ax.text(.02, .04, "Full-data TCGA fit; exploratory", transform=ax.transAxes, fontsize=6.2, color=C["muted"], va="bottom")

    fig.savefig(FIG / f"{FIGURE_TAG}_biological_associations.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG / f"{FIGURE_TAG}_biological_associations.pdf", bbox_inches="tight")
    fig.savefig(PAPER / f"{FIGURE_TAG}_biological_associations.jpg", dpi=450,
                bbox_inches="tight", pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)
    files = []
    for p in sorted(SRC.glob(f"{FIGURE_TAG}_*")):
        files.append({"file": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (SRC / f"{FIGURE_TAG}_MANIFEST.json").write_text(json.dumps({"figure": f"{FIGURE_TAG}_biological_associations", "source": "remote biology_association results", "files": files}, indent=2) + "\n")
    print(json.dumps({"figure": str(FIG / f"{FIGURE_TAG}_biological_associations.pdf"), "paper_jpg": str(PAPER / f"{FIGURE_TAG}_biological_associations.jpg"), "source_files": [x["file"] for x in files]}, indent=2))


if __name__ == "__main__":
    main()
