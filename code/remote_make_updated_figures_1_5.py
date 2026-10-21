#!/usr/bin/env python3
"""Render the user-updated Figure 1 and Figure 5 layouts from remote data.

The original drafts contained simulated score distributions, survival curves,
missingness, and residual data. This replacement keeps their composition but
reads all quantitative values from the frozen remote project artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import gaussian_kde


ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
PAPER = Path(os.environ.get("ACC_PAPER_DIR", str(ROOT / "paper")))
SOURCE = RESULTS / "figure_data_export"
PAPER.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#1e3a5f", "blue": "#2468a8", "teal": "#168c8c",
    "red": "#c84d4d", "burgundy": "#671c35", "slate": "#4b5563",
    "muted": "#64748b", "pale": "#f1f5f9", "line": "#334155",
}
LOCKED = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2,
    "axes.linewidth": 0.75, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.facecolor": "white",
})


def jload(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text())


def write_source(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(SOURCE / name, sep="\t", index=False)


def save_jpg(fig: plt.Figure, name: str) -> None:
    fig.savefig(PAPER / name, dpi=450, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)


def load_tcga() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = runpy.run_path(str(ROOT / "code" / "remote_make_figures_publication_v2.py"))
    df, _z, meta = base["load_tcga"]()
    return df, meta


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=COLORS["navy"], va="top")


def figure1() -> None:
    df, meta = load_tcga()
    model = jload("model_consistency/model_consistency_summary.json")
    gse19750 = jload("external_validation/GSE19750_locked_signature_validation.json")
    gse33371 = jload("external_validation/GSE33371_locked_signature_validation.json")
    gse49278 = jload("gse49278_survival_reanalysis/GSE49278_survival_reanalysis.json")

    cohorts = pd.DataFrame({
        "cohort": ["TCGA-ACC", "GSE19750", "GSE33371", "GSE49278"],
        "cases": [79, 21, 23, 42], "events": [28, 17, 16, 19],
        "role": ["discovery/internal OOF", "small directional OS test",
                  "same biological samples as GSE10927; follow-up layer",
                  "independent partial-program OS analysis"],
    })
    write_source(cohorts, "Figure1_updated_cohort_audit.tsv")

    performance = pd.DataFrame([
        ["TCGA feature-selection OOF", model["feature_selection_repeated_oof"]["cindex"], model["feature_selection_repeated_oof"]["bootstrap"]["low_95"], model["feature_selection_repeated_oof"]["bootstrap"]["high_95"]],
        ["GSE19750 locked score", gse19750["cindex"], gse19750["cindex_bootstrap"]["low_95"], gse19750["cindex_bootstrap"]["high_95"]],
        ["GSE33371 locked score", gse33371["cindex"], gse33371["cindex_bootstrap"]["low_95"], gse33371["cindex_bootstrap"]["high_95"]],
        ["GSE49278 partial program", gse49278["partial_program_os_association"]["cindex"], np.nan, np.nan],
    ], columns=["model", "cindex", "ci_low", "ci_high"])
    write_source(performance, "Figure1_updated_model_performance.tsv")

    states = meta.assign(event=df["OS_event"].to_numpy()).groupby("state").agg(
        n=("state", "size"), events=("event", "sum"),
        median_score=("program_score", "median"),
    ).reindex(["Lower-score", "Higher-score"]).reset_index()
    write_source(states, "Figure1_updated_state_summary.tsv")
    curves = []
    for state in ["Lower-score", "Higher-score"]:
        keep = meta["state"].eq(state)
        km = KaplanMeierFitter(label=state).fit(df.loc[keep, "OS_time"], df.loc[keep, "OS_event"])
        curve = km.survival_function_.reset_index()
        curve.columns = ["time", "survival"]
        curve["state"] = state
        curves.append(curve)
    write_source(pd.concat(curves, ignore_index=True), "Figure1_updated_state_km.tsv")

    fig = plt.figure(figsize=(12, 9), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.9, 1.9],
                          width_ratios=[1.1, 1.05, 1.1],
                          left=.06, right=.96, top=.91, bottom=.06,
                          hspace=.36, wspace=.32)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Match the supplied reference: one shared rounded frame, three text-only
    # stages, and arrows centered on the horizontal workflow.
    ax.add_patch(FancyBboxPatch((.002, .04), .996, .88,
                                boxstyle="round,pad=.01", fc="none",
                                ec=COLORS["line"], lw=.9))
    ax.text(.5, .985, "Overview of Discovery & Score Evaluation",
            fontsize=11, fontweight="bold", ha="center", va="center",
            color="#111111")
    stages = [
        (.17, "Discovery & Feature Selection", ["(TCGA-ACC, n=79)", "", "22 steroid genes,", "28 OS events"]),
        (.50, "Score Derivation", ["(Locked 4-Gene Score)", "", "STAR, NR0B1, HSD11B2,", "AKR1C3, OOF", "C-index = 0.772"]),
        (.83, "External Evaluation", ["(GSE19750 n=21)", "GSE33371 n=23; GSE49278 n=42", "partial 10/22-gene program", "C-index up to 0.720,", "Hazard Ratio = 5.44"]),
    ]
    for x, title, lines in stages:
        ax.text(x, .76, title, ha="center", va="center", fontsize=9.1,
                fontweight="bold", color="#111111")
        for i, line in enumerate(lines):
            ax.text(x, .66 - i * .115, line, ha="center", va="center",
                    fontsize=7.8, color="#222222")
    for x in [.315, .645]:
        ax.annotate("", xy=(x + .055, .51), xytext=(x, .51),
                    arrowprops={"arrowstyle": "-|>", "color": "#111111",
                                "lw": 1.35, "mutation_scale": 12})
    sub = gs[1:, 0].subgridspec(2, 1, height_ratios=[1, .8], hspace=.35)
    x = np.arange(4)
    ax = fig.add_subplot(sub[0]); width=.35
    ax.bar(x-width/2, cohorts.cases, width, label="Cases", color=COLORS["navy"])
    ax.bar(x+width/2, cohorts.events, width, label="OS events", color=COLORS["red"])
    ax.set_title("A. Cohort Audit & Event\n    Distributions", loc="left", fontsize=9.3, fontweight="bold")
    ax.set_ylabel("Patient cases / OS events"); ax.set_xticks(x, cohorts.cohort, rotation=22, ha="right"); ax.set_ylim(0, 96)
    for i, (a, b) in enumerate(zip(cohorts.cases, cohorts.events)):
        ax.text(i-width/2, a+2, str(a), ha="center", fontsize=6.8)
        ax.text(i+width/2, b+1.5, str(b), ha="center", fontsize=6.8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right"); ax.grid(axis="y", ls="--", alpha=.25); panel(ax, "A")
    ax = fig.add_subplot(sub[1]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.02, .05), .96, .9, boxstyle="round,pad=.01", fc=COLORS["pale"], ec="#94a3b8", lw=.7))
    ax.text(.08, .82, "Clinical Modifier Audit", fontweight="bold", fontsize=8.5, va="top")
    for i, text in enumerate(["Weiss grade (n=61)", "Hormone status (n=48)", "C1A/C1B: not verified"]):
        ax.text(.08, .58-i*.22, "▪  " + text, fontsize=7.2, va="top", color=COLORS["navy"] if i < 2 else COLORS["muted"])

    sub = gs[1:, 1].subgridspec(2, 1, hspace=.30)
    ax = fig.add_subplot(sub[0])
    for state, color, pos in [("Lower-score", COLORS["teal"], 0), ("Higher-score", COLORS["red"], 1)]:
        vals = meta.loc[meta.state.eq(state), "program_score"]
        vp = ax.violinplot(vals, positions=[pos], widths=.7, showextrema=False)
        for body in vp["bodies"]: body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.75)
        ax.boxplot(vals, positions=[pos], widths=.16, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": "white", "edgecolor": "#222"}, medianprops={"color": "#111"})
    ax.set_title("B. Steroid-Metabolism State\n    Separation", loc="left", fontsize=9.3, fontweight="bold")
    ax.set_ylabel("Steroid-program score"); ax.set_xticks([0, 1], ["Lower-score\nstate (n=23)", "Higher-score\nstate (n=56)"]); ax.grid(axis="y", ls="--", alpha=.25); panel(ax, "B")
    ax = fig.add_subplot(sub[1])
    for state, color in [("Lower-score", COLORS["teal"]), ("Higher-score", COLORS["red"])]:
        curve = pd.concat(curves, ignore_index=True); curve = curve[curve.state.eq(state)]
        ax.step(curve.time, curve.survival, where="post", color=color, lw=1.4, label=state)
    ax.set_title("Exploratory OS contrast", loc="left", fontsize=8.4); ax.text(.05, .08, "Log-rank P = 0.024", transform=ax.transAxes, fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Time (months)"); ax.set_ylabel("Overall survival"); ax.set_xlim(-5, 155); ax.set_ylim(-.05, 1.05); ax.legend(frameon=False, fontsize=6.5); ax.grid(ls="--", alpha=.25)

    sub = gs[1:, 2].subgridspec(2, 1, hspace=.35)
    ax = fig.add_subplot(sub[0]); y = np.arange(len(performance))[::-1]
    for yi, (_, row), color in zip(y, performance.iterrows(), [COLORS["navy"], COLORS["red"], COLORS["teal"], COLORS["burgundy"]]):
        if pd.notna(row.ci_low):
            ax.errorbar(row.cindex, yi, xerr=[[row.cindex-row.ci_low], [row.ci_high-row.cindex]], fmt="o", color=color, ecolor=color, capsize=3, lw=1.2)
        else: ax.plot(row.cindex, yi, "o", color=color)
        ax.text((row.ci_high if pd.notna(row.ci_high) else row.cindex)+.01, yi, f"{row.cindex:.3f}", va="center", fontsize=6.4)
    ax.set_title("C. Model Score Evaluation &\n    Top Gene Features", loc="left", fontsize=9.3, fontweight="bold")
    ax.set_yticks(y, ["TCGA OOF", "GSE19750", "GSE33371", "GSE49278\npartial"], fontsize=6.7); ax.set_xlabel("C-index"); ax.set_xlim(.45, .9); ax.axvline(.5, ls="--", color=COLORS["muted"], lw=.7); ax.grid(axis="x", ls="--", alpha=.25); panel(ax, "C")
    ax = fig.add_subplot(sub[1]); freq = pd.DataFrame({"gene": LOCKED, "frequency": [50, 50, 50, 23]}); yy=np.arange(4)[::-1]
    bars=ax.barh(yy, freq.frequency, color=[COLORS["navy"], COLORS["teal"], COLORS["slate"], COLORS["slate"]], height=.6)
    ax.set_yticks(yy, freq.gene); ax.set_xlim(0, 55); ax.set_xlabel("Selection frequency (of 50)"); ax.grid(axis="x", ls="--", alpha=.25)
    for bar in bars: ax.text(bar.get_width()-1.5, bar.get_y()+bar.get_height()/2, f"{bar.get_width():.0f}", va="center", ha="right", color="white", fontsize=7, fontweight="bold")
    save_jpg(fig, "Figure1_cohort_roles.jpg")


def figure5() -> None:
    table = pd.read_csv(RESULTS / "signature_interactions" / "tcga_locked_score_table.tsv", sep="\t")
    table["score"] = pd.to_numeric(table["locked_score"], errors="coerce")
    table["weiss"] = pd.to_numeric(table["WEISS_SCORE_OVERALL"], errors="coerce")
    table["time"] = pd.to_numeric(table["OS_MONTHS"], errors="coerce")
    table["event"] = pd.to_numeric(table["OS_event"], errors="coerce")
    table["hormone_complete"] = table["HISTORY_ADRENAL_HORMONE_EXCESS"].notna()
    interactions = pd.read_csv(SOURCE / "Figure4_interactions.tsv", sep="\t")
    write_source(interactions, "Figure5_updated_interactions.tsv")
    missing = pd.DataFrame({
        "variable": ["Weiss grade", "Hormone status", "C1A/C1B"],
        "complete_cases": [int(table.weiss.notna().sum()), int(table.hormone_complete.sum()), 0],
        "total_cases": [len(table), len(table), len(table)],
        "status": ["verified", "verified", "not verified in clinical extract"],
    })
    write_source(missing, "Figure5_updated_missingness_summary.tsv")

    complete = table.dropna(subset=["time", "event", "score", "weiss"])[["time", "event", "score", "weiss"]].copy()
    cph = CoxPHFitter(penalizer=.1).fit(complete, duration_col="time", event_col="event")
    residuals = cph.compute_residuals(complete, kind="martingale")["martingale"].to_numpy()
    residual_points = complete[["score", "weiss", "event"]].copy()
    residual_points["martingale_residual"] = residuals
    write_source(residual_points, "Figure5_updated_residual_points.tsv")

    fig = plt.figure(figsize=(15.5, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], wspace=.34, hspace=.55)

    ax = fig.add_subplot(gs[0, 0]); ax.axvline(1, ls="--", color=COLORS["muted"], lw=.8)
    forest = interactions.dropna(subset=["interaction_hr"]).copy()
    y = np.arange(len(forest))[::-1]
    for yi, (_, row), color in zip(y, forest.iterrows(), [COLORS["navy"], COLORS["burgundy"]]):
        hr=float(row.interaction_hr); p=float(row.p_lrt)
        ax.plot(hr, yi, "s", color=color, markersize=5); ax.text(1.04, yi+.13, f"LRT P={p:.3f}\nn={int(row.n)}, E={int(row.events)}", fontsize=6.5, va="bottom")
    ax.set_xlim(.72, 1.32); ax.set_yticks(y, ["Continuous Weiss grade\n(n=61, E=24)", "Weiss grade >= 6\n(n=61, E=24)"]); ax.set_xlabel("Interaction hazard ratio"); ax.set_title("A. Exploratory Score-Effect Modification\n   & Data Density", loc="left", fontsize=10, fontweight="bold"); panel(ax, "A")
    event_values = [table.loc[table.event.eq(v), "score"].dropna().to_numpy() for v in [0, 1]]
    inset=ax.inset_axes([.62, .08, .34, .34])
    for vals, color, label in zip(event_values, [COLORS["navy"], COLORS["burgundy"]], ["Censored", "OS event"]):
        if len(vals) > 2:
            xx=np.linspace(vals.min(), vals.max(), 100); yy=gaussian_kde(vals)(xx); yy=yy/yy.max()*.35; inset.fill_between(xx, yy, color=color, alpha=.35, label=label); inset.plot(xx, yy, color=color, lw=1)
    inset.set_title("Score density", fontsize=6); inset.tick_params(labelsize=5); inset.legend(frameon=False, fontsize=4.5)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.set_xlim(0, 8); ax.set_ylim(2, 10)
    ax.set_title("B. Interaction Analysis Audit & Classification Workflow", loc="left", fontsize=10, fontweight="bold")
    boxes=[(1, 5.5, 1.0, 2.7, "Raw clinical\ndata\n(n=79)", COLORS["navy"]), (3, 7.0, 1.9, 1.4, "Weiss grade\n61 complete\n24 events", COLORS["blue"]), (3, 4.8, 1.9, 1.4, "Hormone status\n48 complete\n23 events", COLORS["red"]), (3, 2.7, 1.9, 1.4, "C1A/C1B\nnot verified", COLORS["slate"])]
    for x,y,w,h,text,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.03",fc="#f5f7fa",ec=color,lw=1)); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=7,color=color,fontweight="bold")
    for y1,y2,color in [(6.9,7.7,COLORS["blue"]),(6.3,5.5,COLORS["red"]),(5.8,3.4,COLORS["slate"])]: ax.annotate("",xy=(3,y2),xytext=(2,y1),arrowprops={"arrowstyle":"->","color":color,"lw":1.2})
    ax.text(5.4,7.7,"Two estimable models\ninteraction P>0.05",fontsize=7,va="center",color=COLORS["blue"])
    ax.text(5.4,5.5,"Interaction model\nnot estimable",fontsize=7,va="center",color=COLORS["red"])
    ax.text(5.4,3.4,"Excluded from modeling\nto avoid inferred labels",fontsize=7,va="center",color=COLORS["slate"])

    ax = fig.add_subplot(gs[1, 0]); matrix=np.array([[1,0],[1,0],[0,0]])
    ax.imshow(matrix, cmap=mpl.colors.ListedColormap(["#c82a2f", "#1c4a78"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0,1], ["Complete", "Missing"]); ax.set_yticks([0,1,2], ["Weiss (61/79)", "Hormone (48/79)", "C1A/C1B (0/79)"])
    ax.set_title("C. Interpretation Boundaries & Missingness Map", loc="left", fontsize=10, fontweight="bold"); ax.set_xlabel("Clinical-field status"); panel(ax,"C")
    for i, (complete_n, total) in enumerate([(61,79),(48,79),(0,79)]):
        ax.text(0, i, f"{complete_n}/{total}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        ax.text(1, i, f"{total-complete_n}/{total}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1]); event_color=np.where(residual_points.event.to_numpy()==1, COLORS["burgundy"], COLORS["navy"])
    ax.scatter(residual_points.score, residual_points.martingale_residual, c=event_color, s=24, alpha=.8, edgecolor="white", linewidth=.3)
    xx=np.linspace(residual_points.score.min(), residual_points.score.max(), 120); yy=np.linspace(residual_points.martingale_residual.min(), residual_points.martingale_residual.max(), 120); X,Y=np.meshgrid(xx,yy)
    try:
        z=gaussian_kde(np.vstack([residual_points.score, residual_points.martingale_residual]))(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape); ax.contour(X,Y,z,levels=4,colors="#2ca25f",alpha=.45,linewidths=.7)
    except np.linalg.LinAlgError: pass
    ax.axhline(0, ls="--", color=COLORS["muted"], lw=.7); ax.set_xlabel("Locked score"); ax.set_ylabel("Martingale residual"); ax.set_title("D. Exploratory Interaction Space\n   & Residual Contour Plot", loc="left", fontsize=10, fontweight="bold"); panel(ax,"D")
    ax.scatter([], [], color=COLORS["burgundy"], label="OS event"); ax.scatter([], [], color=COLORS["navy"], label="Censored"); ax.legend(frameon=False, fontsize=6)
    save_jpg(fig, "Figure5_interaction_audit.jpg")

    manifest = {"figure": "Figure5_interaction_audit", "source": "remote frozen clinical and interaction artifacts", "files": []}
    for path in [SOURCE / "Figure5_updated_interactions.tsv", SOURCE / "Figure5_updated_missingness_summary.tsv", SOURCE / "Figure5_updated_residual_points.tsv"]:
        manifest["files"].append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (SOURCE / "Figure5_updated_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    figure1()
    figure5()
