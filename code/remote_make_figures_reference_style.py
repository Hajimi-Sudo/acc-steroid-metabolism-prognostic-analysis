#!/usr/bin/env python3
"""Render ACC figures with the supplied multi-panel reference layouts.

All numeric panels use frozen remote result artifacts. This script changes only
composition and visual encoding; it does not fit models or alter estimates.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remote_make_figures_publication_v2 as base

ROOT = Path(base.ROOT)
RES = base.RES
PROC = base.PROC
FIG = base.FIG
SRC = base.SRC
PAPER = Path(os.environ.get("ACC_PAPER_DIR", str(ROOT / "paper")))
PAPER.mkdir(parents=True, exist_ok=True)
C = base.C
GENES = base.GENES
LOCKED = base.LOCKED

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.titlesize": 8.6, "axes.labelsize": 7.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.65, "xtick.major.width": 0.55,
    "ytick.major.width": 0.55, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    # Manuscript-facing raster assets live beside main.tex; the remote PDF/PNG
    # exports remain available for audit and figure-quality review.
    fig.savefig(PAPER / f"{name}.jpg", dpi=450, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=C["navy"], va="top")


def box(ax, xy, width, height, title, body, color):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.008,rounding_size=0.015",
                 facecolor="white", edgecolor=color, linewidth=1.4))
    ax.text(x + width / 2, y + height * .70, title, ha="center", va="center",
            fontsize=7.3, fontweight="bold", color=color)
    ax.text(x + width / 2, y + height * .38, body, ha="center", va="center",
            fontsize=6.5, linespacing=1.25)


def figure1():
    cohorts = pd.DataFrame({
        "cohort": ["TCGA-ACC", "GSE19750", "GSE33371", "GSE49278"],
        "n": [79, 21, 23, 42], "events": [28, 17, 16, 19],
        "role": ["discovery/internal OOF", "small directional OS test",
                 "same biological samples as GSE10927; follow-up layer",
                 "independent partial-program OS analysis"],
    })
    model = base.load_json("model_consistency/model_consistency_summary.json")
    gse19750 = base.load_json("external_validation/GSE19750_locked_signature_validation.json")
    gse33371 = base.load_json("external_validation/GSE33371_locked_signature_validation.json")
    performance = pd.DataFrame([
        {"object": "TCGA feature-selection OOF", "estimate": model["feature_selection_repeated_oof"]["cindex"],
         "low": model["feature_selection_repeated_oof"]["bootstrap"]["low_95"],
         "high": model["feature_selection_repeated_oof"]["bootstrap"]["high_95"], "n": 79, "events": 28},
        {"object": "TCGA fixed four-gene OOF", "estimate": model["fixed_four_gene_repeated_oof"]["cindex"],
         "low": model["fixed_four_gene_repeated_oof"]["bootstrap"]["low_95"],
         "high": model["fixed_four_gene_repeated_oof"]["bootstrap"]["high_95"], "n": 79, "events": 28},
        {"object": "GSE19750 OS test", "estimate": gse19750["cindex"],
         "low": gse19750["cindex_bootstrap"]["low_95"], "high": gse19750["cindex_bootstrap"]["high_95"], "n": 21, "events": 17},
        {"object": "GSE33371 follow-up", "estimate": gse33371["cindex"],
         "low": gse33371["cindex_bootstrap"]["low_95"], "high": gse33371["cindex_bootstrap"]["high_95"], "n": 23, "events": 16},
    ])
    frequencies = pd.DataFrame({"gene": list(model["feature_selection_frequency"]),
                                "selected_folds": list(model["feature_selection_frequency"].values())})
    frequencies = frequencies[frequencies.gene.isin(LOCKED)].copy()
    frequencies["total_folds"] = 50
    frequencies["label"] = frequencies.selected_folds.astype(int).astype(str) + "/50"

    df, _, meta = base.load_tcga()
    state_summary = meta.assign(OS_event=df["OS_event"].to_numpy()).groupby("state").agg(
        n=("state", "size"), events=("OS_event", "sum"),
        median_score=("program_score", "median"),
    ).reindex(["Lower-score", "Higher-score"]).reset_index()
    km_rows = []
    for label, color in [("Lower-score", C["low"]), ("Higher-score", C["high"] )]:
        keep = meta["state"] == label
        km = KaplanMeierFitter(label=label).fit(df.loc[keep, "OS_time"], df.loc[keep, "OS_event"])
        curve = km.survival_function_.reset_index()
        curve.columns = ["time", "survival"]
        curve["state"] = label
        km_rows.append(curve)

    # Figure 1 follows the supplied template: workflow strip above three
    # evidence blocks (cohort audit, state/OS contrast, model evaluation).
    fig = plt.figure(figsize=(13.8, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[.92, 1.42], hspace=.58, wspace=.46)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(.5, .96, "Overview of discovery and score evaluation", ha="center", va="top",
            fontsize=11.2, fontweight="bold", color=C["navy"])
    steps = [
        (.02, "DISCOVERY & FEATURE SELECTION", "TCGA-ACC (n=79)\n22 steroid genes; 28 OS events", C["blue"]),
        (.355, "SCORE DERIVATION", "Locked four-gene score\nSTAR, NR0B1, HSD11B2, AKR1C3", C["teal"]),
        (.69, "INDEPENDENT VALIDATION", "GSE19750 (n=21)\nGSE33371 follow-up; GSE49278 partial program", C["gold"]),
    ]
    for x, title, body, color in steps:
        ax.add_patch(FancyBboxPatch((x, .30), .25, .43,
                     boxstyle="round,pad=.012,rounding_size=.018",
                     facecolor="white", edgecolor=color, linewidth=2.0))
        ax.text(x + .125, .64, title, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=color)
        ax.text(x + .125, .48, body, ha="center", va="center", fontsize=8.0, linespacing=1.35)
    for x in [.285, .62]:
        ax.annotate("", xy=(x + .035, .515), xytext=(x, .515),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": C["muted"]})
    ax.text(.5, .10, "Evidence boundary: internal reproducibility + directional external tests; no efficacy claim",
            ha="center", fontsize=7.4, color=C["red"])

    acol = gs[1, 0].subgridspec(2, 1, height_ratios=[1, 1], hspace=.58)
    ax = fig.add_subplot(acol[0, 0]); x = np.arange(4); width = .58
    ax.bar(x, cohorts.n, width, color=[C["blue"], C["teal"], C["purple"], C["gold"]])
    ax.set_xticks(x, cohorts.cohort, rotation=25, ha="right"); ax.set_ylabel("Patient cases")
    ax.set_title("Cohort audit", fontsize=9.0, pad=3)
    for i, value in enumerate(cohorts.n): ax.text(i, value + 2, str(value), ha="center", fontsize=7)
    panel(ax, "A")
    ax = fig.add_subplot(acol[1, 0]); ax.bar(x, cohorts.events, width, color=C["red"])
    ax.set_xticks(x, cohorts.cohort, rotation=25, ha="right"); ax.set_ylabel("OS events")
    ax.set_title("Endpoint distribution", fontsize=9.0, pad=3)
    for i, value in enumerate(cohorts.events): ax.text(i, value + 1.5, str(value), ha="center", fontsize=7)

    bcol = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 1.18], hspace=.62)
    ax = fig.add_subplot(bcol[0, 0])
    for label, color, pos in [("Lower-score", C["low"], 0), ("Higher-score", C["high"], 1)]:
        vals = meta.loc[meta["state"] == label, "program_score"]
        vp = ax.violinplot(vals, positions=[pos], widths=.70, showextrema=False, quantiles=[[.25, .75]])
        for body in vp["bodies"]: body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.78)
        ax.boxplot(vals, positions=[pos], widths=.16, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": "white", "edgecolor": C["navy"]}, medianprops={"color": C["navy"]})
    ax.set_xticks([0, 1], ["Lower-score\n(n=23)", "Higher-score\n(n=56)"])
    ax.set_ylabel("Program score"); ax.set_title("State separation", fontsize=9.0, pad=3); panel(ax, "B")
    ax = fig.add_subplot(bcol[1, 0])
    for label, color in [("Lower-score", C["low"]), ("Higher-score", C["high"] )]:
        curve = next(row for row in km_rows if row["state"].iloc[0] == label)
        ax.step(curve["time"], curve["survival"], where="post", color=color, lw=1.6, label=label)
    ax.set_ylim(0, 1.05); ax.set_xlabel("OS (months)"); ax.set_ylabel("Survival")
    ax.set_title("Exploratory OS contrast\nlog-rank P=0.024", fontsize=8.4, pad=2)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")

    ccol = gs[1, 2].subgridspec(2, 1, height_ratios=[1.05, 1], hspace=.62)
    ax = fig.add_subplot(ccol[0, 0]); y = np.arange(len(performance))[::-1]
    colors = [C["blue"], C["teal"], C["red"], C["purple"]]
    for yi, (_, row), color in zip(y, performance.iterrows(), colors):
        ax.errorbar(row["estimate"], yi,
                    xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
                    fmt="o", color=color, capsize=3, lw=1.2)
        ax.text(row["high"] + .012, yi, f"{row['estimate']:.3f}", va="center", fontsize=6.3)
    ax.axvline(.5, color=C["muted"], ls="--", lw=.7)
    ax.set_yticks(y, ["TCGA OOF\n(feature)", "TCGA OOF\n(fixed genes)", "GSE19750\nOS test", "GSE33371\nfollow-up"], fontsize=6.4)
    ax.set_xlim(.45, .90); ax.set_xlabel("C-index"); ax.set_title("Model score evaluation", fontsize=9.0, pad=3); panel(ax, "C")
    ax = fig.add_subplot(ccol[1, 0]); selected = frequencies.sort_values("selected_folds")
    yy = np.arange(len(selected)); ax.barh(yy, selected["selected_folds"], color=C["teal"], height=.58)
    ax.set_yticks(yy, selected["gene"]); ax.set_xlim(0, 54); ax.set_xlabel("Selected folds (of 50)")
    ax.set_title("Top gene features", fontsize=8.8, pad=3)
    for yi, value in zip(yy, selected["selected_folds"]): ax.text(value + 1, yi, f"{int(value)}/50", va="center", fontsize=6.8)

    base.write(cohorts.assign(endpoint="OS"), "Figure1_cohort_roles.tsv")
    base.write(state_summary, "Figure1_state_summary.tsv")
    base.write(pd.concat(km_rows, ignore_index=True), "Figure1_state_km_curve_coordinates.tsv")
    base.write(performance, "Figure1_model_performance.tsv")
    base.write(frequencies, "Figure1_gene_selection_frequency.tsv")
    save(fig, "Figure1_cohort_roles")


def figure2():
    df, z, meta = base.load_tcga()
    robustness = base.load_json("state_robustness/state_robustness_summary.json")
    tcga = robustness["tcga"]
    init, patient = tcga["initialization_stability"], tcga["patient_resampling_stability"]
    k_selection = pd.DataFrame(tcga["k_selection"]["table"])
    state_order = ["Lower-score", "Higher-score"]
    ordered = []
    for state_name in state_order:
        idx = meta.index[meta.state == state_name].to_numpy()
        idx = idx[leaves_list(linkage(z.loc[idx, GENES], method="average", metric="correlation", optimal_ordering=True))]
        ordered.append(idx)
    order = np.concatenate(ordered); heat = z.loc[order, GENES].T
    row_link = linkage(z.loc[:, GENES].T, method="average", metric="correlation", optimal_ordering=True)
    heat = heat.iloc[leaves_list(row_link)[::-1]]
    states = meta.loc[order, "state"].map({"Lower-score": 0, "Higher-score": 1}).to_numpy()
    events = df.loc[order, "OS_event"].fillna(0).astype(int).to_numpy()
    stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
    stages = df.loc[order, "AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).map(stage_map).fillna(0).astype(int).to_numpy()
    # Reference layout: one full-width heatmap above four compact panels.
    fig = plt.figure(figsize=(14.6, 8.35))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.72, 1.0], hspace=.42, wspace=.30)
    # Keep the tree attached to the heatmap and place annotation labels on the
    # outside of the top strips, leaving no artificial gap between panels.
    agrid = gs[0, :].subgridspec(3, 3, height_ratios=[.34, .34, 4.75], width_ratios=[.38, 7.0, .72], hspace=.02, wspace=.018)
    title_ax = fig.add_subplot(agrid[0, :]); title_ax.axis("off")
    title_ax.text(.01, .78, "A", transform=title_ax.transAxes,
                  fontsize=14, fontweight="bold", color=C["navy"],
                  ha="left", va="top")
    title_ax.text(.5, .55, "Coordinated steroid-program expression across TCGA-ACC (n=79)", ha="center", fontsize=10.2)
    tree = fig.add_subplot(agrid[1:, 0]); dendrogram(row_link, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=C["muted"], ax=tree); tree.invert_yaxis(); tree.axis("off")
    ann = fig.add_subplot(agrid[1, 1]); rgb = np.stack([
        np.array([["#63a8c6", "#ca5555"][int(x)] for x in states]),
        np.array([["#f2f2f2", "#222222"][int(x)] for x in events]),
        np.array([["#d8d8d8", "#4c78a8", "#f58518", "#54a24b", "#e45756"][int(x)] for x in stages]),
    ])
    ann.imshow(np.array([[mpl.colors.to_rgb(v) for v in row] for row in rgb]), aspect="auto", interpolation="nearest")
    ann.set_yticks([]); ann.set_xticks([]); ann.tick_params(length=0); ann.spines[:].set_visible(False)
    ann_labels = fig.add_subplot(agrid[1, 2]); ann_labels.axis("off")
    for y, label in zip([5/6, 1/2, 1/6], ["state", "OS event", "stage"]):
        ann_labels.text(.04, y, label, ha="left", va="center", fontsize=7.4,
                        transform=ann_labels.transAxes)
    ah = fig.add_subplot(agrid[2, 1]); im = ah.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ah.set_yticks(np.arange(len(heat)), heat.index, fontsize=7.0); ah.yaxis.tick_right(); ah.set_xticks([]); ah.tick_params(axis="y", length=0, pad=2); ah.axvline(len(ordered[0])-.5, color="white", lw=.8)
    cb = fig.colorbar(im, ax=ah, fraction=.025, pad=.13); cb.set_label("Gene-wise z-score", fontsize=7.0); cb.ax.tick_params(labelsize=6.0)
    ax = fig.add_subplot(gs[1, 0])
    for label, color, pos in [("Lower-score", C["low"], 0), ("Higher-score", C["high"], 1)]:
        vals = meta.loc[meta.state == label, "program_score"]
        vp = ax.violinplot(vals, positions=[pos], widths=.7, showextrema=False, quantiles=[[.25, .75]])
        for body in vp["bodies"]: body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(.78)
        ax.boxplot(vals, positions=[pos], widths=.16, patch_artist=True, showfliers=False, boxprops={"facecolor":"white","edgecolor":C["navy"]}, medianprops={"color":C["navy"]})
    ax.set_xticks([0, 1], ["Lower-score\n(n=23; E=3)", "Higher-score\n(n=56; E=25)"]); ax.set_ylabel("Program score", fontsize=7.2); ax.set_title("State separation", pad=3, fontsize=9.0); ax.tick_params(labelsize=6.2); panel(ax, "B")

    cgrid = gs[1, 1].subgridspec(2, 1, height_ratios=[1.0, .68], hspace=.18)
    ax = fig.add_subplot(cgrid[0, 0])
    for label, color in [("Lower-score", C["low"]), ("Higher-score", C["high"] )]:
        keep = meta.state == label; km = KaplanMeierFitter(label=label); km.fit(df.loc[keep, "OS_time"], df.loc[keep, "OS_event"])
        ax.step(km.survival_function_.index, km.survival_function_[label], where="post", color=color, lw=1.5, label=label)
    ax.set_ylim(0, 1.05); ax.set_xlabel(""); ax.set_ylabel("Survival", fontsize=6.8); ax.set_title("Exploratory OS contrast\nlog-rank P=0.024", fontsize=8.1, pad=2); ax.legend(frameon=False, fontsize=5.2, loc="lower left"); ax.tick_params(labelsize=5.4); ax.tick_params(axis="x", which="both", labelbottom=False); panel(ax, "C")
    ax2 = fig.add_subplot(cgrid[1, 0], sharex=ax)
    for label, color in [("Lower-score", C["low"]), ("Higher-score", C["high"] )]:
        keep = meta.state == label; km = KaplanMeierFitter(label=label)
        km.fit(df.loc[keep, "OS_time"], df.loc[keep, "OS_event"])
        sf = km.survival_function_[label]
        ax2.step(sf.index, sf.values, where="post", color=color, lw=1.2)
    ax2.set_ylim(0, .62); ax2.set_ylabel("Survival", fontsize=6.8); ax2.set_xlabel("OS (months)", fontsize=6.8)
    ax2.tick_params(labelsize=5.4)
    ax2.spines["top"].set_visible(False)

    ax = fig.add_subplot(gs[1, 2]); ax.bxp([
        {"label":"Init.","whislo":init["low_ari"],"q1":init.get("q25_ari",init["low_ari"]),"med":init["median_ari"],"q3":init.get("q75_ari",init["high_ari"]),"whishi":init["high_ari"],"fliers":[]},
        {"label":"80% resample","whislo":patient["low_ari"],"q1":patient.get("q25_ari",patient["low_ari"]),"med":patient["median_ari_vs_full_solution"],"q3":patient.get("q75_ari",patient["high_ari"]),"whishi":patient["high_ari"],"fliers":[]},
    ], positions=[0, 1], widths=.38, patch_artist=True, boxprops={"facecolor":"#d7c9e8","edgecolor":C["purple"]}, medianprops={"color":C["navy"]}, whiskerprops={"color":C["purple"]}, capprops={"color":C["purple"]})
    ax.scatter([0], [init["median_ari"]], s=15, color=C["navy"], zorder=4)
    ax.axhline(.8, color=C["muted"], ls="--", lw=.7); ax.set_ylim(0,1.08); ax.set_ylabel("ARI", fontsize=7.2); ax.set_title("ARI stability by perturbation", pad=3, fontsize=8.8); ax.tick_params(axis="x", labelsize=5.8); panel(ax, "D")

    ax = fig.add_subplot(gs[1, 3]); kp = k_selection[k_selection.k >= 2]
    ax.bar(kp.k, kp.silhouette, color=[C["teal"] if k == 2 else C["grid"] for k in kp.k], width=.62)
    ga = ax.twinx(); ga.plot(kp.k, kp.gap, color=C["purple"], marker="o", lw=1.4, ms=3); ga.scatter([5], [kp.loc[kp.k==5,"gap"].iloc[0]], color=C["red"], marker="*", s=45)
    ax.set_xlabel("Number of clusters (k)", fontsize=6.8); ax.set_ylabel("Silhouette", fontsize=7.0); ga.set_ylabel("Gap", color=C["purple"], fontsize=7.0); ax.set_title("k-selection evidence", pad=3, fontsize=8.8); ax.tick_params(labelsize=5.8); ga.tick_params(labelsize=5.8); panel(ax, "E")
    save(fig, "Figure2_steroid_state_stability")


def figure3():
    consistency = base.load_json("model_consistency/model_consistency_summary.json")
    external = base.load_json("external_validation/GSE19750_locked_signature_validation.json")
    sensitivity = base.load_json("external_validation/GSE19750_locked_signature_sensitivity.json")
    interactions = base.load_json("signature_interactions/signature_interaction_summary.json")
    freq = pd.DataFrame(list(interactions["selection_frequency_oof"].items()), columns=["gene", "count"]).sort_values("count", ascending=False)
    coeff = pd.DataFrame(list(interactions["apparent_model_coefficients"].items()), columns=["gene", "coefficient"])
    followup = base.load_json("external_validation/GSE33371_locked_signature_validation.json")
    performance = pd.DataFrame([
        {"layer":"TCGA feature selection", "estimate":consistency["feature_selection_repeated_oof"]["cindex"], "low":consistency["feature_selection_repeated_oof"]["bootstrap"]["low_95"], "high":consistency["feature_selection_repeated_oof"]["bootstrap"]["high_95"]},
        {"layer":"TCGA fixed genes", "estimate":consistency["fixed_four_gene_repeated_oof"]["cindex"], "low":consistency["fixed_four_gene_repeated_oof"]["bootstrap"]["low_95"], "high":consistency["fixed_four_gene_repeated_oof"]["bootstrap"]["high_95"]},
        {"layer":"GSE19750 OS test", "estimate":external["cindex"], "low":external["cindex_bootstrap"]["low_95"], "high":external["cindex_bootstrap"]["high_95"]},
        {"layer":"GSE33371 follow-up", "estimate":followup["cindex"], "low":followup["cindex_bootstrap"]["low_95"], "high":followup["cindex_bootstrap"]["high_95"]},
    ])
    sens = pd.DataFrame(sensitivity["variants"])
    oof = pd.read_csv(RES / "signature_interactions" / "oof_predictions.tsv", sep="\t"); oof = oof.groupby("row_index", as_index=False)["risk"].mean()
    table = pd.read_csv(RES / "signature_interactions" / "tcga_locked_score_table.tsv", sep="\t"); oof["event"] = table.loc[oof.row_index, "OS_event"].to_numpy()
    fig = plt.figure(figsize=(12.8, 7.2)); gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.05, .9], hspace=.68, wspace=.44)
    ax = fig.add_subplot(gs[0, 0], polar=True); order = list(freq.head(10).gene)[::-1]; values = list(freq.head(10)["count"])[::-1]; angles = np.linspace(0, 2*np.pi, len(order), endpoint=False); width=2*np.pi/len(order)*.82
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.bar(angles, values, width=width, color=[C["red"] if g in LOCKED else C["muted"] for g in order], alpha=.88, edgecolor="white", linewidth=.5)
    ax.set_xticks(angles, [f"{g}\n{int(v)}/50" for g,v in zip(order,values)], fontsize=5.5); ax.set_yticks([10,25,40,50]); ax.set_ylim(0,53); ax.set_title("Resampling selection & model fitting", pad=12, fontsize=8.5); panel(ax, "A")
    ax = fig.add_subplot(gs[0, 1]); cf = coeff.set_index("gene").loc[LOCKED].reset_index(); yy=np.arange(len(cf)); ax.axvline(0,color=C["grid"]); ax.hlines(yy,0,cf.coefficient,color=C["teal"],lw=2); ax.scatter(cf.coefficient,yy,s=34,color=C["teal"]); ax.set_yticks(yy,cf.gene); ax.set_xlabel("Locked Cox coefficient"); ax.set_title("Locked coefficients & score sensitivity"); panel(ax,"B")
    sens2=sens.copy(); sens2["variant"]=sens2.variant.str.replace("frozen_weighted_mean_z","frozen weighted",regex=False).str.replace("equal_weight_mean_z","equal weight",regex=False).str.replace("rank_mean","rank mean",regex=False).str.replace("omit_","omit ",regex=False); sens2=sens2.sort_values("cindex"); inset=ax.inset_axes([.05,-.62,.9,.48]); inset.barh(np.arange(len(sens2)),sens2.cindex,color=[C["red"] if "frozen" in x else C["muted"] for x in sens2.variant]); inset.set_yticks(np.arange(len(sens2)),sens2.variant,fontsize=5); inset.set_xlim(.25,.78); inset.set_xlabel("GSE19750 C-index",fontsize=6); inset.tick_params(axis="x",labelsize=5)
    ax = fig.add_subplot(gs[1, 0]); colors=[C["blue"],C["teal"],C["red"],C["purple"]]
    for i,row in performance.iterrows(): ax.errorbar(i,row.estimate,yerr=[[row.estimate-row.low],[row.high-row.estimate]],fmt="o",color=colors[i],capsize=3,lw=1.2); ax.text(i,row.estimate+.03,f"{row.estimate:.3f}",ha="center",fontsize=6)
    ax.axhline(.5,color=C["muted"],ls="--",lw=.7); ax.set_ylim(.45,.9); ax.set_ylabel("C-index"); ax.set_xticks(range(4),["TCGA\nfeature", "TCGA\nfixed", "GSE19750", "GSE33371"]); ax.set_title("Model performance across objects"); panel(ax,"C")
    ax = fig.add_subplot(gs[1, 1]);
    for event,color,label in [(0,C["low"],"Censored"),(1,C["red"],"OS event")]:
        vals=oof.loc[oof.event==event,"risk"]; ax.hist(vals,bins=12,alpha=.58,color=color,label=label,density=True)
    ax.set_xlabel("OOF risk score"); ax.set_ylabel("Density"); ax.set_title("Held-out risk distribution (n=79)"); ax.legend(frameon=False,fontsize=6); panel(ax,"D")
    save(fig,"Figure3_internal_external_performance")


def figure4():
    rows = pd.DataFrame([
        {"modifier":"Continuous Weiss", "n":61, "events":24, "hr":.991941489, "p":.737108813},
        {"modifier":"Weiss grade >=6", "n":61, "events":24, "hr":.917097025, "p":.479307921},
        {"modifier":"Hormone status", "n":48, "events":23, "hr":np.nan, "p":np.nan},
    ])
    fig=plt.figure(figsize=(12.8,7.1)); gs=fig.add_gridspec(2,2,height_ratios=[1.12,.9],hspace=.68,wspace=.42)
    ax=fig.add_subplot(gs[0,0]); y=np.arange(3)[::-1]; ax.axvline(1,color=C["muted"],ls="--",lw=.8); ax.set_xlim(.75,1.3)
    for yi,(_,r) in zip(y,rows.iterrows()):
        if pd.notna(r.hr): ax.hlines(yi,1,r.hr,color=C["purple"],lw=2); ax.scatter(r.hr,yi,s=38,color=C["purple"]); ax.text(1.02,yi,f"LRT P={r.p:.3f}\nn={int(r.n)}, E={int(r.events)}",va="center",fontsize=6)
        else: ax.text(.78,yi,"not estimable\nmissing annotations",va="center",fontsize=6,color=C["red"])
    ax.set_yticks(y,rows.modifier); ax.set_xlabel("Interaction HR"); ax.set_title("Exploratory score-effect modification"); panel(ax,"A")
    ax=fig.add_subplot(gs[0,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title("Interaction analysis audit & classification workflow")
    stages=[("Raw clinical data","61 Weiss\n48 hormone\n0 verified C1A/C1B"),("Model status","2 estimable\n1 not estimable\nlabels not verified"),("Claim boundary","No interaction\nclaim supported")]
    for i,(title,body) in enumerate(stages):
        x=.02+i*.32; ax.add_patch(FancyBboxPatch((x,.28),.25,.42,boxstyle="round,pad=.01",facecolor="#edf2f7",edgecolor=C["muted"],lw=.7)); ax.text(x+.125,.59,title,ha="center",fontsize=6.4,fontweight="bold"); ax.text(x+.125,.42,body,ha="center",fontsize=6.2)
        if i<2: ax.annotate("",xy=(x+.30,.49),xytext=(x+.255,.49),arrowprops={"arrowstyle":"-|>","color":C["muted"]})
    panel(ax,"B")
    ax=fig.add_subplot(gs[1,0]); mat=np.array([[1,1,0],[1,1,0],[0,0,0]]); ax.imshow(mat,cmap=mpl.colors.ListedColormap(["#e8eef3",C["teal"]]),vmin=0,vmax=1,aspect="auto"); ax.set_xticks(range(3),["Weiss","Hormone","C1A/C1B"],rotation=28,ha="right",fontsize=6); ax.set_yticks(range(3),["Complete","Missing","Verified"],fontsize=6); ax.set_title("Interpretation boundaries & missingness map"); ax.text(0,2.45,"C1A/C1B labels were not inferred; hormone interaction not estimable.",fontsize=6.2,color=C["red"]); panel(ax,"C")
    ax=fig.add_subplot(gs[1,1]); x=np.array([.097,.0, .0]); y=np.array([.459,.737,.479]); sizes=np.array([61,48,61]); colors=[C["teal"],C["purple"],C["purple"]]; ax.scatter(x,y,s=sizes*2,c=colors,alpha=.72,edgecolor="white"); ax.axhline(.05,color=C["muted"],ls="--",lw=.7); ax.set_xlim(-.1,.55); ax.set_ylim(0,1); ax.set_xlabel("Association estimate / label status"); ax.set_ylabel("P value"); ax.set_xticks([0,.1,.5],["not estimable","Weiss rho","audit"]); ax.set_title("Exploratory interaction space"); panel(ax,"D")
    save(fig,"Figure4_interaction_audit")


def figure5():
    mapping=base.load_json("external_validation/mapping_audit.json")
    coverage=pd.DataFrame({"gene":list(mapping["gene_probe_coverage"]),"probes":list(mapping["gene_probe_coverage"].values())}); targets=pd.read_csv(RES/"druggability"/"candidate_target_prioritization.tsv",sep="\t"); targets["neglogp"]=-np.log10(targets.os_p.clip(lower=1e-300))
    fig=plt.figure(figsize=(12.8,7.0)); gs=fig.add_gridspec(2,2,height_ratios=[1.2,1],hspace=.65,wspace=.45)
    ax=fig.add_subplot(gs[:,0],polar=True); n=len(coverage); ang=np.linspace(0,2*np.pi,n,endpoint=False); bars=ax.bar(ang,coverage.probes,width=2*np.pi/n*.78,color=[C["teal"] if x>0 else C["navy"] for x in coverage.probes],alpha=.88,edgecolor="white",lw=.5); ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_xticks(ang,coverage.gene,fontsize=5.2); ax.set_ylim(0,max(3,coverage.probes.max()+1)); ax.set_title("GSE49278 native-platform coverage\n10 of 22 genes mapped",pad=16); panel(ax,"A")
    ax=fig.add_subplot(gs[0,1]); top=targets.head(10).sort_values("priority_score"); mat=top[["oof_selection_count","os_hr","dfs_hr","priority_score"]].to_numpy(); im=ax.imshow(mat,aspect="auto",cmap="YlGnBu"); ax.set_xticks(range(4),["OOF\nselection","OS HR","DFS HR","Priority"],fontsize=6); ax.set_yticks(range(len(top)),top.gene,fontsize=6); ax.set_title("Candidate prioritization matrix");
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,f"{mat[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color="white" if mat[i,j]>np.nanpercentile(mat,65) else C["navy"])
    panel(ax,"B")
    ax=fig.add_subplot(gs[1,1]); ax.scatter(targets.oof_selection_count,targets.neglogp,s=24,color=C["muted"],alpha=.7); sel=targets[targets.selected_locked]; ax.scatter(sel.oof_selection_count,sel.neglogp,s=48,color=C["red"],zorder=3)
    for _,r in sel.iterrows(): ax.text(r.oof_selection_count+.7,r.neglogp+.03,r.gene,fontsize=6)
    ax.set_xlabel("OOF selection frequency"); ax.set_ylabel("-log10 OS P"); ax.set_title("Stability and survival association"); ax.axvspan(23,52,color="#f7d8d8",alpha=.35); panel(ax,"C")
    save(fig,"Figure5_GSE49278_mapping")


def figure6():
    summary = json.loads((RES / "biology_associations" / "biology_association_summary.json").read_text())
    table = pd.read_csv(RES / "signature_interactions" / "tcga_locked_score_table.tsv", sep="\t")
    table["score"] = pd.to_numeric(table["locked_score"], errors="coerce")
    table["weiss"] = pd.to_numeric(table["WEISS_SCORE_OVERALL"], errors="coerce")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    table = table.merge(clinical[["patient_id", "AJCC_PATHOLOGIC_TUMOR_STAGE"]], on="patient_id", how="left", suffixes=("", "_clinical"))
    table["stage"] = table["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.extract(r"(IV|III|II|I)")[0].map({"I": 1, "II": 2, "III": 3, "IV": 4})
    hormone = []
    ci = summary["hormone_association"].get("group_median_bootstrap_95", {})
    for group, part in table.dropna(subset=["score", "HISTORY_ADRENAL_HORMONE_EXCESS"]).groupby("HISTORY_ADRENAL_HORMONE_EXCESS"):
        hormone.append({"group": group, "n": len(part), "median_score": part.score.median(), "ci_low": ci.get(group, [np.nan, np.nan])[0], "ci_high": ci.get(group, [np.nan, np.nan])[1]})
    hormone = pd.DataFrame(hormone).sort_values("median_score")
    weiss = table.dropna(subset=["score", "weiss"]).groupby("weiss").agg(n=("score", "size"), median_score=("score", "median"), q1=("score", lambda x: x.quantile(.25)), q3=("score", lambda x: x.quantile(.75))).reset_index()
    stage = table.dropna(subset=["score", "stage"]).groupby("stage").agg(n=("score", "size"), median_score=("score", "median"), q1=("score", lambda x: x.quantile(.25)), q3=("score", lambda x: x.quantile(.75))).reset_index()
    diff = pd.read_csv(RES / "biology_associations" / "program_gene_differential.tsv", sep="\t")
    path = pd.read_csv(RES / "biology_associations" / "program_pathway_scores.tsv", sep="\t")
    baseline = summary["baseline_models"]
    models = []
    for name, model in [("Stage only", baseline["stage_only"]), ("Stage + score", baseline["stage_plus_score"]), ("Weiss only", baseline["weiss_only"]), ("Weiss + score", baseline["weiss_plus_score"])]:
        key = "score" if "score" in name else list(model["hazard_ratios"])[0]
        models.append({"model": name, "term": key, "hr": model["hazard_ratios"][key], "p": model["p_values"][key]})

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.70, wspace=.42)
    ax = fig.add_subplot(gs[0, 0], polar=True)
    theta = np.linspace(0, 2*np.pi, len(hormone), endpoint=False)
    rr = hormone.median_score.to_numpy()
    ax.plot(np.r_[theta, theta[0]], np.r_[rr, rr[0]], color=C["purple"], lw=1.3)
    ax.fill(np.r_[theta, theta[0]], np.r_[rr, rr[0]], color=C["purple"], alpha=.16)
    ax.scatter(theta, rr, color=C["purple"], s=20, zorder=3)
    ax.set_xticks(theta, [str(x).replace("|", "/") for x in hormone.group], fontsize=5.2)
    ax.set_title("Hormone-secretion group profile", pad=12)
    ax.text(0, -0.20, f"Kruskal-Wallis P={summary['hormone_association']['p']:.3f}", transform=ax.transAxes, fontsize=6, color=C["muted"])
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    for _, row in weiss.iterrows():
        ax.fill_between([-.05, .05], row.weiss-.35, row.weiss+.35, alpha=.16, color=C["teal"])
        ax.scatter(row.median_score, row.weiss, s=max(12, row.n*2), color=C["teal"], edgecolor="white", linewidth=.4)
    ax.set_xlabel("Locked score"); ax.set_ylabel("Weiss grade")
    ax.set_title(f"Weiss-grade association\nSpearman rho={summary['weiss_association']['spearman_rho']:.3f}, P={summary['weiss_association']['p']:.3f}")
    panel(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    colors = [C["blue"], C["teal"], C["purple"], C["red"]]
    for x, (_, row) in enumerate(stage.iterrows()):
        vals = table.loc[table.stage == row.stage, "score"]
        vp = ax.violinplot(vals, positions=[row.stage], widths=.68, showextrema=False)
        for body in vp["bodies"]: body.set_facecolor(colors[x]); body.set_edgecolor("none"); body.set_alpha(.72)
        ax.scatter(np.full(len(vals), row.stage), vals, s=7, color=colors[x], alpha=.45)
    ax.set_xticks([1,2,3,4], ["I","II","III","IV"]); ax.set_xlabel("AJCC stage"); ax.set_ylabel("Locked score")
    ax.set_title(f"Score by AJCC stage\nSpearman rho={summary['stage_association']['spearman_rho']:.3f}, P<0.001")
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 0])
    show = diff.sort_values("median_difference_log2")
    ax.barh(np.arange(len(show)), show.median_difference_log2, color=[C["red"] if p < .05 else C["muted"] for p in show.fdr_bh], height=.72)
    ax.set_yticks(np.arange(len(show)), show.gene, fontsize=5.8); ax.set_xlabel("Higher-score minus lower-score median (log2)"); ax.set_title("Prespecified 22-gene differences")
    panel(ax, "D")

    ax = fig.add_subplot(gs[1, 1])
    ax.pie(path.median_difference.abs(), labels=[x.replace("_", " ") for x in path.pathway], colors=[C["blue"], C["gold"], C["purple"], C["teal"]], startangle=90, wedgeprops={"linewidth":.7, "edgecolor":"white"}, textprops={"fontsize":5.5})
    ax.set_title("Predefined pathway modules"); ax.text(.02, -.08, "No genome-wide enrichment", transform=ax.transAxes, fontsize=6, color=C["muted"]); panel(ax, "E")

    ax = fig.add_subplot(gs[1, 2])
    show = pd.DataFrame(models); y = np.arange(len(show))[::-1]
    ax.axvline(1, color=C["muted"], ls="--", lw=.7)
    for yi, (_, row) in zip(y, show.iterrows()):
        col = C["red"] if "score" in row.term else C["blue"]
        ax.hlines(yi, 1, row.hr, color=col, lw=2); ax.scatter(row.hr, yi, color=col, s=28); ax.text(max(row.hr + .04, 1.04), yi, f"HR {row.hr:.2f}; P={row.p:.3g}", va="center", fontsize=5.8)
    ax.set_yticks(y, show.model, fontsize=6); ax.set_xlabel("Hazard ratio"); ax.set_xlim(.8, 2.7); ax.set_title("Low-complexity prognostic comparison"); panel(ax, "F")
    save(fig, "Figure6_biological_associations")


def main():
    figure1(); figure2(); figure3(); figure4(); figure5(); figure6()
    print(json.dumps({"figure_dir":str(FIG),"source_dir":str(SRC)},indent=2))


if __name__ == "__main__":
    main()
