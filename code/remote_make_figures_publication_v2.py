#!/usr/bin/env python3
"""Create information-dense, evidence-bound ACC figures on the remote server.

The script reads raw/processed inputs only on the server. It writes publication
figures plus aggregate source-data files; patient-level matrices and clinical
rows are not exported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from sklearn.cluster import KMeans

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
FIG = RES / "figures"
SRC = RES / "figure_data_export"
FIG.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
LOCKED = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]
C = {
    "blue": "#2468a8", "teal": "#168c8c", "red": "#c84d4d",
    "gold": "#d49b26", "purple": "#7251a3", "navy": "#243447",
    "muted": "#75859b", "grid": "#d9e2ec", "pale": "#f3f6f9",
    "low": "#63a8c6", "high": "#ca5555",
}
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 10,
    "axes.labelsize": 8.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=C["navy"], va="top")


def load_json(relative: str) -> dict:
    return json.loads((RES / relative).read_text())


def event_status(value: object) -> float:
    text = str(value).upper()
    if "DECEASED" in text or "1:" in text:
        return 1.0
    if "LIVING" in text or "0:" in text:
        return 0.0
    return np.nan


def load_tcga() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expr = pd.read_csv(PROC / "TCGA_ACC_steroid_genes.tsv.gz", sep="\t")
    clinical = pd.read_csv(PROC / "TCGA_ACC_clinical_selected.tsv", sep="\t")
    expr["patient_id"] = expr["sample_id"].str.rsplit("-", n=1).str[0]
    df = expr.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    values = np.log2(df[GENES].apply(pd.to_numeric, errors="coerce") + 1.0)
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    program = z.mean(axis=1)
    labels = KMeans(n_clusters=2, n_init=50, random_state=20261020).fit_predict(z)
    means = pd.Series(program).groupby(labels).mean()
    high_cluster = int(means.idxmax())
    state = pd.Series(np.where(labels == high_cluster, "Higher-score", "Lower-score"), index=df.index)
    df["OS_time"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["OS_event"] = df["OS_STATUS"].map(event_status)
    meta = pd.DataFrame({"state": state, "program_score": program}, index=df.index)
    return df, z, meta


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(SRC / name, sep="\t", index=False)


def export_manifest() -> None:
    files = []
    for p in sorted(SRC.iterdir()):
        if p.is_file() and p.name not in {"MANIFEST.json", "README.txt"}:
            files.append({"file": p.name, "bytes": p.stat().st_size,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {
        "project": "acc-steroid-metabolism",
        "source": "remote frozen result artifacts",
        "privacy_boundary": "aggregate figure data only; no raw matrix or patient-level clinical rows",
        "figures": "Figure1-Figure5 publication v2",
        "files": files,
    }
    (SRC / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (SRC / "README.txt").write_text(
        "Aggregate source data for Figures 1-5. Generated on the remote server from frozen result artifacts.\n"
        "The bundle contains plot-ready summaries and curve coordinates only; no raw expression matrix or patient-level clinical rows are included.\n"
    )


def figure1() -> None:
    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.54, wspace=0.34)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [
        (0.02, "DISCOVERY", "TCGA-ACC\nRNA-seq; OS/DSS; Weiss\n79 cases; 28 OS events", C["blue"]),
        (0.27, "PROGRAM", "22 frozen adrenal\nsteroid-metabolism genes\nwithin-cohort z-score", C["teal"]),
        (0.52, "MODEL", "k=2 state stability\nleakage-free OOF\n4-gene locked score", C["purple"]),
        (0.77, "EXTERNAL LAYERS", "GSE19750: OS test\nGSE33371: follow-up layer\nGSE49278: partial program", C["gold"]),
    ]
    for x, title, body, color in steps:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, 0.34), 0.19, 0.38,
                     boxstyle="round,pad=0.012,rounding_size=0.02",
                     facecolor="white", edgecolor=color, linewidth=2.2))
        ax.text(x + 0.095, 0.65, title, ha="center", va="center", fontweight="bold", color=color, fontsize=9)
        ax.text(x + 0.095, 0.49, body, ha="center", va="center", linespacing=1.35, fontsize=8)
        if x < 0.77:
            ax.annotate("", xy=(x + 0.225, 0.53), xytext=(x + 0.195, 0.53),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": C["muted"]})
    ax.text(0.5, 0.12, "EVIDENCE CEILING", ha="center", fontweight="bold", color=C["red"], fontsize=9)
    ax.text(0.5, 0.045, "internal state/model stability  |  directional external OS test  |  partial expression transportability  |  no efficacy claim",
            ha="center", color=C["muted"], fontsize=7.7)
    panel(ax, "A")

    cohorts = pd.DataFrame({"cohort": ["TCGA-ACC", "GSE19750", "GSE33371", "GSE49278"],
                            "n": [79, 21, 23, 42], "events": [28, 17, 16, 19]})
    write(cohorts.assign(endpoint=["OS", "OS", "OS", "OS"],
                         status=["discovery/internal OOF", "small directional test",
                                 "same biological samples as GSE10927; follow-up layer",
                                 "independent partial-program analysis"]),
          "Figure1_cohort_roles.tsv")
    ax = fig.add_subplot(gs[1, 0]); x = np.arange(4); width = 0.34
    ax.bar(x - width / 2, cohorts["n"], width, color=C["blue"], label="Cases")
    ax.bar(x + width / 2, cohorts["events"], width, color=C["red"], label="OS events")
    ax.set_xticks(x, cohorts["cohort"], rotation=28, ha="right"); ax.set_ylabel("Count")
    ax.set_title("Cohort size and endpoint audit"); ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.text(2, 26, "16 events", ha="center", color=C["red"], fontsize=7)
    ax.text(3, 46, "19 events", ha="center", color=C["red"], fontsize=7)
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    layers = ["TCGA\ninternal", "GSE19750\nOS test", "GSE33371\nfollow-up", "GSE49278\npartial OS"]
    vals = [1.0, 0.72, 0.62, 0.45]
    ax.barh(np.arange(4), vals, color=[C["blue"], C["teal"], C["purple"], C["gold"]], height=0.56)
    ax.set_yticks(np.arange(4), layers); ax.set_xlim(0, 1.05); ax.set_xlabel("Evidence completeness")
    ax.set_xticks([0, 0.5, 1], ["limited", "partial", "primary"]); ax.set_title("Evidence layers were not interchangeable")
    for i, value in enumerate(vals):
        label = ["discovery / OOF", "n=21; E=17", "n=23; E=16", "10/22 genes"][i]
        ax.text(value - 0.025, i, label, va="center", ha="right", fontsize=7,
                color="white" if value > 0.55 else C["navy"], fontweight="bold" if value > 0.55 else "normal")
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 2])
    items = ["Weiss grade", "Hormone status", "C1A/C1B"]
    counts = [61, 48, 0]
    ax.barh(np.arange(3), counts, color=[C["purple"], C["purple"], C["red"]], height=0.56)
    ax.set_yticks(np.arange(3), items); ax.set_xlabel("Verified complete cases"); ax.set_xlim(0, 86); ax.set_title("Clinical modifier audit")
    for i, value in enumerate(counts): ax.text(value + 2, i, "not verified" if value == 0 else str(value), va="center", fontsize=7)
    panel(ax, "D")
    save(fig, "Figure1_cohort_roles")


def figure2() -> None:
    df, z, meta = load_tcga()
    robustness = load_json("state_robustness/state_robustness_summary.json")
    tcga_robustness = robustness["tcga"]
    patient_stability = tcga_robustness["patient_resampling_stability"]
    init_stability = tcga_robustness["initialization_stability"]
    k_selection = pd.DataFrame(tcga_robustness["k_selection"]["table"])
    external_state = robustness["external_gse49278"]
    # Match the compact reference layout: a clustered heatmap with annotation
    # strips, state separation, OS curves, ARI perturbation, and k-selection.
    state_order = ["Lower-score", "Higher-score"]
    ordered_parts = []
    for state_name in state_order:
        state_idx = meta.index[meta["state"] == state_name].to_numpy()
        if len(state_idx) > 2:
            state_link = linkage(z.loc[state_idx, GENES].to_numpy(), method="average", metric="correlation", optimal_ordering=True)
            state_idx = state_idx[leaves_list(state_link)]
        ordered_parts.append(state_idx)
    order = np.concatenate(ordered_parts)
    heat = z.loc[order, GENES].T
    row_link = linkage(z.loc[:, GENES].to_numpy().T, method="average", metric="correlation", optimal_ordering=True)
    row_order = leaves_list(row_link)[::-1]
    heat = heat.iloc[row_order]
    states = meta.loc[order, "state"].map({"Lower-score": 0, "Higher-score": 1}).to_numpy()
    events = df.loc[order, "OS_event"].fillna(0).astype(int).to_numpy()
    stage_text = df.loc[order, "AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str)
    stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
    stages = stage_text.map(stage_map).fillna(0).astype(int).to_numpy()

    gene_order = pd.DataFrame({"row_order": np.arange(len(GENES)), "gene": heat.index})
    write(gene_order, "Figure2_heatmap_gene_order.tsv")
    write(pd.DataFrame({
        "annotation": ["state", "OS event", "AJCC stage"],
        "categories": ["Lower-score;Higher-score", "Censored;Event", "Missing;I;II;III;IV"],
        "n_observed": [int(len(states)), int(np.sum(pd.notna(df.loc[order, "OS_event"]))), int(np.sum(stages > 0))],
    }), "Figure2_heatmap_annotation_summary.tsv")

    fig = plt.figure(figsize=(13.0, 7.35))
    # Match the reference figure's hierarchy: a full-width heatmap hero panel
    # above four compact evidence panels.
    gs = fig.add_gridspec(2, 4, height_ratios=[1.62, 1.0], hspace=0.24, wspace=0.40)

    # Panel A: row-dendrogram heatmap and sample-level annotation strips.
    agrid = gs[0, :].subgridspec(3, 2, height_ratios=[0.55, 0.34, 5.0], width_ratios=[0.48, 7.0], hspace=0.03, wspace=0.025)
    ax_col_blank = fig.add_subplot(agrid[0, 1]); ax_col_blank.axis("off")
    ax_col_blank.text(0.5, 0.58, "Coordinated steroid-program expression across TCGA-ACC (n=79)",
                      ha="center", va="center", fontsize=9.5)
    ax_row_tree = fig.add_subplot(agrid[1:, 0])
    dendrogram(row_link, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=C["muted"], ax=ax_row_tree, link_color_func=lambda _: C["muted"])
    ax_row_tree.invert_yaxis(); ax_row_tree.axis("off")
    ax_ann = fig.add_subplot(agrid[1, 1])
    ann = np.vstack([states, events, stages])
    ann_cmap = mpl.colors.ListedColormap(["#f5f5f5", C["low"], C["high"], "#222222", "#d8d8d8", "#4c78a8", "#f58518", "#54a24b", "#e45756"])
    # Each annotation row uses a deliberately separate categorical palette.
    state_rgb = np.array([["#63a8c6", "#ca5555"][int(x)] for x in states])
    event_rgb = np.array([["#f2f2f2", "#222222"][int(x)] for x in events])
    stage_rgb = np.array([["#d8d8d8", "#4c78a8", "#f58518", "#54a24b", "#e45756"][int(x)] for x in stages])
    ann_rgb = np.stack([state_rgb, event_rgb, stage_rgb])
    ax_ann.imshow(np.array([[mpl.colors.to_rgb(v) for v in row] for row in ann_rgb]), aspect="auto", interpolation="nearest")
    ax_ann.set_yticks(np.arange(3), ["state", "OS event", "stage"], fontsize=6.8)
    ax_ann.set_xticks([]); ax_ann.tick_params(length=0); ax_ann.spines[:].set_visible(False)
    ax_heat = fig.add_subplot(agrid[2, 1])
    im = ax_heat.imshow(heat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(heat.index)), heat.index, fontsize=6.8)
    ax_heat.yaxis.tick_right(); ax_heat.yaxis.set_label_position("right")
    ax_heat.set_xticks([]); ax_heat.tick_params(axis="y", length=0, pad=2)
    split = len(ordered_parts[0]); ax_heat.axvline(split - 0.5, color="white", lw=1.0)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.20)
    cbar.set_label("Gene-wise z-score", fontsize=7); cbar.ax.tick_params(labelsize=6)
    panel(ax_heat, "A")

    state_summary = meta.assign(event=df["OS_event"].to_numpy()).groupby("state").agg(n=("state", "size"), events=("event", "sum"), mean_score=("program_score", "mean"), median_score=("program_score", "median")).reindex(state_order).reset_index()
    write(state_summary, "Figure2_state_summary.tsv")
    write(z.assign(state=meta["state"].to_numpy()).groupby("state")[GENES].mean().T.reset_index().rename(columns={"index": "gene"}), "Figure2_state_gene_means.tsv")
    robustness_table = pd.DataFrame([
        {"analysis": "random initialization", "estimate": init_stability["median_ari"], "low": init_stability["low_ari"], "high": init_stability["high_ari"], "q25": init_stability.get("q25_ari", init_stability["low_ari"]), "q75": init_stability.get("q75_ari", init_stability["high_ari"]), "n_runs": init_stability["n_runs"], "fraction_ari_at_least_0_8": np.nan},
        {"analysis": "80% patient resampling", "estimate": patient_stability["median_ari_vs_full_solution"], "low": patient_stability["low_ari"], "high": patient_stability["high_ari"], "q25": patient_stability.get("q25_ari", patient_stability["low_ari"]), "q75": patient_stability.get("q75_ari", patient_stability["high_ari"]), "n_runs": patient_stability["n_resamples"], "fraction_ari_at_least_0_8": patient_stability.get("fraction_ari_at_least_0_8", np.nan)},
    ])
    write(robustness_table, "Figure2_state_robustness.tsv")
    write(k_selection, "Figure2_k_selection_metrics.tsv")
    external_counts = []
    for state, counts in external_state["state_by_molecular_group_counts"].items():
        for molecular_group, count in counts.items():
            external_counts.append({"state": state, "molecular_group": molecular_group, "n": count})
    write(pd.DataFrame(external_counts), "Figure2_external_state_by_molecular_group.tsv")
    write(pd.DataFrame([{
        "cohort": external_state["cohort"], "mapped_gene_count": external_state["mapped_gene_count"],
        "n_expression_samples": external_state["n_expression_samples"],
        "molecular_group_n": external_state["molecular_group_n"],
        "molecular_group_chi_square_p": external_state["molecular_group_chi_square_p"],
        "molecular_group_cramers_v": external_state["molecular_group_cramers_v"],
        "age_state_mann_whitney_p": external_state["age_state_mann_whitney_p"],
        "gender_state_fisher_p": external_state["gender_state_fisher_p"],
    }]), "Figure2_external_state_summary.tsv")

    ax = fig.add_subplot(gs[1, 0])
    for label, color, pos in [("Lower-score", C["low"], 0), ("Higher-score", C["high"], 1)]:
        vals = meta.loc[meta["state"] == label, "program_score"]
        parts = ax.violinplot(vals, positions=[pos], widths=0.70, showmeans=False, showextrema=False, quantiles=[[0.25, 0.75]])
        for body in parts["bodies"]: body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(0.78)
        ax.scatter([pos] * len(vals), vals, s=8, color=color, alpha=0.45, edgecolors="none")
        ax.boxplot(vals, positions=[pos], widths=0.16, patch_artist=True, showfliers=False, boxprops={"facecolor": "white", "edgecolor": C["navy"], "linewidth": 0.7}, medianprops={"color": C["navy"], "linewidth": 1.0}, whiskerprops={"color": C["navy"], "linewidth": 0.7}, capprops={"color": C["navy"], "linewidth": 0.7})
    ax.set_xticks([0, 1], ["Lower-score\n(n=23; E=3)", "Higher-score\n(n=56; E=25)"])
    ax.set_ylabel("Steroid-program score"); ax.set_title("State separation"); panel(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    km_rows = []
    for label, color in [("Lower-score", C["low"]), ("Higher-score", C["high"] )]:
        keep = meta["state"] == label; km = KaplanMeierFitter(label=label)
        km.fit(df.loc[keep, "OS_time"], df.loc[keep, "OS_event"])
        curve = km.survival_function_.reset_index(); curve.columns = ["time", "survival"]; curve["state"] = label; km_rows.append(curve)
        ax.step(curve["time"], curve["survival"], where="post", color=color, lw=2.0, label=label)
    km_out = pd.concat(km_rows, ignore_index=True); write(km_out, "Figure2_km_curve_coordinates.tsv")
    ax.set_xlabel("Overall survival (months)"); ax.set_ylabel("Survival probability"); ax.set_ylim(0, 1.05)
    ax.set_title("Exploratory OS contrast\nlog-rank P=0.024"); ax.legend(frameon=False, fontsize=6.5, loc="lower left"); panel(ax, "C")

    # Show initialization stability and patient perturbation as distinct evidence.
    # The latter uses recorded quantiles rather than an invented raw distribution.
    ax = fig.add_subplot(gs[1, 2])
    init = init_stability
    patient = patient_stability
    ax.bxp([
        {"label": "Random\ninitialization", "whislo": init["low_ari"], "q1": init.get("q25_ari", init["low_ari"]), "med": init["median_ari"], "q3": init.get("q75_ari", init["high_ari"]), "whishi": init["high_ari"], "fliers": []},
        {"label": "80% patient\nresampling", "whislo": patient["low_ari"], "q1": patient.get("q25_ari", patient["low_ari"]), "med": patient["median_ari_vs_full_solution"], "q3": patient.get("q75_ari", patient["high_ari"]), "whishi": patient["high_ari"], "fliers": []},
    ], positions=[0, 1], widths=0.42, patch_artist=True,
        boxprops={"facecolor": "#d7c9e8", "edgecolor": C["purple"], "linewidth": 0.9},
        medianprops={"color": C["navy"], "linewidth": 1.3},
        whiskerprops={"color": C["purple"], "linewidth": 1.0},
        capprops={"color": C["purple"], "linewidth": 1.0})
    ax.axhline(0.8, color=C["muted"], ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["Random\ninitialization", "80% patient\nresampling"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("ARI vs full-data solution")
    ax.set_title("ARI stability\nunder perturbation", pad=8)
    ax.text(0, 1.01, f"median 1.00\n(n={init['n_runs']})", ha="center", va="bottom", fontsize=6.5)
    ax.text(1, patient["median_ari_vs_full_solution"] + 0.10, f"median {patient['median_ari_vs_full_solution']:.3f}\n{patient.get('fraction_ari_at_least_0_8', np.nan):.1%} >= 0.8", ha="center", va="bottom", fontsize=6.5)
    panel(ax, "D")

    ax = fig.add_subplot(gs[1, 3])
    k_plot = k_selection[k_selection["k"] >= 2]
    ax.bar(k_plot["k"], k_plot["silhouette"], color=[C["teal"] if k == 2 else C["grid"] for k in k_plot["k"]], width=0.62, label="Silhouette")
    gap_ax = ax.twinx()
    gap_ax.plot(k_plot["k"], k_plot["gap"], color=C["purple"], marker="o", lw=1.8, ms=4, label="Gap")
    gap_ax.scatter([5], [k_plot.loc[k_plot["k"] == 5, "gap"].iloc[0]], color=C["red"], marker="*", s=65, zorder=4)
    ax.set_xticks(k_plot["k"]); ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette")
    gap_ax.set_ylabel("Gap statistic", color=C["purple"])
    gap_ax.tick_params(axis="y", colors=C["purple"])
    ax.set_title("k-selection evidence")
    ax.text(0.02, 0.96, "silhouette: k=2", transform=ax.transAxes, va="top", fontsize=7, color=C["teal"], fontweight="bold")
    ax.text(0.02, 0.86, "Gap 1-SE: k=5", transform=ax.transAxes, va="top", fontsize=7, color=C["red"], fontweight="bold")
    panel(ax, "E")
    save(fig, "Figure2_steroid_state_stability")


def figure3() -> None:
    consistency = load_json("model_consistency/model_consistency_summary.json")
    external = load_json("external_validation/GSE19750_locked_signature_validation.json")
    sensitivity = load_json("external_validation/GSE19750_locked_signature_sensitivity.json")
    interactions = load_json("signature_interactions/signature_interaction_summary.json")
    freq = pd.DataFrame(list(interactions["selection_frequency_oof"].items()), columns=["gene", "count"]).sort_values("count", ascending=False)
    coeff = pd.DataFrame(list(interactions["apparent_model_coefficients"].items()), columns=["gene", "coefficient"])
    write(freq, "Figure3_selection_frequency.tsv"); write(coeff, "Figure3_locked_coefficients.tsv")
    followup = load_json("external_validation/GSE33371_locked_signature_validation.json")
    performance = pd.DataFrame([
        {"layer": "TCGA feature-selection repeated OOF", "estimate": consistency["feature_selection_repeated_oof"]["cindex"], "low": consistency["feature_selection_repeated_oof"]["bootstrap"]["low_95"], "high": consistency["feature_selection_repeated_oof"]["bootstrap"]["high_95"], "n": 79, "events": 28},
        {"layer": "TCGA fixed four-gene repeated OOF", "estimate": consistency["fixed_four_gene_repeated_oof"]["cindex"], "low": consistency["fixed_four_gene_repeated_oof"]["bootstrap"]["low_95"], "high": consistency["fixed_four_gene_repeated_oof"]["bootstrap"]["high_95"], "n": 79, "events": 28},
        {"layer": "GSE19750 OS test", "estimate": external["cindex"], "low": external["cindex_bootstrap"]["low_95"], "high": external["cindex_bootstrap"]["high_95"], "n": 21, "events": 17},
        {"layer": "GSE33371 follow-up", "estimate": followup["cindex"], "low": followup["cindex_bootstrap"]["low_95"], "high": followup["cindex_bootstrap"]["high_95"], "n": followup["n_acc_complete_positive_os"], "events": followup["events"]},
    ])
    sens = pd.DataFrame(sensitivity["variants"]); write(performance, "Figure3_performance.tsv"); write(sens.drop(columns=["bootstrap"]), "Figure3_score_sensitivity.tsv")
    oof = pd.read_csv(RES / "signature_interactions" / "oof_predictions.tsv", sep="\t")
    oof_mean = oof.groupby("row_index", as_index=False)["risk"].mean()
    table = pd.read_csv(RES / "signature_interactions" / "tcga_locked_score_table.tsv", sep="\t")
    oof_mean["event"] = table.loc[oof_mean["row_index"], "OS_event"].to_numpy()
    bins = np.linspace(oof_mean["risk"].min(), oof_mean["risk"].max(), 13)
    oof_mean["risk_bin"] = pd.cut(oof_mean["risk"], bins=bins, include_lowest=True)
    hist = oof_mean.groupby(["risk_bin", "event"], observed=False).size().reset_index(name="count")
    hist["risk_bin"] = hist["risk_bin"].astype(str); write(hist, "Figure3_oof_risk_histogram_bins.tsv")

    fig = plt.figure(figsize=(13.0, 8.4)); gs = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.36)
    ax = fig.add_subplot(gs[0, 0]); f = freq.head(10).sort_values("count")
    ax.barh(f["gene"], f["count"], color=[C["red"] if g in LOCKED else C["muted"] for g in f["gene"]])
    ax.axvline(50, color=C["grid"], lw=0.8, ls="--"); ax.set_xlabel("Selected folds (of 50)"); ax.set_title("Feature-selection stability")
    for i, v in enumerate(f["count"]): ax.text(v + 0.8, i, str(v), va="center", fontsize=7)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1]); coeff = coeff.set_index("gene").loc[LOCKED].reset_index(); yy = np.arange(len(coeff))
    ax.axvline(0, color=C["grid"], lw=1); ax.hlines(yy, 0, coeff["coefficient"], color=C["teal"], lw=2.5, alpha=0.72); ax.scatter(coeff["coefficient"], yy, s=62, color=C["teal"], zorder=3)
    ax.set_yticks(yy, coeff["gene"]); ax.set_xlabel("Locked Cox coefficient"); ax.set_title("Frozen four-gene score"); panel(ax, "B")

    ax = fig.add_subplot(gs[0, 2]); vals = performance["estimate"].to_numpy();
    colors = [C["blue"], C["teal"], C["red"], C["purple"]]
    for i, row in performance.iterrows():
        ax.errorbar([i], [row["estimate"]], yerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]], fmt="o", ms=7, color=colors[i], capsize=4, lw=1.5)
        ax.text(i, row["estimate"] + 0.025, f"{row['estimate']:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color=C["muted"], ls="--", lw=0.8); ax.set_xticks([0, 1, 2, 3], ["TCGA\nfeature\nselection", "TCGA\nfixed genes", "GSE19750\nOS test", "GSE33371\nfollow-up"]); ax.set_ylim(0.45, 0.9); ax.set_ylabel("C-index"); ax.set_title("Model-object comparison")
    ax.text(0.02, 0.04, "Intervals: patient-level bootstrap; external layers are directional.", transform=ax.transAxes, fontsize=6.5, color=C["muted"]); panel(ax, "C")

    ax = fig.add_subplot(gs[1, 0]); sens2 = sens.copy(); sens2["variant"] = sens2["variant"].str.replace("frozen_weighted_mean_z", "frozen weighted", regex=False).str.replace("equal_weight_mean_z", "equal weight", regex=False).str.replace("rank_mean", "rank mean", regex=False).str.replace("omit_", "omit ", regex=False); sens2 = sens2.sort_values("cindex")
    ax.barh(np.arange(len(sens2)), sens2["cindex"], color=[C["red"] if "frozen" in x else C["muted"] for x in sens2["variant"]]); ax.set_yticks(np.arange(len(sens2)), sens2["variant"], fontsize=7); ax.set_xlim(0.2, 0.78); ax.set_xlabel("GSE19750 C-index"); ax.set_title("Descriptive score sensitivity")
    for i, v in enumerate(sens2["cindex"]): ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    panel(ax, "D")

    ax = fig.add_subplot(gs[1, 1]);
    for event, color, label in [(0, C["low"], "censored"), (1, C["red"], "OS event")]:
        vals = oof_mean.loc[oof_mean["event"] == event, "risk"]; ax.hist(vals, bins=12, alpha=0.6, color=color, label=label, edgecolor="white")
    ax.set_xlabel("Mean held-out risk score"); ax.set_ylabel("Cases"); ax.set_title("OOF risk distribution\n79 unique cases"); ax.legend(frameon=False, fontsize=7); panel(ax, "E")

    ax = fig.add_subplot(gs[1, 2]); selected = freq[freq["gene"].isin(LOCKED)].sort_values("count"); ax.barh(selected["gene"], selected["count"], color=C["red"]); ax.set_xlim(0, 54); ax.set_xlabel("Selection frequency"); ax.set_title("Locked genes retained across folds")
    for i, v in enumerate(selected["count"]): ax.text(v + 0.8, i, f"{v}/50", va="center", fontsize=7)
    ax.text(0.02, 0.04, "Three genes were selected in every fold; AKR1C3 in 23/50.", transform=ax.transAxes, fontsize=7, color=C["muted"]); panel(ax, "F")
    save(fig, "Figure3_internal_external_performance")


def figure4() -> None:
    interactions = load_json("signature_interactions/signature_interaction_summary.json")["interactions"]
    rows = pd.DataFrame([
        {"modifier": "Continuous Weiss", "n": 61, "events": 24, "interaction_hr": 0.9919414890, "p_lrt": 0.7371088132, "status": "estimable"},
        {"modifier": "Weiss >=6", "n": 61, "events": 24, "interaction_hr": 0.9170970248, "p_lrt": 0.4793079211, "status": "estimable"},
        {"modifier": "Hormone status", "n": 48, "events": 23, "interaction_hr": np.nan, "p_lrt": np.nan, "status": "not estimable"},
        {"modifier": "C1A/C1B", "n": 0, "events": 0, "interaction_hr": np.nan, "p_lrt": np.nan, "status": "not verified"},
    ])
    write(rows, "Figure4_interactions.tsv")
    fig = plt.figure(figsize=(12.8, 6.8)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.9], hspace=0.6, wspace=0.4)
    ax = fig.add_subplot(gs[0, :]); y = np.arange(3)[::-1];
    est = rows.iloc[:3]; ax.axvline(1, color=C["muted"], ls="--", lw=0.9); ax.set_xscale("log"); ax.set_xlim(0.84, 1.06)
    for yi, (_, row) in zip(y, est.iterrows()):
        if pd.notna(row["interaction_hr"]):
            ax.hlines(yi, 0.85, row["interaction_hr"], color="#b9a9d2", lw=2); ax.scatter(row["interaction_hr"], yi, s=80, color=C["purple"], zorder=3); ax.text(1.012, yi, f"LRT P={row['p_lrt']:.3f}; n={int(row['n'])}, E={int(row['events'])}", va="center", fontsize=7)
        else: ax.text(1.012, yi, f"not estimable; n={int(row['n'])}, E={int(row['events'])}", va="center", fontsize=7, color=C["red"])
    ax.set_yticks(y, ["Continuous Weiss", "Weiss >=6", "Hormone status"]); ax.set_xlabel("Interaction hazard ratio (log scale)"); ax.set_title("Exploratory score-effect modification")
    panel(ax, "A")
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.03, 0.91, "What was tested", fontsize=10, fontweight="bold", color=C["navy"])
    text = [("Weiss grade", "continuous and >=6", "two estimable models"), ("Hormone status", "binary indicator", "48 complete; not estimable"), ("C1A/C1B", "sample-level class", "not verified; not modeled")]
    for i, (a, b, d) in enumerate(text):
        yy = 0.68 - i * 0.27; ax.add_patch(mpl.patches.FancyBboxPatch((0.03, yy - 0.08), 0.91, 0.15, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=C["pale"] if i % 2 == 0 else "white", edgecolor=C["grid"], lw=0.7)); ax.text(0.06, yy + 0.025, a, fontweight="bold", fontsize=8); ax.text(0.06, yy - 0.025, b, fontsize=7, color=C["muted"]); ax.text(0.94, yy, d, ha="right", fontsize=7, color=C["red"] if "not" in d else C["navy"])
    panel(ax, "B")
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.text(0.04, 0.90, "Interpretation boundary", fontsize=10, fontweight="bold", color=C["navy"])
    ax.text(0.04, 0.66, "No interaction was supported\nby these small complete-case analyses.", fontsize=9, color=C["navy"])
    ax.text(0.04, 0.39, "This is not evidence that the score is\nindependent of, or modified by, these features.", fontsize=8, color=C["muted"])
    ax.text(0.04, 0.12, "C1A/C1B labels were not inferred.", fontsize=8, color=C["red"]); panel(ax, "C")
    save(fig, "Figure4_interaction_audit")


def figure5() -> None:
    mapping = load_json("external_validation/mapping_audit.json")
    coverage = pd.DataFrame({"gene": list(mapping["gene_probe_coverage"]), "probes": list(mapping["gene_probe_coverage"].values())})
    coverage["mapped"] = coverage["probes"] > 0; write(coverage, "Figure5_probe_coverage.tsv")
    targets = pd.read_csv(RES / "druggability" / "candidate_target_prioritization.tsv", sep="\t")
    targets["neglogp"] = -np.log10(targets["os_p"].clip(lower=1e-300)); write(targets.drop(columns=["ensembl_ids", "opentargets_target_ids"], errors="ignore"), "Figure5_candidate_targets.tsv")
    top = targets.head(10).sort_values("priority_score")
    fig = plt.figure(figsize=(13.0, 8.2)); gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.36)
    ax = fig.add_subplot(gs[0, :]); ax.bar(np.arange(len(coverage)), coverage["probes"], color=[C["teal"] if x else "#dce3eb" for x in coverage["mapped"]], width=0.82); ax.set_xticks(np.arange(len(coverage)), coverage["gene"], rotation=62, ha="right", fontsize=7); ax.set_ylabel("Matched probes"); ax.set_title("GSE49278 native-platform coverage: 10 of 22 genes mapped"); ax.axhline(0, color=C["navy"], lw=0.7); panel(ax, "A")
    ax = fig.add_subplot(gs[1, 0]); ax.barh(np.arange(len(top)), top["priority_score"], color=[C["red"] if bool(x) else C["muted"] for x in top["selected_locked"]]); ax.set_yticks(np.arange(len(top)), top["gene"], fontsize=7); ax.set_xlabel("Candidate priority score"); ax.set_title("Candidate prioritization"); ax.text(0.02, 0.03, "red = locked-score genes", transform=ax.transAxes, fontsize=7, color=C["muted"]); panel(ax, "B")
    ax = fig.add_subplot(gs[1, 1]); ax.scatter(targets["oof_selection_count"], targets["neglogp"], s=32, color=C["muted"], alpha=0.72); selected = targets[targets["selected_locked"]]; ax.scatter(selected["oof_selection_count"], selected["neglogp"], s=64, color=C["red"], zorder=3); 
    for _, row in selected.iterrows(): ax.text(row["oof_selection_count"] + 0.7, row["neglogp"] + 0.03, row["gene"], fontsize=7)
    ax.set_xlabel("OOF selection frequency"); ax.set_ylabel("-log10 univariate OS P"); ax.set_title("Stability and survival association"); ax.text(0.02, 0.04, "Public target identifiers support prioritization only", transform=ax.transAxes, fontsize=7, color=C["muted"]); panel(ax, "C")
    save(fig, "Figure5_GSE49278_mapping")


def main() -> None:
    figure1(); figure2(); figure3(); figure4(); figure5(); export_manifest()
    print(json.dumps({"figure_dir": str(FIG), "source_dir": str(SRC), "source_files": sorted(p.name for p in SRC.iterdir() if p.is_file())}, indent=2))


if __name__ == "__main__":
    main()
