#!/usr/bin/env python3
"""Generate publication-oriented multi-panel figures from frozen remote results.

All input matrices and patient-level tables are read and processed on the remote
server. The outputs are aggregate figures only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans

ROOT = Path(os.environ.get("ACC_PROJECT_ROOT", "/root/autodl-tmp/acc-steroid-metabolism"))
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GENES = [
    "AKR1C3", "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP19A1",
    "CYP21A2", "DHCR7", "HSD11B1", "HSD11B2", "HSD17B1", "HSD17B3",
    "HSD3B1", "HSD3B2", "LHCGR", "MC2R", "MRAP", "NR0B1", "NR5A1",
    "POR", "SCARB1", "STAR",
]
LOCKED = ["STAR", "NR0B1", "HSD11B2", "AKR1C3"]
COLORS = {
    "blue": "#2b6cb0", "teal": "#168c8c", "red": "#c94c4c",
    "gold": "#d69e2e", "purple": "#7251a3", "ink": "#243447",
    "muted": "#718096", "grid": "#d9e2ec", "light": "#f5f7fa",
}
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
    "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=COLORS["ink"], va="top")


def load_json(path: str) -> dict:
    return json.loads((RES / path).read_text())


def event_status(value: object) -> float:
    text = str(value).upper()
    if "DECEASED" in text or "1:" in text:
        return 1.0
    if "LIVING" in text or "0:" in text:
        return 0.0
    return np.nan


def load_tcga() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
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
    return df, z, pd.Series(program, index=df.index).rename("program_score").to_frame().assign(state=state)


def figure1() -> None:
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.95], hspace=0.5, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [
        (0.03, "Public ACC cohorts", "TCGA-ACC\nRNA-seq; OS/DSS; Weiss\n79 cases; 28 OS events", COLORS["blue"]),
        (0.27, "Frozen steroid program", "22 adrenal steroid\nmetabolism genes\nwithin-cohort z-score", COLORS["teal"]),
        (0.51, "State + score", "k=2 state stability\npatient-level OOF\n4-gene locked score", COLORS["purple"]),
        (0.75, "Evidence layers", "GSE19750: OS test\nGSE49278: expression only\nno external label tuning", COLORS["gold"]),
    ]
    for x, title, body, color in steps:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, 0.39), 0.19, 0.34,
                     boxstyle="round,pad=0.012,rounding_size=0.02",
                     facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x + 0.095, 0.65, title, ha="center", va="center", fontweight="bold", color=color)
        ax.text(x + 0.095, 0.51, body, ha="center", va="center", linespacing=1.35)
        if x < 0.75:
            ax.annotate("", xy=(x + 0.225, 0.56), xytext=(x + 0.195, 0.56),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": COLORS["muted"]})
    ax.text(0.5, 0.13, "Claim boundary", ha="center", fontweight="bold", color=COLORS["red"])
    ax.text(0.5, 0.055, "internal state/model stability  |  directional external OS test  |  partial expression transportability  |  no drug efficacy claim",
            ha="center", color=COLORS["muted"])
    panel(ax, "A")

    cohorts = pd.DataFrame({"cohort": ["TCGA-ACC", "GSE19750", "GSE49278"],
                            "n": [79, 21, 44], "events": [28, 17, np.nan]})
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(3); width = 0.34
    ax.bar(x - width / 2, cohorts["n"], width, color=COLORS["blue"], label="Cases")
    ax.bar(x + width / 2, cohorts["events"].fillna(0), width, color=COLORS["red"], label="OS events")
    ax.set_xticks(x, cohorts["cohort"], rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Cohort size and endpoint availability")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.text(2, 45, "OS not available", ha="center", color=COLORS["red"], fontsize=7)
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    layers = ["TCGA\ninternal", "GSE19750\nOS test", "GSE49278\nexpression"]
    vals = [1.0, 0.72, 0.45]
    ax.barh(np.arange(3), vals, color=[COLORS["blue"], COLORS["teal"], COLORS["gold"]], height=0.55)
    ax.set_yticks(np.arange(3), layers); ax.set_xlim(0, 1.05); ax.set_xlabel("Evidence completeness")
    ax.set_xticks([0, 0.5, 1], ["limited", "partial", "primary"])
    ax.set_title("Evidence layers were not interchangeable")
    for i, value in enumerate(vals):
        label = ["discovery/OOF", "n=21; E=17", "10/22 genes"][i]
        ax.text(value - 0.025, i, label, va="center", ha="right", fontsize=7,
                color="white" if value > 0.55 else COLORS["ink"], fontweight="bold" if value > 0.55 else "normal")
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 2])
    items = ["Weiss grade", "Hormone status", "C1A/C1B"]
    counts = [61, 79, 0]
    ax.barh(np.arange(3), counts, color=[COLORS["purple"], COLORS["purple"], COLORS["red"]], height=0.55)
    ax.set_yticks(np.arange(3), items); ax.set_xlabel("Verified complete cases")
    ax.set_xlim(0, 86); ax.set_title("Clinical modifier audit")
    for i, value in enumerate(counts): ax.text(value + 2, i, "not verified" if value == 0 else str(value), va="center", fontsize=7)
    panel(ax, "D")
    save(fig, "Figure1_cohort_roles")


def figure2() -> None:
    df, z, state = load_tcga()
    order = state.sort_values(["state", "program_score"], ascending=[True, True]).index
    heat = z.loc[order, GENES].T
    states = state.loc[order, "state"].map({"Lower-score": 0, "Higher-score": 1}).to_numpy()
    events = df.loc[order, "OS_event"].fillna(0).to_numpy()
    fig = plt.figure(figsize=(13, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], width_ratios=[1.8, 1], hspace=0.48, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(np.arange(len(GENES)), GENES, fontsize=7)
    ax.set_xticks([]); ax.set_ylabel("Steroid-program genes")
    ax.set_title("Coordinated steroid-program expression across TCGA-ACC")
    ax.axvline(np.where(states == 1)[0][0] - 0.5, color="white", lw=1.2)
    ax2 = ax.inset_axes([0, -0.11, 1, 0.055])
    ax2.imshow(states[np.newaxis, :], aspect="auto", cmap=mpl.colors.ListedColormap(["#9ecae1", COLORS["red"]]), vmin=0, vmax=1)
    ax2.set_xticks([]); ax2.set_yticks([0], ["state"]); ax2.tick_params(length=0, labelsize=7)
    ax3 = ax.inset_axes([0, -0.18, 1, 0.055])
    ax3.imshow(events[np.newaxis, :], aspect="auto", cmap=mpl.colors.ListedColormap(["#f0f0f0", "#111111"]), vmin=0, vmax=1)
    ax3.set_xticks([]); ax3.set_yticks([0], ["OS event"]); ax3.tick_params(length=0, labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.02, label="Gene-wise z-score")
    panel(ax, "A")

    ax = fig.add_subplot(gs[1, 0])
    for label, color in [("Lower-score", "#5fa6c4"), ("Higher-score", COLORS["red"])]:
        vals = state.loc[state["state"] == label, "program_score"]
        parts = ax.violinplot(vals, positions=[0 if label == "Lower-score" else 1], widths=0.72, showmeans=True, showextrema=False)
        for body in parts["bodies"]: body.set_facecolor(color); body.set_edgecolor("none"); body.set_alpha(0.75)
        ax.scatter([0 if label == "Lower-score" else 1] * len(vals), vals, s=8, color=color, alpha=0.45, edgecolors="none")
    ax.set_xticks([0, 1], ["Lower-score\n(n=23)", "Higher-score\n(n=56)"])
    ax.set_ylabel("Steroid-program score")
    ax.set_title("State separation")
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    for label, color in [("Lower-score", "#5fa6c4"), ("Higher-score", COLORS["red"])]:
        keep = state["state"] == label
        km = KaplanMeierFitter(label=label)
        km.fit(df.loc[keep, "OS_time"], df.loc[keep, "OS_event"])
        ax.step(km.survival_function_.index, km.survival_function_[label], where="post", color=color, lw=2, label=label)
    ax.set_xlabel("Overall survival (months)"); ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05); ax.set_title("Exploratory OS contrast; log-rank P=0.024")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    panel(ax, "C")
    save(fig, "Figure2_steroid_state_stability")


def figure3() -> None:
    robust = load_json("robustness/leakage_free_robustness_summary.json")
    external = load_json("external_validation/GSE19750_locked_signature_validation.json")
    sensitivity = load_json("external_validation/GSE19750_locked_signature_sensitivity.json")
    interactions = load_json("signature_interactions/signature_interaction_summary.json")
    freq = pd.read_csv(RES / "signature_interactions" / "gene_selection_frequency.tsv", sep="\t")
    freq = freq["gene"].value_counts().rename_axis("gene").reset_index(name="count")
    coeff = interactions["apparent_model_coefficients"]
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.33)
    ax = fig.add_subplot(gs[0, 0])
    f = freq.head(10).sort_values("count")
    ax.barh(f["gene"], f["count"], color=[COLORS["red"] if g in LOCKED else COLORS["muted"] for g in f["gene"]])
    ax.axvline(50, color=COLORS["grid"], lw=0.8, ls="--")
    ax.set_xlabel("Selection frequency across 50 folds"); ax.set_title("Gene-selection stability")
    for i, v in enumerate(f["count"]): ax.text(v + 0.8, i, str(v), va="center", fontsize=7)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    genes = list(coeff); vals = [coeff[g] for g in genes]
    yy = np.arange(len(genes))
    ax.axvline(0, color=COLORS["grid"], lw=1)
    ax.scatter(vals, yy, s=55, color=COLORS["teal"], zorder=3)
    for y, value in zip(yy, vals): ax.plot([0, value], [y, y], color=COLORS["teal"], lw=2, alpha=0.65)
    ax.set_yticks(yy, genes); ax.set_xlabel("Apparent Cox coefficient")
    ax.set_title("Locked four-gene score coefficients")
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    names = ["TCGA\nOOF", "GSE19750\nOS test"]
    vals = [robust["leakage_free_repeated_nested_like_oof"]["cindex"], external["cindex"]]
    ax.errorbar([1], [vals[1]], yerr=[[vals[1] - external["cindex_bootstrap"]["low_95"]], [external["cindex_bootstrap"]["high_95"] - vals[1]]], fmt="o", ms=8, color=COLORS["red"], capsize=5, lw=1.7)
    ax.scatter([0], [vals[0]], s=75, color=COLORS["blue"], zorder=3)
    ax.axhline(0.5, color=COLORS["muted"], ls="--", lw=0.8)
    ax.set_xticks([0, 1], names); ax.set_ylim(0.45, 0.88); ax.set_ylabel("C-index")
    ax.set_title("Performance by evidence layer")
    ax.text(0, vals[0] + 0.025, f"{vals[0]:.3f}", ha="center", fontsize=8)
    ax.text(1, vals[1] + 0.025, f"{vals[1]:.3f}", ha="center", fontsize=8)
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    sens = pd.DataFrame(sensitivity["variants"])
    sens["variant"] = sens["variant"].str.replace("frozen_weighted_mean_z", "frozen weighted", regex=False).str.replace("equal_weight_mean_z", "equal weight", regex=False).str.replace("rank_mean", "rank mean", regex=False).str.replace("omit_", "omit ", regex=False)
    sens = sens.sort_values("cindex")
    ax.barh(np.arange(len(sens)), sens["cindex"], color=[COLORS["red"] if "frozen" in x else COLORS["muted"] for x in sens["variant"]])
    ax.set_yticks(np.arange(len(sens)), sens["variant"], fontsize=7); ax.set_xlim(0.2, 0.78); ax.set_xlabel("GSE19750 C-index")
    ax.set_title("Descriptive score sensitivity")
    for i, v in enumerate(sens["cindex"]): ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    panel(ax, "D")
    save(fig, "Figure3_internal_external_performance")


def figure4() -> None:
    interactions = load_json("signature_interactions/signature_interaction_summary.json")["interactions"]
    names = ["Continuous Weiss", "Weiss >=6", "Hormone status"]
    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    estimable = [x.get("interaction_hr") is not None for x in interactions]
    hrs = [x.get("interaction_hr", np.nan) for x in interactions]
    ps = [x.get("interaction_p_lrt") for x in interactions]
    ns = [x["n"] for x in interactions]; events = [x["events"] for x in interactions]
    y = np.arange(3)[::-1]
    ax.axvline(1, color=COLORS["muted"], ls="--", lw=0.9)
    ax.scatter([hr if ok else np.nan for hr, ok in zip(hrs, estimable)], y, s=75, color=COLORS["purple"], zorder=3)
    for yi, hr, ok in zip(y, hrs, estimable):
        if ok:
            ax.plot([0.85, hr], [yi, yi], color=COLORS["purple"], alpha=0.45, lw=1.4)
    ax.set_yticks(y, names); ax.set_xlim(0.84, 1.03); ax.set_xlabel("Interaction hazard ratio")
    ax.set_title("No evidence for score-effect modification")
    for yi, hr, p, n, e, ok in zip(y, hrs, ps, ns, events, estimable):
        label = f"P={p:.3f}; n={n}, E={e}" if ok else f"not estimable; n={n}, E={e}"
        ax.text(1.005, yi, label, va="center", fontsize=7, color=COLORS["ink"] if ok else COLORS["red"])
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = [("Weiss grade", "61 complete cases", "24 events", "P=0.737"), ("Weiss >=6", "61 complete cases", "24 events", "P=0.479"), ("Hormone status", "48 complete cases", "23 events", "not estimable"), ("C1A/C1B", "not verified", "not modeled", "blocked")]
    ax.text(0.03, 0.94, "Modifier audit", fontsize=10, fontweight="bold", color=COLORS["ink"])
    ax.text(0.03, 0.86, "modifier", fontsize=7, color=COLORS["muted"])
    ax.text(0.55, 0.86, "complete", fontsize=7, color=COLORS["muted"])
    ax.text(0.72, 0.86, "events", fontsize=7, color=COLORS["muted"])
    ax.text(0.91, 0.86, "LRT P", fontsize=7, color=COLORS["muted"])
    for i, (a, b, c, d) in enumerate(rows):
        y0 = 0.78 - i * 0.18
        ax.add_patch(mpl.patches.Rectangle((0.02, y0 - 0.075), 0.96, 0.13, facecolor=COLORS["light"] if i % 2 == 0 else "white", edgecolor="none"))
        short_b = {"61 complete cases": "n=61", "48 complete cases": "n=48", "not verified": "unavailable"}[b]
        short_c = {"24 events": "E=24", "23 events": "E=23", "28 events": "E=28", "not modeled": "--"}[c]
        short_d = d.replace("P=", "") if d != "blocked" else d
        ax.text(0.04, y0, a, fontweight="bold", va="center", fontsize=7.2)
        ax.text(0.55, y0, short_b, va="center", ha="center", fontsize=7)
        ax.text(0.72, y0, short_c, va="center", ha="center", fontsize=7)
        ax.text(0.91, y0, short_d, va="center", ha="center", fontsize=7, color=COLORS["red"] if d == "blocked" else COLORS["ink"])
    ax.text(0.03, 0.08, "Complete-case interaction analyses were exploratory and\nwere not used to change the locked score.", color=COLORS["muted"], fontsize=7)
    panel(ax, "B")
    save(fig, "Figure4_interaction_audit")


def figure5() -> None:
    mapping = load_json("external_validation/mapping_audit.json")
    coverage = pd.DataFrame({"gene": list(mapping["gene_probe_coverage"]), "probes": list(mapping["gene_probe_coverage"].values())})
    coverage["mapped"] = coverage["probes"] > 0
    targets = pd.read_csv(RES / "druggability" / "candidate_target_prioritization.tsv", sep="\t").head(10).sort_values("priority_score")
    fig = plt.figure(figsize=(12.2, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    mat = coverage["mapped"].astype(int).to_numpy()[np.newaxis, :]
    ax.imshow(mat, aspect="auto", cmap=mpl.colors.ListedColormap(["#e6eaf0", COLORS["teal"]]), vmin=0, vmax=1)
    ax.set_yticks([0], ["GPL16686"]); ax.set_xticks(np.arange(len(coverage)), coverage["gene"], rotation=65, ha="right", fontsize=7)
    ax.set_title("GSE49278 native-platform coverage: 10 of 22 genes mapped")
    for x, mapped in enumerate(coverage["mapped"]): ax.text(x, 0, "yes" if mapped else "no", ha="center", va="center", fontsize=6, color="white" if mapped else COLORS["muted"])
    panel(ax, "A")

    ax = fig.add_subplot(gs[1, 0])
    ax.barh(np.arange(len(targets)), targets["priority_score"], color=[COLORS["red"] if bool(x) else COLORS["muted"] for x in targets["selected_locked"]])
    ax.set_yticks(np.arange(len(targets)), targets["gene"], fontsize=7); ax.set_xlabel("Candidate priority score")
    ax.set_title("Top candidate genes")
    ax.text(0.02, 0.03, "red = locked-score genes", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 1])
    all_targets = pd.read_csv(RES / "druggability" / "candidate_target_prioritization.tsv", sep="\t")
    all_targets["neglogp"] = -np.log10(all_targets["os_p"].clip(lower=1e-300))
    ax.scatter(all_targets["oof_selection_count"], all_targets["neglogp"], s=35, color=COLORS["muted"], alpha=0.75)
    selected = all_targets[all_targets["selected_locked"]]
    ax.scatter(selected["oof_selection_count"], selected["neglogp"], s=60, color=COLORS["red"], zorder=3)
    for _, row in selected.iterrows(): ax.text(row["oof_selection_count"] + 0.8, row["neglogp"] + 0.02, row["gene"], fontsize=7)
    ax.set_xlabel("OOF selection frequency"); ax.set_ylabel("-log10 univariate OS P")
    ax.set_title("Selection stability and survival association")
    ax.text(0.02, 0.03, "Public target identifiers support prioritization only", transform=ax.transAxes, fontsize=7, color=COLORS["muted"])
    panel(ax, "C")
    save(fig, "Figure5_GSE49278_mapping")


def main() -> None:
    figure1(); figure2(); figure3(); figure4(); figure5()
    print(json.dumps({"figure_dir": str(OUT), "figures": sorted(p.name for p in OUT.glob("Figure*.png"))}, indent=2))


if __name__ == "__main__":
    main()
