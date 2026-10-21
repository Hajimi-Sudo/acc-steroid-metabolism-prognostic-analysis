#!/usr/bin/env python3
"""Render Figure 5 as a compact quantitative result figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PAPER = PROJECT / "paper"
DATA = PAPER / "figure_data"
OUTPUT = PAPER / "Figure5_interaction_audit.jpg"

INK = "#1d2730"
NAVY = "#17324d"
TEAL = "#2d8c8c"
BLUE = "#365f91"
CORAL = "#a84a43"
PURPLE = "#72599a"
GRID = "#d7dee5"
MUTED = "#66727c"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "axes.titlesize": 9.4,
    "axes.labelsize": 8.4,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def panel(ax: plt.Axes, text: str) -> None:
    ax.text(-0.15, 1.12, text, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=INK, va="top", ha="left")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA / "Figure5_interaction_audit.tsv", sep="\t"),
        pd.read_csv(DATA / "Figure5_annotation_audit.tsv", sep="\t"),
    )


def render() -> None:
    interaction, audit = load_data()
    fig = plt.figure(figsize=(12.4, 7.15), dpi=320)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.68, 1.0], height_ratios=[1.0, 1.0],
        left=0.08, right=0.975, top=0.93, bottom=0.12,
        wspace=0.36, hspace=0.70,
    )

    # A. Main result: score-effect modification, displayed as a reference-style
    # horizontal estimate plot without assigning coordinates to unavailable terms.
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A")
    y = np.array([3, 2, 1, 0])
    ax.axvline(1.0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.85)
    for yi, (_, row), color in zip(y[:2], interaction.iloc[:2].iterrows(), [BLUE, CORAL]):
        hr = float(row["interaction_hr"])
        ax.hlines(yi, 0.84, hr, color=color, linewidth=2.3)
        ax.scatter(hr, yi, s=48, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(1.045, yi, f"HR {hr:.3f}, P={float(row['interaction_p']):.3f}",
                va="center", fontsize=7.5, color=INK)
    ax.text(1.045, y[2], "not estimable", va="center", fontsize=7.5,
            color=CORAL, fontweight="bold")
    ax.text(1.045, y[3], "not verified", va="center", fontsize=7.5,
            color=MUTED, fontweight="bold")
    ax.set_xlim(0.80, 1.30); ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(y, ["Continuous Weiss grade", "Weiss grade >= 6", "Hormone status", "C1A/C1B class"])
    ax.set_xlabel("Interaction hazard ratio")
    ax.set_title("Exploratory score-effect modification", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.tick_params(axis="y", length=0)
    ax.text(0.01, -0.25, "Null reference HR=1; Weiss models: n=61, 24 OS events",
            transform=ax.transAxes, fontsize=7.0, color=MUTED)

    # B. A compact point-and-error style summary of the clinical fields.
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "B")
    fields = ["Weiss grade", "Hormone status", "C1A/C1B"]
    available = audit["available_cases"].to_numpy(dtype=float)
    events = audit["events"].to_numpy(dtype=float)
    yy = np.arange(3)[::-1]
    ax.hlines(yy, 0, available, color=TEAL, linewidth=4, alpha=0.85)
    ax.scatter(available, yy, s=48, color=TEAL, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(events, yy, s=48, color=CORAL, edgecolor="white", linewidth=0.8, zorder=3)
    for yi, n, e in zip(yy, available, events):
        ax.text(n + 2.1, yi + 0.08, f"{int(n)}/79", color=TEAL, fontsize=7.3, va="center")
        ax.text(e + 2.1, yi - 0.16, f"{int(e)} events", color=CORAL, fontsize=7.1, va="center")
    ax.set_yticks(yy, fields); ax.set_xlim(0, 83); ax.set_ylim(-0.65, 2.65)
    ax.set_xlabel("Cases")
    ax.set_title("Clinical-field coverage", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=36, color=TEAL, label="Available")
    ax.scatter([], [], s=36, color=CORAL, label="OS events")
    ax.legend(frameon=False, fontsize=6.9, loc="lower right")

    # C. Reproduce the reference style with a compact P-value result panel.
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "C")
    pvals = interaction["interaction_p"].iloc[:2].to_numpy(dtype=float)
    neglogp = -np.log10(pvals)
    xx = np.arange(2)
    ax.axhline(-np.log10(0.05), color=MUTED, linestyle=(0, (3, 2)), linewidth=0.85)
    ax.vlines(xx, 0, neglogp, color=PURPLE, linewidth=2.2)
    ax.scatter(xx, neglogp, s=52, color=PURPLE, edgecolor="white", linewidth=0.8, zorder=3)
    for x, value, p in zip(xx, neglogp, pvals):
        ax.text(x, value + 0.07, f"P={p:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.text(1.01, -np.log10(0.05) + 0.04, "P=0.05", transform=ax.get_yaxis_transform(),
            fontsize=7.0, color=MUTED, va="bottom")
    ax.set_xticks(xx, ["Continuous\nWeiss", "Weiss >=6"])
    ax.set_ylabel("-log10 interaction P")
    ax.set_ylim(0, 1.55)
    ax.set_title("Interaction-test evidence", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.text(0.01, -0.25, "Both estimable tests were above the nominal 0.05 threshold",
            transform=ax.transAxes, fontsize=7.0, color=MUTED)

    # D. Result-style evidence density panel, using the same point language.
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "D")
    colors = [BLUE, CORAL, MUTED]
    ax.scatter(available, events, s=[130, 115, 95], c=colors,
               edgecolor="white", linewidth=0.9, zorder=3)
    annotations = ["Weiss\n61 cases, 24 events", "Hormone\n48 cases, 23 events", "C1A/C1B\n0 cases, 0 events"]
    offsets = [(2, 1.2), (2, -2.1), (2, 1.4)]
    for x, y, text, color, (dx, dy) in zip(available, events, annotations, colors, offsets):
        ax.text(x + dx, y + dy, text, fontsize=7.1, color=color, va="center")
    ax.set_xlim(-4, 78); ax.set_ylim(-4, 29)
    ax.set_xlabel("Available cases"); ax.set_ylabel("OS events")
    ax.set_title("Annotation and event support", loc="left", fontweight="bold", pad=7)
    ax.grid(color=GRID, linewidth=0.6)

    fig.savefig(OUTPUT, dpi=450, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 96, "subsampling": 0})
    plt.close(fig)


if __name__ == "__main__":
    render()
