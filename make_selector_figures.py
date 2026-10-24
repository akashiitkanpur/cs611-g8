"""Figures for the selector section (Aryan).   python make_selector_figures.py
Reads results/selector_bench.csv (bench_selector.py) and results/selector_scaling.md
(bench_scaling.py); writes figures/fig6_selector_accuracy_vs_cost.png and
figures/fig7_selector_scaling.png."""
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# one colour per objective (<= 3 series in a scatter), reference palette slots 1-3
OBJ = {"weights only (way 1)": "#2a78d6", "H-weighted (way 2)": "#eb6834", "full Y on fit columns": "#1baf7a"}
LABEL = {"exhaustive_w": "brute force", "exhaustive_h": "brute force", "exhaustive_y": "brute force, full H",
         "sketch": "exact sketch", "sketch_h": "exact sketch", "sketch_nb1_h": "sketch 1 bin",
         "sketch_nb4_h": "4 bins", "sketch_nb16_h": "16 bins", "certified_is_h": "certified (IS)",
         "certified_h": "certified (uniform)", "sketch_s10_h": "sketch 10% sample", "closed_form": "closed form",
         "closed_form_h": "closed form", "akash_hist_h": "predictor.py", "m1": "M1", "m1_h": "M1",
         "hybrid_h": "hybrid", "exh_rows10": "brute 10% rows", "learned_tree": "learned tree"}


# label offsets (points) per panel, set by hand so no two labels collide
OFFS = {1: {"sketch_h": (-6, 5, "right"), "sketch_nb16_h": (-6, -2, "right"), "sketch_nb4_h": (6, -4, "left"),
            "exhaustive_h": (-6, 7, "right"), "certified_is_h": (6, 8, "left"), "akash_hist_h": (6, -6, "left"),
            "hybrid_h": (6, 4, "left"), "m1_h": (-6, 0, "right"), "sketch": (-6, -4, "right"),
            "sketch_nb1_h": (-6, 5, "right"), "exhaustive_w": (6, 3, "left"), "exhaustive_y": (-6, 4, "right")},
        2: {"sketch_h": (-6, -6, "right"), "exhaustive_h": (-6, -8, "right"), "certified_is_h": (6, -6, "left"),
            "akash_hist_h": (0, -10, "center"), "hybrid_h": (0, 9, "center"), "sketch_nb16_h": (-6, -4, "right"),
            "sketch_nb4_h": (-6, 4, "right"), "sketch": (-6, -5, "right"), "exhaustive_w": (6, -5, "left"),
            "learned_tree": (6, 4, "left"), "exh_rows10": (6, 4, "left")}}


def objective(a):
    if a == "exhaustive_y":
        return "full Y on fit columns"
    return "H-weighted (way 2)" if a.endswith("_h") else "weights only (way 1)"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig6():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "results", "selector_bench.csv"))) if r["suite"] == "random"]
    # closed_form and m1 without H sit on top of their H versions / of brute force: plot one of each
    names = [a for a in dict.fromkeys(r["approach"] for r in rows) if a not in ("closed_form", "m1")]
    stats = {}
    for a in names:
        A = [r for r in rows if r["approach"] == a]
        stats[a] = (1000 * np.median([float(r["seconds"]) for r in A]), 100 * np.mean([int(r["hit"]) for r in A]),
                    np.mean([float(r["regret_y_db"]) for r in A]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
    for ax, k, ylab in ((axs[0], 1, "top-1 agreement with held-out truth (%)"), (axs[1], 2, "mean regret (dB, log scale)")):
        style(ax)
        for a in names:
            x, y = stats[a][0], max(stats[a][k], 1e-3)
            ax.scatter(x, y, s=46, color=OBJ[objective(a)], edgecolor=SURF, linewidth=2, zorder=3)
            dx, dy, ha = OFFS[k].get(a, (6, 0, "left"))
            ax.annotate(LABEL[a], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK2,
                        ha=ha, va="center")
        ax.set_xscale("log")
        ax.set_xlabel("median time per 128x128 matrix (ms, log scale)", color=INK, fontsize=9)
        ax.set_ylabel(ylab, color=INK, fontsize=9)
        if k == 2:
            ax.set_yscale("log")
    axs[0].set_title("Accuracy vs cost: 1000 random matrices, 4 activation families", color=INK, fontsize=10, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l) for l, c in OBJ.items()]
    axs[1].legend(handles=handles, frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig6_selector_accuracy_vs_cost.png"), dpi=160, facecolor=SURF)


def fig7():
    path = os.path.join(HERE, "results", "selector_scaling.md")
    if not os.path.exists(path):
        return
    pts = []
    for line in open(path):
        m = re.match(r"\| (\d+)x\d+ \| ([\d,]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d,]+) of ([\d,]+)", line)
        if m:
            pts.append([float(m.group(2).replace(",", ""))] + [float(m.group(i)) for i in (3, 4, 5)]
                       + [float(m.group(6).replace(",", "")) / float(m.group(7).replace(",", ""))])
    P = np.array(pts)[1:]                       # drop the 128x128 warm-up row (kernel tables are built there)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURF)
    style(ax)
    ser = (("brute force (quantize all 5)", 1, "#2a78d6"), ("exact sketch", 2, "#eb6834"), ("certified, importance-sampled", 3, "#1baf7a"))
    for lab, i, c in ser:
        ax.plot(P[:, 0], P[:, i], color=c, lw=2, marker="o", markersize=6, markeredgecolor=SURF, markeredgewidth=2, label=lab)
    for x, y, f in zip(P[:, 0], P[:, 3], P[:, 4]):
        ax.annotate(f"{100 * f:.2g}%", (x, y), xytext=(0, -11), textcoords="offset points", fontsize=7,
                    color=INK2, ha="center")
    ax.text(0.98, 0.04, "% = share of blocks the certified selector rounded", transform=ax.transAxes,
            fontsize=7, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weights in the matrix (log scale)", color=INK, fontsize=9)
    ax.set_ylabel("selection time (s, log scale)", color=INK, fontsize=9)
    ax.set_title("Selection cost vs matrix size (LLM-like weights, H-weighted)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.set_xlim(P[0, 0] / 1.6, P[-1, 0] * 1.6)
    ax.set_ylim(P[:, 1:4].min() / 2.5, P[:, 1:4].max() * 2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig7_selector_scaling.png"), dpi=160, facecolor=SURF)


if __name__ == "__main__":
    fig6()
    fig7()
