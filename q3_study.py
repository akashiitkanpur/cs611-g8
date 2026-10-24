"""Q3: how outliers, sparsity and scale variation change the selection (dataset C, 6-bit).
Rework of Nikita's study (review: docs/REVIEW_Q3_REPORT.md; outlier artifact: issue #5).

    python q3_study.py            # about 2 min; results/q3_study.md, figures/fig8..fig10

Conventions: formats.py v2.1 (five candidates, BFP6 16|4|5, MXINT6 two's complement),
weights only, X FP32.  Metric (D6): relative Frobenius error of Y = W X on the held-out odd
columns of X_cal; both the class rule `ocp` and `opt`.  Every point is the mean of 10 seeds;
margins are given as mean +- sd of (runner-up - winner) in percentage points.  Each experiment
also records how often the selector (selector.sketch_select, H from the even columns) picks the
held-out winner, and mechanism counts from quantize(..., return_stats=True).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import selector as S              # noqa: E402
from formats import FORMATS, quantize, split_X   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = list(FORMATS)
SHORT = {"MXFP6_E3M2": "E3M2", "MXFP6_E2M3": "E2M3", "MXFP6_E4M1": "E4M1", "MXINT6": "MXINT6", "BFP6": "BFP6"}
COL = {"MXFP6_E2M3": "#2a78d6", "BFP6": "#eb6834", "MXINT6": "#1baf7a", "MXFP6_E3M2": "#eda100", "MXFP6_E4M1": "#e87ba4"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEEDS = 10

W0 = np.load(os.path.join(HERE, "data", "W_C.npy")).astype(np.float64)
X = np.load(os.path.join(HERE, "data", "X_cal.npy")).astype(np.float64)
XF, XE = split_X(X)
H = (XF ** 2).sum(1)
SIG = W0.std()


def evaluate(W, cols=None, rows=None):
    """Held-out Y error (%) per format and policy; cols / rows: also the error of the part of Y
    coming from those input columns / in those output rows."""
    out = {}
    for pol in ("ocp", "opt"):
        for n in NAMES:
            Wq, _, st = quantize(W, n, policy=pol, return_stats=True)
            D = W - Wq
            e = 100 * np.linalg.norm(D @ XE) / np.linalg.norm(W @ XE)
            sub = None
            if cols is not None:
                sub = 100 * np.linalg.norm(D[:, cols] @ XE[cols]) / np.linalg.norm(W[:, cols] @ XE[cols])
            if rows is not None:
                sub = 100 * np.linalg.norm(D[rows] @ XE) / np.linalg.norm(W[rows] @ XE)
            nbfp = st.get("n_scale_clamped", 0)
            out[n, pol] = (e, sub, st["n_zeroed"], st["n_range"], nbfp)
    sel = S.sketch_select(W, H)
    out["pick"] = min(sel, key=sel.get)
    return out


def summarize(trials, pol="ocp", key=0):
    E = {n: np.array([t[n, pol][key] for t in trials]) for n in NAMES}
    mean = {n: E[n].mean() for n in NAMES}
    order = sorted(NAMES, key=mean.get)
    d = E[order[1]] - E[order[0]]
    wins = [min(NAMES, key=lambda n: t[n, pol][key]) for t in trials]
    agree = np.mean([t["pick"] == min(NAMES, key=lambda n: t[n, "ocp"][0]) for t in trials])
    return dict(mean=mean, sd={n: E[n].std() for n in NAMES}, winner=order[0], second=order[1],
                margin=d.mean(), margin_sd=d.std(), seed_wins=wins.count(order[0]), agree=agree,
                zeroed={n: np.mean([t[n, pol][2] for t in trials]) for n in NAMES},
                clamped={n: np.mean([t[n, pol][4] for t in trials]) for n in NAMES},
                clipped={n: np.mean([t[n, pol][3] for t in trials]) for n in NAMES})


def row(label, s_ocp, s_opt, extra=""):
    f = lambda s: f"{SHORT[s['winner']]} ({s['seed_wins']}/{SEEDS}) +{s['margin']:.2f} ± {s['margin_sd']:.2f}"
    m = " | ".join(f"{s_ocp['mean'][n]:.2f}" for n in NAMES)
    return f"| {label} | {m} | {f(s_ocp)} | {f(s_opt)} | {100 * s_ocp['agree']:.0f}% |{extra}"


HEAD = ("| " + " | ".join(SHORT[n] for n in NAMES) + " | winner ocp (seeds), margin pp | winner opt | selector picks the held-out winner |")


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- outliers
def outliers(md):
    md += ["## 1. Outliers", "",
           "n weights of W_C replaced by ±k·σ (σ = std of W_C). Magnitudes are **continuous** (k·σ·U(0.7, 1.3)); "
           "the last block repeats Nikita's exact-magnitude protocol to show the grid artifact (issue #5). "
           "Columns E3M2…BFP6 are held-out Y error % under the class rule.", "",
           "| n, k, magnitudes | blocks with an outlier | " + HEAD[2:], "|" + "---|" * (len(NAMES) + 6)]
    fig_pts = []
    for exact in (False, True):
        for n in ((1, 5, 20, 100, 400) if not exact else (20, 100)):
            for k in ((5, 10, 20, 50, 100) if not exact else (20, 50)):
                trials, fb = [], []
                for s in range(SEEDS):
                    r = np.random.default_rng(1000 * n + k + s + (7 if exact else 0))
                    W = W0.copy()
                    idx = r.choice(W.size, n, replace=False)
                    W.flat[idx] = r.choice([-1, 1], n) * k * SIG * (1.0 if exact else r.uniform(0.7, 1.3, n))
                    rows_, cols_ = np.unravel_index(idx, W.shape)
                    fb.append(len(set(zip(rows_, np.minimum(cols_ // 32, 3)))) / 400)
                    trials.append(evaluate(W))
                so, sp = summarize(trials), summarize(trials, "opt")
                md.append(row(f"{n}, {k}σ, {'exact' if exact else 'continuous'} | {100 * np.mean(fb):.1f}%", so, sp))
                if not exact:
                    fig_pts.append((n, k, np.mean(fb), so))
    md += ["", "Mechanism, continuous magnitudes, n = 100, class rule: values flushed to zero by the quantizer / values above the "
           "format's range before rounding (mean per matrix):", ""]
    so = [p[3] for p in fig_pts if p[0] == 100 and p[1] == 20][0]
    md.append("| | " + " | ".join(SHORT[n] for n in NAMES) + " |")
    md.append("|---|" + "---|" * len(NAMES))
    md.append("| flushed to 0 | " + " | ".join(f"{so['zeroed'][n]:.0f}" for n in NAMES) + " |")
    md.append("| above range | " + " | ".join(f"{so['clipped'][n]:.0f}" for n in NAMES) + " |")
    md.append("")
    # figure: E3M2 - E2M3 and BFP6 - E2M3 gap vs fraction of blocks touched, per k
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor=SURF)
    style(ax)
    ks = (5, 20, 100)
    for k, c in zip(ks, ("#2a78d6", "#eb6834", "#1baf7a")):
        P = sorted([p for p in fig_pts if p[1] == k], key=lambda p: p[2])
        x = [100 * p[2] for p in P]
        ax.plot(x, [p[3]["mean"]["MXFP6_E3M2"] - p[3]["mean"]["MXFP6_E2M3"] for p in P], color=c, lw=2, marker="o",
                markersize=5, markeredgecolor=SURF, label=f"E3M2 − E2M3, outliers {k}σ")
        ax.plot(x, [p[3]["mean"]["BFP6"] - p[3]["mean"]["MXFP6_E2M3"] for p in P], color=c, lw=1.5, ls="--",
                marker="s", markersize=4, markeredgecolor=SURF, label=f"BFP6 − E2M3, outliers {k}σ")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("blocks containing an outlier (%, log scale)", fontsize=9, color=INK)
    ax.set_ylabel("error gap to E2M3 (pp; > 0: E2M3 better)", fontsize=9, color=INK)
    ax.set_title("Outliers (continuous magnitudes): E3M2 never wins; BFP6 does at 100σ", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig8_q3_outliers.png"), dpi=160, facecolor=SURF)


# --------------------------------------------------------------------------- sparsity
RHO = {}


def sparsity(md):
    md += ["## 2. Sparsity", "",
           "A fraction p of W_C's weights (uniform random positions) set to zero.", "",
           "| zeros | " + HEAD[2:], "|" + "---|" * (len(NAMES) + 4)]
    ps = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    curves = []
    for p in ps:
        trials, rho = [], []
        for s in range(SEEDS):
            r = np.random.default_rng(50000 + 100 * p + s)
            W = W0 * (r.random(W0.shape) >= p / 100)
            trials.append(evaluate(W))
            cf, e2 = S.closed_form(W), (W ** 2).sum()
            rho.append((cf["MXINT6"] * 12 * 4 ** 4 / e2, cf["BFP6"] * 12 * 4 ** 4 / e2, cf["MXFP6_E2M3"] * 12 * 4 ** 3 / e2))
        RHO[p] = np.mean(rho, 0)
        so, sp = summarize(trials), summarize(trials, "opt")
        md.append(row(f"{p}%", so, sp))
        curves.append((p, so))
    md += ["", "Mechanism (class rule, mean per matrix): share of the remaining non-zero weights that the quantizer "
           "flushes to zero, and the closed-form statistics: ρ = Σ n_nz·4^F / Σw² for 32- and 16-blocks "
           "(integer error ∝ ρ) and φ = Σ 4^floor(log2|w|) / Σw² (FP error ∝ φ):", "",
           "| zeros | " + " | ".join(SHORT[n] for n in NAMES) + " | ρ32 | ρ16 | φ |", "|---|" + "---|" * (len(NAMES) + 3)]
    for p, so in curves:
        if p in (0, 40, 50, 80, 95):
            nnz = W0.size * (1 - p / 100)
            md.append(f"| {p}% | " + " | ".join(f"{100 * so['zeroed'][n] / nnz:.1f}%" for n in NAMES)
                      + f" | {RHO[p][0]:.2f} | {RHO[p][1]:.2f} | {RHO[p][2]:.3f} |")
    md.append("")
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor=SURF)
    style(ax)
    for n in ("MXFP6_E2M3", "BFP6", "MXINT6", "MXFP6_E3M2"):
        m = np.array([so["mean"][n] for _, so in curves])
        sd = np.array([so["sd"][n] for _, so in curves])
        ax.plot(ps, m, color=COL[n], lw=2, label=SHORT[n])
        ax.fill_between(ps, m - sd, m + sd, color=COL[n], alpha=0.15, lw=0)
    ax.set_xlabel("weights set to zero (%)", fontsize=9, color=INK)
    ax.set_ylabel("held-out Y error (%, class rule)", fontsize=9, color=INK)
    ax.set_title("Sparsity: the integer grids overtake E2M3", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig9_q3_sparsity.png"), dpi=160, facecolor=SURF)


# --------------------------------------------------------------------------- scale variation
def scale(md):
    md += ["## 3. Variation in scale", "",
           "Gains 10^U(−a, a) applied (a) per **row** (one gain per output row: every block lies inside one row, so its "
           "scale absorbs the gain), (b) per **column** (input channel: gains change inside every block). "
           "'weak 25%' = error of the Y rows with the 25% smallest gains (row mode), or of the part of Y coming from the 25% "
           "of input columns with the smallest gains (column mode). Last column: BFP6 blocks whose 4-bit exponent "
           "clamps (out of 700).", "",
           "| spread | " + HEAD[2:] + " weak-25% error: " + " / ".join(SHORT[n] for n in NAMES) + " | weak-25% winner | BFP6 clamped blocks |",
           "|" + "---|" * (len(NAMES) + 7)]
    fig_rows = []
    for mode in ("row", "column"):
        for a in (0.5, 1, 1.5, 2):
            trials = []
            for s in range(SEEDS):
                r = np.random.default_rng(90000 + int(10 * a) * 10 + s + (5000 if mode == "column" else 0))
                if mode == "row":
                    g = 10 ** r.uniform(-a, a, 100)
                    trials.append(evaluate(W0 * g[:, None], rows=np.argsort(g)[:25]))
                else:
                    g = 10 ** r.uniform(-a, a, 100)
                    trials.append(evaluate(W0 * g[None, :], cols=np.argsort(g)[:25]))
            so, sp, sw = summarize(trials), summarize(trials, "opt"), summarize(trials, key=1)
            extra = (" " + " / ".join(f"{sw['mean'][n]:.1f}" for n in NAMES) + f" | {SHORT[sw['winner']]} | "
                     f"{so['clamped']['BFP6']:.0f} |")
            md.append(row(f"{mode} 10^±{a} (max/min 10^{2 * a:g})", so, sp, extra))
            fig_rows.append((mode, a, so, sw))
    md.append("")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF, sharey=False)
    for ax, key, title in ((axs[0], 2, "global held-out Y error"), (axs[1], 3, "error on the weakest 25% of input channels")):
        style(ax)
        for n in ("MXFP6_E2M3", "BFP6", "MXINT6", "MXFP6_E3M2", "MXFP6_E4M1"):
            P = [r_ for r_ in fig_rows if r_[0] == "column"]
            ax.plot([2 * p[1] for p in P], [p[key]["mean"][n] for p in P], color=COL[n], lw=2, marker="o",
                    markersize=5, markeredgecolor=SURF, label=SHORT[n])
        ax.set_xlabel("per-column gain spread, log10(max/min)", fontsize=9, color=INK)
        ax.set_ylabel("%", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "fig10_q3_scale.png"), dpi=160, facecolor=SURF)


def main():
    base = evaluate(W0)
    md = ["<!-- generated by q3_study.py; do not edit by hand -->",
          "# Q3 study: outliers, sparsity, variation in scale (dataset C, 6-bit)", "",
          "formats.py v2.1, weights only, held-out Y error (odd X_cal columns), mean of "
          f"{SEEDS} seeds. Margin = runner-up − winner in percentage points, mean ± sd over seeds; "
          "'(s/10)' = seeds in which that format won. Last column: how often the selector "
          "(exact sketch, H from the even columns) picks the held-out class-rule winner.", "",
          "Unmodified W_C, class rule: " + ", ".join(f"{SHORT[n]} {base[n, 'ocp'][0]:.3f}" for n in NAMES) +
          f"; selector picks {SHORT[base['pick']]}.", ""]
    outliers(md)
    sparsity(md)
    scale(md)
    with open(os.path.join(HERE, "results", "q3_study.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote results/q3_study.md and figures/fig8-fig10")


if __name__ == "__main__":
    main()
