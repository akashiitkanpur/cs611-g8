"""Generate the report figures into figures/ (needs matplotlib)."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load
from formats import FORMATS, quantize
import predictor as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
W, X = load()
H = X @ X.T
names = list(FORMATS)
short = {n: n.replace("MXFP6_", "") for n in names}
COL = dict(zip(names, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
STY = dict(zip(names, ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight"})

def db(sig, err): return 10 * np.log10(sig / err)

# Fig 1: weight distribution vs Gaussian (log scale shows the tails)
fig, ax = plt.subplots(figsize=(6, 3.2))
w = W.ravel()
bins = np.linspace(-6, 6, 121)
ax.hist(w, bins=bins, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.3, density=True, label="W_C")
xs = np.linspace(-6, 6, 400)
ax.plot(xs, np.exp(-(xs - w.mean())**2 / (2 * w.var())) / np.sqrt(2 * np.pi * w.var()),
        color=INK, lw=1.5, ls="--", label="Gaussian, same mean/std")
ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
ax.set_xlabel("weight value"); ax.set_ylabel("density (log)")
k = ((w - w.mean())**4).mean() / w.var()**2
ax.set_title(f"W_C: std {w.std():.2f}, max |w| = {np.abs(w).max():.2f} ({np.abs(w).max()/w.std():.1f}σ), kurtosis {k:.2f}",
             fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8)
fig.savefig(f"{OUT}/fig1_weight_distribution.png"); plt.close(fig)

# Fig 2: error model g_f(u) per format (the core of the predictor)
fig, ax = plt.subplots(figsize=(6, 3.4))
u = np.geomspace(2**-8, 2, 4000)
for n in names:
    e = P.g(n, u, np.zeros_like(u))               # model: step^2/12, clipping, flush-to-zero
    rel = np.sqrt(e) / u * 100
    # BFP6 16|4|5 and MXINT6 have the same element grid relative to the block binade
    # (step 2^-4, max 31/16); they differ only through block size (16 vs 32).
    lab = "MXINT6 / BFP6 (same grid)" if n == "BFP6" else short[n]
    if n == "MXINT6":                              # identical curve to BFP6
        continue
    ax.plot(u, rel, color=COL[n], ls=STY[n], lw=2, label=lab)
    ax.annotate(lab, (u[-1], rel[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK, va="center")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("normalised magnitude u = |w| / 2^floor(log2 block max)")
ax.set_ylabel("RMS relative error (%)"); ax.set_xlim(right=4); ax.set_ylim(0.5, 150)
ax.axvspan(1.5, 2, color=GRID, alpha=0.6, lw=0)
ax.text(1.22, 0.62, "clipping\nzone", fontsize=7, color=MUTED, ha="right")
ax.set_title("Per-format error vs. position in the block (what the predictor integrates)",
             fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.savefig(f"{OUT}/fig2_error_model.png"); plt.close(fig)

# Fig 3: predicted vs exhaustive output SQNR on W_C
sig = ((W @ X)**2).sum()
ex = {n: db(sig, ((W @ X - quantize(W, n)[0] @ X)**2).sum()) for n in names}
pr = {n: db(sig, v) for n, v in P.predict(P.build_stats(W, H)).items()}
order = sorted(names, key=lambda n: -ex[n])
fig, ax = plt.subplots(figsize=(6, 3.2))
x = np.arange(len(order)); bw = 0.38
b1 = ax.bar(x - bw/2 - 0.01, [ex[n] for n in order], bw, color="#2a78d6", label="exhaustive (quantize + matmul)")
b2 = ax.bar(x + bw/2 + 0.01, [pr[n] for n in order], bw, color="#eb6834", label="predictor (histogram only)")
for b, v in zip(list(b1) + list(b2), [ex[n] for n in order] + [pr[n] for n in order]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x, [short[n] for n in order]); ax.set_ylabel("output SQNR on X_cal (dB)")
ax.set_ylim(0, max(ex.values()) + 4); ax.grid(axis="x", visible=False)
ax.set_title("Predicted vs. exhaustive output quality on W_C (higher is better)", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.savefig(f"{OUT}/fig3_predicted_vs_exhaustive.png"); plt.close(fig)

# Fig 4: sensitivity, output SQNR per format under controlled perturbations
rng = np.random.default_rng(0)
def pert():
    out = [("original", W), ("clip at 3σ", np.clip(W, -3, 3))]
    Wo = W.copy(); Wo.flat[rng.choice(W.size, 20, replace=False)] *= 50; out.append(("20 outliers ×50", Wo))
    out.append(("50% sparse", W * (rng.random(W.shape) > .5)))
    out.append(("90% sparse", W * (rng.random(W.shape) > .9)))
    out.append(("row scale 1e±2", W * 10.0 ** rng.uniform(-2, 2, (100, 1))))
    out.append(("uniform", rng.uniform(-1, 1, W.shape)))
    out.append(("Student-t ν=3", rng.standard_t(3, W.shape)))
    return out
cases = pert()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
y = np.arange(len(cases))
for n in names:
    vals = []
    for _, A in cases:
        s = ((A @ X)**2).sum(); vals.append(db(s, ((A @ X - quantize(A, n)[0] @ X)**2).sum()))
    ax.plot(vals, y, marker="o", ms=6, ls="none", color=COL[n], label=short[n],
            markeredgecolor="#fcfcfb", markeredgewidth=1)
ax.set_yticks(y, [c[0] for c in cases]); ax.invert_yaxis()
ax.set_xlabel("output SQNR (dB, higher is better)")
ax.set_title("Sensitivity: which format wins as the weight statistics change", fontsize=9, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.16))
fig.savefig(f"{OUT}/fig4_sensitivity.png"); plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))
