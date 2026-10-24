"""Permanent test matrices for the selectors (Aryan).  Everything is seeded and regenerated.

  datasets()       W_A ... W_E with the course X_cal (needs assignment_data.npz)
  adversarial()    28 hand-made families: heavy tails, sparsity, outliers (continuous
                   magnitudes, see issue #5), wide dynamic range, few-level alphabets, pre-quantized
                   data, column spread, empty rows; X white Gaussian
  random_suite(n)  n random matrices mixing a base distribution, sparsity, random / channel /
                   block outliers, row and column gains, rescaled by a power of two to max|W| in [8, 16); each gets a random X family:
                   white, channel gains (log-normal), a few outlier channels, or low rank
                   (correlated) - the last three are where "way 2" (H-weighting) can matter.
Each generator yields (name, W, X) with X of shape (K, 256): the even columns fit H, the odd
columns are held out (D6).
"""
import os

import numpy as np

SHAPE = (128, 128)
NX = 256


def _x(rng, k, kind):
    X = rng.standard_normal((k, NX))
    if kind == "white":
        return X
    if kind == "chan_gain":
        return X * np.exp(rng.normal(0, 1.0, (k, 1)))
    if kind == "outlier_chan":
        g = np.ones((k, 1))
        g[rng.choice(k, max(1, k // 32), replace=False)] = 10 ** rng.uniform(1, 2)
        return X * g
    if kind == "lowrank":
        r = int(rng.integers(1, 6))
        return rng.standard_normal((k, r)) @ rng.standard_normal((r, NX)) + 0.1 * X
    raise ValueError(kind)


def find_npz(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (path, os.environ.get("CS611_DATA"), os.path.join(here, "data", "assignment_data.npz")):
        if p and os.path.exists(p):
            return p
    return None


def datasets(npz=None):
    p = find_npz(npz)
    if not p:
        return
    with np.load(p) as d:
        X = d["X_cal"].astype(np.float64)
        for k in "ABCDE":
            yield "dataset " + k, d["W_" + k].astype(np.float64), X


def adversarial(seed=1):
    rng = np.random.default_rng(seed)
    sh = SHAPE
    def outl(n, k):
        W = rng.standard_normal(sh)
        i = rng.choice(W.size, n, replace=False)
        W.flat[i] = rng.choice([-1, 1], n) * k * rng.uniform(0.7, 1.3, n)   # continuous (#5)
        return W
    def onepb():
        W = rng.standard_normal(sh) * 0.3
        W[:, ::32] = 8.0 * rng.uniform(0.7, 1.3, (sh[0], 1)) * rng.choice([-1, 1], (sh[0], 1))
        return W
    def zero_rows():
        W = rng.standard_normal(sh)
        W[rng.random(sh[0]) < 0.5] = 0
        return W
    def pre8():
        W = rng.standard_normal(sh)
        s = np.abs(W).max() / 127
        return np.round(W / s) * s
    sgn = lambda: rng.choice([-1, 1], sh)
    fam = {
        "gauss": lambda: rng.standard_normal(sh),
        "laplace": lambda: rng.laplace(size=sh),
        "student-t3": lambda: rng.standard_t(3, sh),
        "uniform": lambda: rng.uniform(-1, 1, sh),
        "sparse50": lambda: rng.standard_normal(sh) * (rng.random(sh) > 0.5),
        "sparse80": lambda: rng.standard_normal(sh) * (rng.random(sh) > 0.8),
        "sparse95": lambda: rng.standard_normal(sh) * (rng.random(sh) > 0.95),
        "outl 20x20": lambda: outl(20, 20), "outl 20x50": lambda: outl(20, 50),
        "outl 200x20": lambda: outl(200, 20), "outl 1/block x8": onepb,
        "loguni 4": lambda: sgn() * 2.0 ** rng.uniform(-4, 0, sh),
        "loguni 8": lambda: sgn() * 2.0 ** rng.uniform(-8, 0, sh),
        "loguni 12": lambda: sgn() * 2.0 ** rng.uniform(-12, 0, sh),
        "pareto 1.5": lambda: sgn() * (rng.pareto(1.5, sh) + 1),
        "pareto 3": lambda: sgn() * (rng.pareto(3, sh) + 1),
        "lognormal 1": lambda: sgn() * np.exp(rng.standard_normal(sh)),
        "lognormal 2": lambda: sgn() * np.exp(2 * rng.standard_normal(sh)),
        "ternary": lambda: rng.choice([-0.5, 0, 0.5], sh),
        "binary": lambda: rng.choice([-1.0, 1.0], sh) * 1.1,
        "6-level": lambda: rng.choice([-3, -2, -1, 1, 2, 3], sh) * 0.3,
        "bimodal": lambda: sgn() * (1 + 0.1 * rng.standard_normal(sh)),
        "sinusoid": lambda: np.sin(2 * np.pi * np.arange(sh[1])[None] / 50 + rng.uniform(0, 6, (sh[0], 1)))
                            + 0.05 * rng.standard_normal(sh),
        "col spread 1e4": lambda: rng.standard_normal(sh) * np.repeat(10 ** rng.uniform(-2, 2, 16), 8)[None],
        "half rows zero": zero_rows,
        "pre-quantized int8": pre8,
        "bfp clamp high x1e3": lambda: 1e3 * rng.standard_normal(sh),
        "bfp clamp low x1e-4": lambda: 1e-4 * rng.standard_normal(sh),
    }
    for name, g in fam.items():
        yield name, g(), _x(rng, sh[1], "white")


def random_suite(n=1000, seed=21):
    rng = np.random.default_rng(seed)
    sh = SHAPE
    for it in range(n):
        base = rng.choice(["gauss", "laplace", "t3", "loguni"])
        if base == "gauss":
            W = rng.standard_normal(sh)
        elif base == "laplace":
            W = rng.laplace(size=sh)
        elif base == "t3":
            W = rng.standard_t(3, sh)
        else:
            W = rng.choice([-1, 1], sh) * 2.0 ** rng.uniform(-rng.uniform(2, 14), 0, sh)
        W = W * (rng.random(sh) > rng.choice([0, 0, 0.3, 0.6, 0.9]))
        po, k = 10 ** rng.uniform(-4, -1.3), 10 ** rng.uniform(0.5, 3)
        mode = rng.choice(["random", "channel", "block"])
        if mode == "random":
            m = rng.random(sh) < po
        elif mode == "channel":
            m = np.zeros(sh, bool)
            m[:, rng.choice(sh[1], max(1, int(po * sh[1] * 3)), replace=False)] = True
        else:
            m = np.zeros(sh, bool)
            m[:, ::32] = rng.random((sh[0], 4)) < po * 30
        W = np.where(m, W * k * rng.uniform(0.5, 1.5, sh) + np.sign(W + 1e-9) * k * 0.5, W)
        W = W * 10 ** rng.uniform(-1, 1, (sh[0], 1)) * (10 ** rng.uniform(-1, 1, (1, sh[1])) if rng.random() < 0.3 else 1)
        # power-of-two rescale so max|W| is in [8, 16): MX results are unchanged (scale-equivariant),
        # and BFP6's 4-bit exponent [-8, 7] only clamps for blocks 11+ octaves below the max
        W = W * 2.0 ** (3 - np.floor(np.log2(np.abs(W).max())))
        xk = rng.choice(["white", "chan_gain", "outlier_chan", "lowrank"])
        yield f"rand{it:04d} {base}/{mode}/X={xk}", W, _x(rng, sh[1], xk)
