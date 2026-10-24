"""Format selectors for CS611 A1, Group 8: predict every candidate's error, pick the smallest.
Owner: Aryan.  Uses formats.py v2.1 (five candidates, MXINT6 two's complement, BFP6 16|4|5).

Every approach has the same interface
    approach(W, h=None, **kw) -> {format name: predicted squared error}
with h = per-input-channel weight H_jj = (X X^T)_jj.  h=None scores the weight error
sum (W - Wq)^2 ("way 1": from the weights alone); h given scores sum_ij h_j (W - Wq)_ij^2,
the diagonal part of ||(W - Wq) X||^2 ("way 2": the activations decide how much each weight
matters).  select(W, h, approach) returns the argmin.  APPROACHES lists them all, from the
cheapest rule to brute force, so bench_selector.py can plot accuracy against cost.

The main method: a format-independent sketch plus a per-format kernel
--------------------------------------------------------------------
For a block with reference exponent R (= floor(log2 max|w|), BFP: clamped to its 4-bit field)
write each non-zero weight as |w| = mu * 2^t, mu in [1, 2), t = floor(log2|w|), and let
d = R - t be its octave offset below the block peak.  Under the class rule every candidate's
scale is 2^(R - emax_f), so the quantized magnitude is c_f(d, mu, sign) * 2^t with c_f a fixed
function of the format alone, and the squared error of the weight is
    4^t (mu - c)^2 = m2 - 2 c m1 + c^2 m0,   m0 = 4^t, m1 = 4^t mu, m2 = w^2.
Bin the weights by (sign, d, mantissa bin of mu) and keep the three sums per bin: that is the
sketch.  If c_f is constant on every mantissa bin the prediction is EXACT, not approximate.
For 6-bit formats all rounding midpoints and saturation points lie on multiples of 1/32 of a
binade, so 64 mantissa bins give the exact class-rule error of all five candidates from one
shared sketch (test_selector.py checks it to 1e-9).  Coarser bins fall back to a uniform
density inside the bin; nb = 1 is "one bin per octave", and dropping the octave detail too gives
the two-statistic closed forms rho (block peaks) and phi (element binades).
The sketch is additive, so it can be built from a random sample of blocks (an unbiased estimate
with an error bar) or merged across shards; scoring a format costs O(#bins), whatever the size
of W.  The weighted-histogram idea is Akash's predictor.py; this version makes it exact,
handles BFP's 16-blocks and exponent clamp and the two's-complement sign asymmetry.
"""
import time

import numpy as np

from formats import FORMATS, spec, layout, elem_round, floor_log2, quantize

CANDIDATES = list(FORMATS)
DMIN, DMAX = -8, 40          # octave offsets kept; d > DMAX flushes in every candidate
NB = 64                      # mantissa bins per octave (exact for 6-bit formats)
UNIT = 32                    # sampling unit: one MX block (two BFP blocks of 16 nest inside)


# --------------------------------------------------------------------------- helpers
def _family(name):
    """Block organisation shared by formats that use the same reference exponent R."""
    kind, _, _ = spec(name)
    lay = layout(name)
    return (lay["block"], tuple(lay["field"]) if kind == "bfp" else None)


def _units(W, h):
    """W (r, k) -> |W| and sign as (r * nu, UNIT) zero-padded units, plus h per element.
    Padding with zeros never changes a block maximum and contributes no error."""
    r, k = W.shape
    nu = -(-k // UNIT)
    P = np.zeros((r, nu * UNIT))
    P[:, :k] = W
    hp = np.zeros(nu * UNIT)
    hp[:k] = 1.0 if h is None else h
    return P.reshape(r * nu, UNIT), np.tile(hp.reshape(nu, UNIT), (r, 1))


def _sample(Wu, Hu, frac, rng):
    """Uniform sample of units without replacement; returns the units and the expansion factor."""
    n = Wu.shape[0]
    if frac >= 1.0:
        return Wu, Hu, 1.0
    m = max(1, int(round(frac * n)))
    idx = (rng or np.random.default_rng(0)).choice(n, m, replace=False)
    return Wu[idx], Hu[idx], n / m


def _ref_exp(A, block, field):
    """Per-element reference exponent R of its block (A: |W| units, block divides UNIT)."""
    n = A.shape[0]
    amax = A.reshape(n, UNIT // block, block).max(2)
    R = floor_log2(np.where(amax > 0, amax, 1.0))
    if field is not None:
        R = np.clip(R, field[0], field[1])
    return np.repeat(R, block, axis=1)


# --------------------------------------------------------------------------- the sketch
def sketch(W, h=None, nb=NB, frac=1.0, rng=None, families=None):
    """Format-independent sketch: {family: array (2 signs, DMAX-DMIN+1 offsets, nb, 3 moments)}."""
    Wu, Hu = _units(np.asarray(W, dtype=np.float64), h)
    Wu, Hu, scale = _sample(Wu, Hu, frac, rng)
    A = np.abs(Wu)
    nz = A > 0
    f, ex = np.frexp(A[nz])
    t = (ex - 1).astype(np.float64)
    mu = 2.0 * f                                         # exact: |w| = mu * 2^t, mu in [1, 2)
    j = np.minimum(((mu - 1.0) * nb).astype(int), nb - 1)
    s = (Wu[nz] < 0).astype(int)
    hw = Hu[nz] * scale
    m0 = hw * 4.0 ** t
    out = {}
    for fam in families or sorted({_family(n) for n in CANDIDATES}, key=str):
        R = _ref_exp(A, *fam)[nz]
        d = np.clip(R - t, DMIN, DMAX).astype(int) - DMIN
        flat = (s * (DMAX - DMIN + 1) + d) * nb + j
        size = 2 * (DMAX - DMIN + 1) * nb
        mom = np.stack([np.bincount(flat, hw * A[nz] ** 2, size),      # m2, m1, m0: the
                        np.bincount(flat, m0 * mu, size),              # order of kernel()'s
                        np.bincount(flat, m0, size)], 1)               # coefficients a, b, c
        out[fam] = mom.reshape(2, DMAX - DMIN + 1, nb, 3)
    return out


_KCACHE = {}


def kernel(name, nb=NB, sub=16):
    """Coefficients (a, b, c) per (sign, offset, bin) so that the error of a bin is
    a*m2 + b*m1 + c*m0.  Exact where the quantized mantissa is constant on the bin, otherwise
    the mean squared error under a uniform density of mu inside the bin."""
    key = (name, nb, sub)
    if key in _KCACHE:
        return _KCACHE[key]
    kind, p, emax = spec(name)
    ds = np.arange(DMIN, DMAX + 1, dtype=np.float64)[:, None, None]
    mu = 1.0 + (np.arange(nb)[:, None] + (np.arange(sub) + 0.5) / sub)[None] / nb   # (1, nb, sub)
    K = np.zeros((2, ds.shape[0], nb, 3))
    for si, sg in enumerate((1.0, -1.0)):
        z = mu * 2.0 ** (emax - ds)                       # weight in units of the format's scale
        c = np.abs(elem_round(kind, p, sg * z)) * 2.0 ** (ds - emax)   # quantized, units of 2^t
        const = (c == c[..., :1]).all(-1)
        c0 = c[..., 0]
        K[si, ..., 0] = np.where(const, 1.0, 0.0)
        K[si, ..., 1] = np.where(const, -2.0 * c0, 0.0)
        K[si, ..., 2] = np.where(const, c0 ** 2, ((mu - c) ** 2).mean(-1))
    _KCACHE[key] = K
    return K


def score_sketch(sk, names=None, nb=NB):
    names = names or CANDIDATES
    return {n: float((kernel(n, nb) * sk[_family(n)]).sum()) for n in names}


# --------------------------------------------------------------------------- approaches
def sketch_select(W, h=None, nb=NB, frac=1.0, rng=None):
    """Approach 2 (main): sketch + kernel; nb=64 is exact for the class rule."""
    return score_sketch(sketch(W, h, nb, frac, rng), nb=nb)


def closed_form(W, h=None):
    """Two statistics, no flush/clip terms: INT/BFP from block peaks (rho), FP from binades (phi).
        INT/BFP  E = 4^-4/12 * sum h 4^R      FP  E = 4^-m/12 * sum h 4^t"""
    Wu, Hu = _units(np.asarray(W, dtype=np.float64), h)
    A = np.abs(Wu)
    nz = A > 0
    t = floor_log2(np.where(nz, A, 1.0))
    phi = (Hu * nz * 4.0 ** t).sum()
    out = {}
    for n in CANDIDATES:
        kind, p, _ = spec(n)
        if kind == "fp":
            out[n] = float(phi * 4.0 ** -p["mbits"] / 12)
        else:
            R = _ref_exp(A, *_family(n))
            out[n] = float((Hu * nz * 4.0 ** R).sum() * 4.0 ** -4 / 12)
    return out


def _elem_model(W, h, name, exact_top):
    """Per-element expected error (M1): step^2/12, flushed z^2, clipped (z - vmax)^2.
    exact_top: round the weights in the top octave of each block exactly (the hybrid)."""
    kind, p, emax = spec(name)
    Wu, Hu = _units(np.asarray(W, dtype=np.float64), h)
    A = np.abs(Wu)
    block, field = _family(name)
    R = _ref_exp(A, block, field)
    sc = 2.0 ** (R - emax)
    z = A / sc
    if kind == "fp":
        step = 2.0 ** (np.maximum(floor_log2(np.where(z > 0, z, 1.0)), 1 - p["bias"]) - p["mbits"])
        vm = np.full_like(z, p["maxval"])
    elif kind == "int":
        step = np.full_like(z, 2.0 ** -p["frac_bits"])
        vm = p["kmax"] * step
        if p.get("kmin") is not None:                    # two's complement: -32 is a code
            vm = np.where(Wu < 0, -p["kmin"] * step, vm)
    else:
        step = np.ones_like(z)
        vm = np.full_like(z, float(p["kmax"]))
    err = np.where(z > vm, (z - vm) ** 2, np.where(z < step / 2, z ** 2, step ** 2 / 12))
    if exact_top:
        top = (A > 0) & (floor_log2(np.where(A > 0, A, 1.0)) >= R)
        ex = (z - np.abs(elem_round(kind, p, np.sign(Wu) * z))) ** 2
        err = np.where(top, ex, err)
    return float((Hu * (A > 0) * err * sc ** 2).sum())


def m1(W, h=None):
    return {n: _elem_model(W, h, n, False) for n in CANDIDATES}


def hybrid(W, h=None):
    return {n: _elem_model(W, h, n, True) for n in CANDIDATES}


def akash_hist(W, h=None):
    """Akash's predictor.py (v1), adapted: BFP6 scored on 16-blocks with the integer grid."""
    from predictor import build_stats, predict
    H = None if h is None else np.diag(h)
    out = predict(build_stats(W, H), [n for n in CANDIDATES if n != "BFP6"])
    out["BFP6"] = predict(build_stats(W, H, block=16), ["MXINT6"])["MXINT6"]
    return out


def exhaustive(W, h=None, X=None, policy="ocp"):
    """Brute force: quantize with every candidate.  X given: the full ||(W - Wq) X||^2."""
    out = {}
    for n in CANDIDATES:
        D = W - quantize(W, n, policy=policy)[0]
        if X is not None:
            out[n] = float(((D @ X) ** 2).sum())
        else:
            out[n] = float(((D ** 2) * (1.0 if h is None else h[None, :])).sum())
    return out


def exhaustive_rows(W, h=None, frac=0.1, rng=None):
    """Brute force on a random sample of rows (blocks never cross rows)."""
    r = W.shape[0]
    m = max(1, int(round(frac * r)))
    idx = (rng or np.random.default_rng(0)).choice(r, m, replace=False)
    return exhaustive(W[idx], h)


def certified(W, h=None, n0=32, z=3.0, rng=None):
    """Sampled sketch with an error bar: double the number of sampled units until the best
    format beats every other one by z standard errors (finite-population corrected), or all
    units are used.  Returns the scores; certified.last = (units used, total, certified?)."""
    Wu, Hu = _units(np.asarray(W, dtype=np.float64), h)
    N = Wu.shape[0]
    order = (rng or np.random.default_rng(0)).permutation(N)
    n = min(n0, N)
    while True:
        idx = order[:n]
        E = np.stack([_unit_errors(Wu[idx], Hu[idx], name) for name in CANDIDATES], 1)  # (n, F)
        tot = E.mean(0) * N
        b = int(tot.argmin())
        ok = True
        if n < N:
            for o in range(len(CANDIDATES)):
                if o == b:
                    continue
                dlt = E[:, o] - E[:, b]
                se = dlt.std(ddof=1) / np.sqrt(n) * np.sqrt(1 - n / N) if n > 1 else np.inf
                if not dlt.mean() > z * se:
                    ok = False
                    break
        if ok or n >= N:
            certified.last = (n, N, ok)
            return {name: float(v) for name, v in zip(CANDIDATES, tot)}
        n = min(2 * n, N)


def certified_is(W, h=None, n0=32, z=3.0, rng=None, mix=0.3):
    """Importance-sampled version of certified(): read every unit's maximum (a max-reduction,
    no rounding; an MX-stored model already has it as the shared exponent), sample units with
    probability proportional to (1 - mix) * sum h 4^F + mix * uniform, and estimate each
    format's total with the Hansen-Hurwitz estimator.  Rare heavy blocks (outliers, BFP
    saturation) are then sampled almost surely, which plain uniform sampling misses.
    certified_is.last = (distinct units read, total, certified?)."""
    Wu, Hu = _units(np.asarray(W, dtype=np.float64), h)
    N = Wu.shape[0]
    A = np.abs(Wu)
    amax = A.max(1)
    F = floor_log2(np.where(amax > 0, amax, 1.0))
    q = (Hu * (A > 0)).sum(1) * 4.0 ** F * (amax > 0)
    if q.sum() == 0:
        certified_is.last = (0, N, True)
        return {name: 0.0 for name in CANDIDATES}
    pi = (1 - mix) * q / q.sum() + mix / N
    rng = rng or np.random.default_rng(0)
    draws = np.empty(0, dtype=int)
    n = n0
    while True:
        draws = np.concatenate([draws, rng.choice(N, n - draws.size, p=pi)])
        uniq = np.unique(draws)
        if uniq.size >= N:                             # cheaper to do everything exactly
            E = np.stack([_unit_errors(Wu, Hu, name) for name in CANDIDATES], 1)
            certified_is.last = (N, N, True)
            return {name: float(v) for name, v in zip(CANDIDATES, E.sum(0))}
        Eu = np.stack([_unit_errors(Wu[uniq], Hu[uniq], name) for name in CANDIDATES], 1)
        pos = np.searchsorted(uniq, draws)
        Y = Eu[pos] / pi[draws][:, None]               # one estimate of the total per draw
        tot = Y.mean(0)
        b = int(tot.argmin())
        ok = all((Y[:, o] - Y[:, b]).mean() > z * (Y[:, o] - Y[:, b]).std(ddof=1) / np.sqrt(len(draws))
                 for o in range(len(CANDIDATES)) if o != b)
        if ok:
            certified_is.last = (uniq.size, N, True)
            return {name: float(v) for name, v in zip(CANDIDATES, tot)}
        n = 2 * len(draws)


certified_is.last = None


def _unit_errors(Wu, Hu, name, nb=NB):
    """Exact class-rule error of each unit via the kernel (no rounding of the weights)."""
    A = np.abs(Wu)
    nz = A > 0
    f, ex = np.frexp(np.where(nz, A, 1.0))
    t = ex - 1.0
    mu = 2.0 * f
    j = np.minimum(((mu - 1.0) * nb).astype(int), nb - 1)
    R = _ref_exp(A, *_family(name))
    d = np.clip(R - t, DMIN, DMAX).astype(int) - DMIN
    s = (Wu < 0).astype(int)
    K = kernel(name, nb)[s, d, j]                       # (n, UNIT, 3)
    m0 = 4.0 ** t
    e = K[..., 0] * A ** 2 + K[..., 1] * m0 * mu + K[..., 2] * m0
    return (Hu * nz * e).sum(1)


certified.last = None


# --------------------------------------------------------------------------- learned rule
FEATURES = ("sparsity", "kurtosis_nz", "log_rho32", "log_rho16", "log_phi", "flush_frac")


def features(W):
    """Cheap scalar features for the learned rule (one pass, no rounding)."""
    W = np.asarray(W, dtype=np.float64)
    nzv = W[W != 0]
    e2 = (W ** 2).sum()
    cf = closed_form(W)
    x = nzv - nzv.mean() if nzv.size else np.zeros(1)
    kurt = float((x ** 4).mean() / max((x ** 2).mean() ** 2, 1e-300))
    A = np.abs(_units(W, None)[0])
    R = _ref_exp(A, 32, None)
    t = floor_log2(np.where(A > 0, A, 1.0))
    flush = float(((A > 0) & (R - t >= 6)).sum() / max((A > 0).sum(), 1))   # below half an INT6 step
    return np.array([1 - nzv.size / W.size, np.log(kurt),
                     np.log(cf["MXINT6"] * 12 * 4 ** 4 / e2), np.log(cf["BFP6"] * 12 * 4 ** 4 / e2),
                     np.log(cf["MXFP6_E2M3"] * 12 * 4 ** 3 / e2), flush])


def fit_tree(Xf, y, depth=4, min_leaf=10):
    """Tiny CART (Gini) on features -> winner label; returns a nested dict."""
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    node = {"leaf": labels[counts.argmax()]}
    if depth == 0 or len(labels) == 1 or len(y) < 2 * min_leaf:
        return node
    def gini(v):
        _, c = np.unique(v, return_counts=True)
        q = c / c.sum()
        return 1 - (q ** 2).sum()
    best = None
    for k in range(Xf.shape[1]):
        for thr in np.unique(np.percentile(Xf[:, k], np.linspace(5, 95, 19))):
            L = Xf[:, k] <= thr
            if L.sum() < min_leaf or (~L).sum() < min_leaf:
                continue
            g = L.mean() * gini(y[L]) + (~L).mean() * gini(y[~L])
            if best is None or g < best[0]:
                best = (g, k, thr, L)
    if best is None or best[0] >= gini(y):
        return node
    _, k, thr, L = best
    return {"k": k, "thr": float(thr), "lo": fit_tree(Xf[L], y[L], depth - 1, min_leaf),
            "hi": fit_tree(Xf[~L], y[~L], depth - 1, min_leaf)}


def tree_predict(tree, x):
    while "leaf" not in tree:
        tree = tree["lo"] if x[tree["k"]] <= tree["thr"] else tree["hi"]
    return tree["leaf"]


def learned(W, h=None, tree=None):
    """Learned rule: the fitted tree's label gets score 0, the rest 1 (no error estimate)."""
    lab = tree_predict(tree, features(W))
    return {n: 0.0 if n == lab else 1.0 for n in CANDIDATES}


# --------------------------------------------------------------------------- registry
#  name: (function, kwargs, uses_h, cost class, one-line description)
APPROACHES = {
    "learned_tree":   (learned, {}, False, "features", "CART on 6 cheap features (sparsity, kurtosis, rho, phi, flush)"),
    "closed_form":    (closed_form, {}, False, "2 stats", "rho (block peaks) / phi (binades), no flush/clip"),
    "closed_form_h":  (closed_form, {}, True, "2 stats", "same, each weight weighted by H_jj"),
    "sketch_nb1_h":   (sketch_select, {"nb": 1}, True, "sketch", "octave histogram only (1 mantissa bin)"),
    "sketch_nb4_h":   (sketch_select, {"nb": 4}, True, "sketch", "4 mantissa bins per octave"),
    "sketch_nb16_h":  (sketch_select, {"nb": 16}, True, "sketch", "16 mantissa bins per octave"),
    "sketch_h":       (sketch_select, {}, True, "sketch", "64 bins: exact class-rule diag-H error"),
    "sketch":         (sketch_select, {}, False, "sketch", "64 bins, weights only"),
    "sketch_s10_h":   (sketch_select, {"frac": 0.1}, True, "sampled", "exact sketch on 10% of blocks"),
    "certified_h":    (certified, {}, True, "sampled", "uniform sample until the winner is 3 s.e. ahead"),
    "certified_is_h": (certified_is, {}, True, "sampled", "block maxima + importance sample, 3 s.e."),
    "akash_hist_h":   (akash_hist, {}, True, "sketch", "predictor.py (expected error + exact block max)"),
    "m1":             (m1, {}, False, "1 pass", "per-element expected error"),
    "m1_h":           (m1, {}, True, "1 pass", "per-element expected error, H-weighted"),
    "hybrid_h":       (hybrid, {}, True, "1 pass", "M1 with the top octave rounded exactly"),
    "exh_rows10":     (exhaustive_rows, {"frac": 0.1}, False, "sampled", "quantize 10% of rows"),
    "exhaustive_w":   (exhaustive, {}, False, "brute", "quantize all 5, weight error"),
    "exhaustive_h":   (exhaustive, {}, True, "brute", "quantize all 5, diag-H error"),
}


def run(name, W, h=None, X=None, **extra):
    """Run one registered approach; returns (scores, seconds)."""
    fn, kw, uses_h, _, _ = APPROACHES[name]
    t0 = time.perf_counter()
    sc = fn(W, h if uses_h else None, **{**kw, **extra})
    return sc, time.perf_counter() - t0


def select(W, h=None, approach="sketch_h"):
    sc, _ = run(approach, W, h)
    return min(sc, key=sc.get), sc
