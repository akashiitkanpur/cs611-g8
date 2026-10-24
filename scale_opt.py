"""Shared-exponent (scale) policies for the scale-optimization study.  Owner: Anuj.

Written from the team conventions (plan section 2) WITHOUT importing formats.py, so that
scale_study.py can cross-check the two implementations.  The method differs on purpose:
  * element rounding is a table lookup on the grid of values enumerated from the bit
    codes (formats.py computes the local step from the binade);
  * "ceil" is derived as e_ocp or e_ocp + 1 by checking which elements would clip
    (formats.py uses ceil(log2(amax / vmax)) with exact corrections).
floor(log2 amax) comes from frexp in both: formats.py switched to it (commit 32530fa)
after this cross-check found that np.floor(np.log2(a)) is off by one for float64 values
a few ulps below a power of two.

A format is an element grid g (sorted non-negative values; a second grid for negative
values when the code is not symmetric, as in two's complement) plus the element exponent
emax used by the class rule, the block size and the stored shared-exponent range.
An element decodes to  sign * g[code] * 2^e,  e = the shared exponent of its block.
Exponent names follow formats.py v2: for BFP the stored field E is the scale exponent of
the k/16 grid, so a BFP element is +-k * 2^(E - 4). MXINT6 follows the revised D10
(23 Oct): two's complement k in [-32, 31]; the symmetric [-31, 31] code is MXINT6_sym.

Policies, per block, with e_ocp = floor(log2 amax) - emax:
  ocp     e_ocp                                                              [CLASS]
  ceil    e_ocp + 1 if any element would clip at e_ocp, else e_ocp (never clips)
  opt     e_ocp clamped to the exponent field, plus d for d in a window (default -2..+1),
          clamped again; keep the lowest measured error: weight SSE, or r^T H_bb r when
          H = X X^T is given (block-diagonal slice). Ties keep the smallest |d|, then the
          negative d (the formats.py convention).
  model   same candidates, but the error is predicted without rounding anything: exact
          for clipped and flushed-to-zero elements, step^2 / 12 for the others.
  joint_search()  per row, all block exponents chosen together for r^T H r with the
          FULL H (cross-block terms included); the output-error oracle.
"""
import itertools
from typing import NamedTuple, Optional

import numpy as np

E8M0 = (-127, 127)
OPT_WINDOW = (-2, -1, 0, 1)
WIDE_WINDOW = (-3, -2, -1, 0, 1, 2)
ORACLE_WINDOW = tuple(range(-8, 9))


# --------------------------------------------------------------------------- element grids
def fp_grid(ebits, mbits, bias, vmax):
    """Non-negative values of a sign-magnitude FP format with subnormals, up to vmax.

    The list is in code order (exponent field, then mantissa field) and strictly
    increasing, so a value's index has the parity of its mantissa code: rounding ties
    to the even index is ties-to-even.
    """
    vals = []
    for E in range(2 ** ebits):
        for M in range(2 ** mbits):
            lead, ex = (0, 1 - bias) if E == 0 else (1, E - bias)
            v = (lead + M / 2 ** mbits) * 2.0 ** ex
            if v <= vmax:
                vals.append(v)
    return np.array(vals)


def int_grid(kmax, frac_bits):
    """k * 2^-frac_bits for k = 0..kmax (the index is k, so even index = even code)."""
    return np.arange(kmax + 1) * 2.0 ** -frac_bits


def round_to_grid(x, grid):
    """Nearest grid value, ties to the even index, saturate at grid[-1]; keeps the sign."""
    a = np.abs(x)
    i = np.clip(np.searchsorted(grid, a, side="right") - 1, 0, len(grid) - 1)
    j = np.minimum(i + 1, len(grid) - 1)
    lo, hi = grid[i], grid[j]
    up = (hi - a < a - lo) | ((hi - a == a - lo) & (i % 2 == 1))
    return np.copysign(np.where(up, hi, lo), x)


class Fmt(NamedTuple):
    grid: np.ndarray                     # non-negative element values, sorted
    emax: int                            # element exponent used by the class rule
    block: int                           # values per shared exponent
    e_range: tuple                       # allowed stored exponents
    e_bits: int                          # bits that store one shared exponent
    nxt: float                           # next value above vmax if the grid went on
    default: str = "ocp"                 # policy used when none is given
    grid_neg: Optional[np.ndarray] = None    # magnitudes for negative values (None = grid)
    nxt_neg: Optional[float] = None
    neg_zero: bool = True                # False for an integer code: zero has no sign


def _fp(ebits, mbits, bias, vmax, emax, **kw):
    top_step = 2.0 ** (np.frexp(vmax)[1] - 1 - mbits)       # spacing in vmax's binade
    return Fmt(fp_grid(ebits, mbits, bias, vmax), emax, nxt=vmax + top_step, **kw)


def _int(kmax, block, e_range, e_bits, kmin=None, **kw):
    step = 2.0 ** -4
    neg = {} if kmin is None else dict(grid_neg=int_grid(-kmin, 4), nxt_neg=(1 - kmin) * step)
    return Fmt(int_grid(kmax, 4), 0, block, e_range, e_bits, nxt=(kmax + 1) * step, **neg, **kw)


MX = dict(block=32, e_range=E8M0, e_bits=8)
FMTS = {
    "MXFP6_E3M2": _fp(3, 2, 3, 28.0, 4, **MX),
    "MXFP6_E2M3": _fp(2, 3, 1, 7.5, 2, **MX),
    "MXFP6_E4M1": _fp(4, 1, 7, 384.0, 8, **MX),                 # D3: no reserved code
    "MXINT6":     _int(31, **MX, kmin=-32, neg_zero=False),     # D10 (23 Oct): two's complement [-32, 31]
    "BFP6":       _int(31, 16, (-8, 7), 4),                     # D1: 16|4|5, sign + magnitude, +-k * 2^(E-4)
}
VARIANTS = {
    "MXFP6_E4M1_v192": _fp(4, 1, 7, 192.0, 7, **MX),            # top exponent reserved
    "MXFP6_E4M1_nan1": _fp(4, 1, 7, 256.0, 8, **MX),            # one NaN code (E4M3-style)
    "MXINT6_sym":      _int(31, **MX, neg_zero=False),          # symmetric [-31, 31] (OCP style)
    "BFP6_g32e8":      _int(31, **MX),                          # BFP 32|8|5 floor (= MXINT6_sym)
    "BFP6_b32_ceil":   _int(31, **MX, default="ceil"),          # formats.py v1's BFP6
}


def spec(name):
    if name in FMTS:
        return FMTS[name]
    if name in VARIANTS:
        return VARIANTS[name]
    raise KeyError(f"unknown format {name!r}")


def block_slices(n, block):
    return [slice(s, min(s + block, n)) for s in range(0, n, block)]


def bits_per_elem(name, n_cols=100, elem_bits=6):
    """True storage per real element; every block, the tail included, has its own exponent."""
    f = spec(name)
    return elem_bits + len(block_slices(n_cols, f.block)) * f.e_bits / n_cols


# --------------------------------------------------------------------------- elements
def round_elem(f, v):
    """Element rounding of v (already divided by the scale), sign-dependent grid if any.
    An integer code has a single zero, so a negative value that rounds to 0 decodes to +0."""
    if f.grid_neg is None:
        q = round_to_grid(v, f.grid)
    else:
        q = np.where(v < 0, round_to_grid(v, f.grid_neg), round_to_grid(v, f.grid))
    return q if f.neg_zero else q + 0.0


def out_of_range(f, v):
    """|v| beyond the largest element of its sign, before rounding (formats.py: n_range)."""
    lim = f.grid[-1] if f.grid_neg is None else np.where(v < 0, f.grid_neg[-1], f.grid[-1])
    return np.abs(v) > lim


def saturated(f, v):
    """Lecture 19 'saturated': v rounds to a value beyond the largest element, so the
    saturation changes it. Rounds on the grid extended by the next value (formats.py:
    n_saturated)."""
    a = np.abs(v)
    sat = round_to_grid(a, np.append(f.grid, f.nxt)) > f.grid[-1]
    if f.grid_neg is not None:
        sat_n = round_to_grid(a, np.append(f.grid_neg, f.nxt_neg)) > f.grid_neg[-1]
        sat = np.where(v < 0, sat_n, sat)
    return sat


# --------------------------------------------------------------------------- shared exponent
def floor_log2(a):
    """floor(log2 a) for a > 0, exactly: frexp gives a = m * 2^k with m in [0.5, 1)."""
    return np.frexp(a)[1] - 1


def base_exp(f, amax, rule="ocp", Wb=None):
    """Unclamped class-rule exponent per block ("ocp"), or the no-clip exponent ("ceil").

    The block max sits in [2^emax, 2^(emax+1)) * 2^e_ocp and vmax >= 2^emax, so the
    smallest exponent that clips nothing is e_ocp or e_ocp + 1; "ceil" needs the block
    values Wb to check which. All-zero block: 0.
    """
    pos = amax > 0
    e = floor_log2(np.where(pos, amax, 1.0)) - f.emax
    if rule == "ceil":
        clips = out_of_range(f, Wb / np.ldexp(1.0, e)[:, None]).any(1)
        e = e + clips
    elif rule != "ocp":
        raise ValueError(rule)
    return np.where(pos, e, 0)


def candidates(f, e0, offsets):
    """Exponents tried per block: the base exponent clamped to the field, then each
    offset, clamped again (so the search stays inside the field; formats.py v2 does the same)."""
    base = np.clip(e0, *f.e_range)
    return np.stack([np.clip(base + d, *f.e_range) for d in offsets], 1)


def qblock(f, Wb, e):
    """Quantize the rows of one block, row i with shared exponent e[i]."""
    s = np.ldexp(1.0, e)[:, None]
    return round_elem(f, Wb / s) * s


def _sse(r, Hbb):
    return (r * r).sum(1) if Hbb is None else np.einsum("ij,jk,ik->i", r, Hbb, r)


def local_step(f, a, grid=None):
    """Width of the grid cell that holds the element magnitude a (the top cell above vmax)."""
    g = f.grid if grid is None else grid
    i = np.clip(np.searchsorted(g, a, side="right") - 1, 0, len(g) - 2)
    return g[i + 1] - g[i]


def _model_elem(f, a, g):
    step = local_step(f, a, g)
    return np.where(a > g[-1], (a - g[-1]) ** 2, np.where(a < g[1] / 2, a ** 2, step ** 2 / 12))


def model_err(f, Wb, e, h=None):
    """Predicted per-row error of a block at exponents e, without rounding.

    clipped (|w|/s > vmax): exact (|w| - vmax s)^2;  flushed (|w|/s < g[1]/2): exact w^2;
    otherwise uniform rounding noise step^2 / 12 with the step of the grid cell that
    holds |w|/s.  h = diag(X X^T) for this block weights each column (None = 1).
    """
    s = np.ldexp(1.0, e)[:, None]
    a = np.abs(Wb) / s
    err = _model_elem(f, a, f.grid)
    if f.grid_neg is not None:
        err = np.where(Wb < 0, _model_elem(f, a, f.grid_neg), err)
    err = err * s ** 2
    return err.sum(1) if h is None else err @ h


def _preferred(window):
    return sorted(window, key=lambda d: (abs(d), d))


def quantize(W, name, policy=None, window=OPT_WINDOW, H=None):
    """Quantize W row-wise in blocks.  Returns (Wq, exps), exps[i, b] = exponent of block b."""
    f = spec(name)
    W = np.asarray(W, dtype=np.float64)
    policy = policy or f.default
    if policy not in ("ocp", "ceil", "opt", "model"):
        raise ValueError(policy)
    offsets = _preferred(window) if policy in ("opt", "model") else [0]
    Wq = np.empty_like(W)
    exps = []
    for sl in block_slices(W.shape[1], f.block):
        Wb = W[:, sl]
        e0 = base_exp(f, np.abs(Wb).max(1), "ceil" if policy == "ceil" else "ocp", Wb)
        cand = candidates(f, e0, offsets)
        if len(offsets) == 1:
            pick = np.zeros(len(Wb), dtype=int)
        elif policy == "opt":
            Hbb = None if H is None else H[sl, sl]
            err = np.stack([_sse(Wb - qblock(f, Wb, c), Hbb) for c in cand.T], 1)
            pick = err.argmin(1)                  # first minimum = preferred offset
        else:
            h = None if H is None else np.diag(H)[sl]
            err = np.stack([model_err(f, Wb, c, h) for c in cand.T], 1)
            pick = err.argmin(1)
        e = cand[np.arange(len(Wb)), pick]
        Wq[:, sl] = qblock(f, Wb, e)
        exps.append(e)
    return Wq, np.stack(exps, 1)


# --------------------------------------------------------------------------- output oracle
def _cross_terms(W, f, H, window):
    """Residual candidates and G[b, k, c, l, i] = r_b^(k)T H_bc r_c^(l) for every row i."""
    sls = block_slices(W.shape[1], f.block)
    offsets = _preferred(window)
    cands, resid = [], []
    for sl in sls:
        Wb = W[:, sl]
        c = candidates(f, base_exp(f, np.abs(Wb).max(1), "ocp"), offsets)
        cands.append(c)
        resid.append([Wb - qblock(f, Wb, ck) for ck in c.T])
    nb, no = len(sls), len(offsets)
    G = np.empty((nb, no, nb, no, W.shape[0]))
    for b, sb in enumerate(sls):
        for k in range(no):
            P = resid[b][k] @ H[sb, :]
            for c, sc in enumerate(sls):
                for l in range(no):
                    G[b, k, c, l] = (P[:, sc] * resid[c][l]).sum(1)
    return sls, cands, G


def joint_search(W, name, H, window=WIDE_WINDOW, exact=False, max_sweeps=50):
    """Per row, choose all block exponents together to minimise r^T H r with the full H.

    Coordinate descent starting from the block-diagonal optimum (= policy "opt" with H):
    one block at a time is re-optimised with the others fixed, accepting only strict
    improvements, until a sweep changes nothing.  It is never worse than its start but
    is only a local optimum; exact=True enumerates every combination instead.
    Returns (Wq, exps, objective per row).
    """
    f = spec(name)
    W = np.asarray(W, dtype=np.float64)
    sls, cands, G = _cross_terms(W, f, H, window)
    nb, no, n = G.shape[0], G.shape[1], W.shape[0]
    rows = np.arange(n)
    diag = np.stack([G[b, :, b, :][np.arange(no), np.arange(no)] for b in range(nb)])
    choice = diag.argmin(1).T.copy()                       # (n, nb), block-diagonal start

    def total(ch):
        return sum(G[b, ch[:, b], c, ch[:, c], rows] for b in range(nb) for c in range(nb))

    if exact:
        best, best_val = choice.copy(), total(choice)
        for combo in itertools.product(range(no), repeat=nb):
            ch = np.broadcast_to(np.array(combo), (n, nb))
            v = total(ch)
            better = v < best_val
            best[better], best_val[better] = ch[better], v[better]
        choice = best
    else:
        for _ in range(max_sweeps):
            changed = False
            for b in range(nb):
                cost = np.stack([G[b, k, b, k] + 2 * sum(G[b, k, c, choice[:, c], rows]
                                                         for c in range(nb) if c != b)
                                 for k in range(no)], 1)
                cur = cost[rows, choice[:, b]]
                k_best = cost.argmin(1)
                move = cost[rows, k_best] < cur
                if move.any():
                    choice[move, b] = k_best[move]
                    changed = True
            if not changed:
                break
    Wq = np.empty_like(W)
    exps = np.stack([cands[b][rows, choice[:, b]] for b in range(nb)], 1)
    for b, sl in enumerate(sls):
        Wq[:, sl] = qblock(f, W[:, sl], exps[:, b])
    return Wq, exps, total(choice)


# --------------------------------------------------------------------------- diagnostics
def element_exps(exps, n_cols, block):
    """Expand per-block exponents to one exponent per element."""
    out = np.empty((exps.shape[0], n_cols), dtype=exps.dtype)
    for b, sl in enumerate(block_slices(n_cols, block)):
        out[:, sl] = exps[:, b:b + 1]
    return out


def error_parts(W, Wq, exps, name):
    """Split the weight squared error into out-of-range / flushed-to-zero / rounded elements.

    n_range: |w|/scale beyond the largest element before rounding; n_saturated: the rounded
    value is beyond it, so saturation changed it (Lecture 19; the names of formats.py v2).
    """
    f = spec(name)
    s = np.ldexp(1.0, element_exps(exps, W.shape[1], f.block))
    rng_ = out_of_range(f, W / s)
    zero = (W != 0) & (Wq == 0) & ~rng_
    rnd = ~rng_ & ~zero
    d2 = (W - Wq) ** 2
    return dict(sse_clip=float(d2[rng_].sum()), sse_zero=float(d2[zero].sum()),
                sse_round=float(d2[rnd].sum()), n_range=int(rng_.sum()),
                n_saturated=int(saturated(f, W / s).sum()), n_zero=int(zero.sum()))


def rel_err(W, Wq, X=None):
    """Relative Frobenius error of W (X None) or of Y = W @ X."""
    dW = np.asarray(W, float) - Wq
    if X is None:
        return float(np.linalg.norm(dW) / np.linalg.norm(W))
    return float(np.linalg.norm(dW @ X) / np.linalg.norm(W @ X))
