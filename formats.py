"""6-bit candidate formats for CS611 A1, Group 8 (dataset C).  Owner: Akash.   Version 2.1 (24 Oct: D10 revised).

Conventions. [CLASS] = Lecture 19 OCP-MX notebook / board notes; [TEAM Dn] = our decision
from TEAM_PLAN.md section 1, to be stated and defended in the report.

  Blocks    MX: 32 consecutive values along each row of W, the reduction axis of Y = W @ X.
            K = 100 -> blocks of 32, 32, 32 and a tail of 4 real values.            [CLASS]
            Class zero-pads the tail; that gives exactly the same quantized values,
            because padded zeros never change a block's max. Statistics count only
            real elements. Storage counts the tail with its own scale.          [TEAM D11]
  Scale     power of two 2^e. MX stores e as E8M0, clamped to [-127, 127].         [CLASS]
            An all-zero block stores exponent 0 (scale 1); its elements stay 0.    [CLASS]
  Policies  "ocp"   e = floor(log2 amax) - emax_elem                     [CLASS default]
            "ceil"  smallest power-of-two scale that clips nothing; equal to the class
                    notebook's "floor rule, +1 if the max does not fit"  [CLASS follow-up]
            "opt"   per block, try e_ocp + {-2, -1, 0, +1}, keep the lowest error:
                    weight MSE, or output-aware r^T H_bb r with H = X X^T if H is given.
                    Ties keep the offset closest to 0. The scale stays a power of two in
                    the same exponent field, so the format definition is unchanged. [TEAM D5]
  Elements  round to nearest, ties to even, then saturate at vmax (no Inf/NaN).    [CLASS]
            FP subnormals: fixed spacing 2^(emin - m) below 2^emin.               [CLASS]

  The five candidates (FORMATS):
  MXFP6_E3M2  m=2, bias 3, emin=-2, emax=4, vmax=28                                [CLASS]
  MXFP6_E2M3  m=3, bias 1, emin=0,  emax=2, vmax=7.5                               [CLASS]
  MXFP6_E4M1  m=1, bias 7, emin=-6, emax=8, vmax=1.5*2^8=384, no reserved code  [TEAM D3]
  MXINT6      S I F F F F: k * 2^-4, two's complement k in [-32, 31] (class INT notes),
              vmax = 31/16 (most negative -2), emax_elem = 0, S = 2^floor(log2 amax)
                                                        [board notes; TEAM D10, revised 23 Oct]
  BFP6        g|e|m = 16|4|5: groups of 16, ONE 4-bit shared exponent E = floor(log2 amax)
              clamped to [-8, 7], sign + 5-bit magnitude M in [0, 31], value = +-M * 2^(E-4),
              RNE, saturate at M = 31.                                         [TEAM D1]

  Variants (VARIANTS; sensitivity rows, never in the main ranking):
  MXFP6_E4M1_v192  top exponent code reserved (IEEE-like) -> emax=7, vmax=192     [TEAM D3]
  MXFP6_E4M1_nan1  one NaN code (E4M3-style, S.1111.1) -> emax=8, vmax=256        [TEAM D3]
  BFP6_g32e8       BFP 32|8|5, floor rule: numerically IDENTICAL to MXINT6_sym    [TEAM D1]
  BFP6_b32_ceil    v1's BFP6: block 32, 8-bit exponent, "ceil" rule (= MXINT6_sym under ceil)
  BFP6_row         one block per row, 8-bit exponent, "ceil" rule (6.08 bits/elem)
  MXINT6_sym       symmetric k in [-31, 31] (OCP style; the MXINT6 of v1/v2)       [TEAM D10]

  Stored exponent field. quantize() returns exps = the value written to the shared-exponent
  field. For MX formats that is the E8M0 scale exponent e. For BFP it is E, and the step of
  the integer magnitude is 2^(E - 4) (5-bit magnitude read as fixed point m.mmmm).

  Statistics follow the Lecture 19 notebook: n_range = |w|/scale > vmax BEFORE rounding,
  n_saturated = the ROUNDED value is beyond vmax, so saturation changed it (the two need
  not be equal); n_zeroed, n_subnormal, n_scale_clamped (blocks whose exponent hit the
  field limits).

Public API (everything the other scripts use):
  FORMATS, VARIANTS, POLICIES, BLOCK, spec(name), layout(name)
  quantize(W, name, block=None, shift_search=0, H=None, policy=None, return_stats=False)
  quantize_X(X, name, ...)             W+X mode: blocks run down the columns of X   [TEAM D2]
  metrics(W, Wq, X, Xq=None), split_X(X), exhaustive(W, X, ..., mode="w"|"wx"), ranking(rows)
  save_results(rows, path), storage_bits(n_rows, n_cols, block, elem_bits, scale_bits),
  format_storage(name, n_rows, n_cols, block=None), blocks(n, block), default_exp(...),
  elem_round(...)
"""
import csv
import hashlib
import json
import os
import time

import numpy as np

try:                                    # progress bars if tqdm is installed, silent otherwise
    from tqdm.auto import tqdm
except ImportError:                     # pragma: no cover
    def tqdm(it, **_):
        return it

FORMATS_VERSION = "v2.1"
BLOCK = 32
E8M0_MIN, E8M0_MAX = -127, 127
OPT_OFFSETS = (-2, -1, 0, 1)
POLICIES = ("ocp", "ceil", "opt")


# --------------------------------------------------------------------------- exact log2
def floor_log2(x):
    """Exact floor(log2(x)) for x > 0.

    np.floor(np.log2(x)) is wrong for float64 values one ulp below a power of two:
    log2 rounds up to the integer, e.g. floor(log2(8 - ulp)) gives 3 instead of 2.
    frexp splits x = m * 2^k with m in [0.5, 1) exactly, so floor(log2 x) = k - 1.
    """
    return (np.frexp(x)[1] - 1).astype(np.float64)


# --------------------------------------------------------------------------- elements
def fp_round(v, ebits, mbits, bias, maxval):
    """Round to a sign-magnitude FP grid with subnormals: RNE, then saturate at maxval.

    ebits is not used in the arithmetic (bias and maxval fix the range); it is kept so
    the parameter set reads like the format name.
    """
    a = np.abs(v)
    emin = 1 - bias
    e = floor_log2(np.where(a > 0, a, 1.0))
    e = np.maximum(e, emin)              # below 2^emin: subnormal, fixed spacing
    step = 2.0 ** (e - mbits)
    q = np.round(a / step) * step        # np.round is round-half-to-even
    q = np.minimum(q, maxval)            # saturate: no Inf/NaN codes
    return np.copysign(q, v)


def int_round(v, frac_bits, kmax, kmin=None):
    """Fixed point k * 2^-frac_bits, RNE, k in [kmin, kmax] (kmin defaults to -kmax)."""
    step = 2.0 ** -frac_bits
    return np.clip(np.round(v / step), -kmax if kmin is None else kmin, kmax) * step


FORMATS = {
    # name: (kind, element params, emax_elem used by the "ocp" scale rule)
    "MXFP6_E3M2": ("fp", dict(ebits=3, mbits=2, bias=3, maxval=28.0), 4),
    "MXFP6_E2M3": ("fp", dict(ebits=2, mbits=3, bias=1, maxval=7.5), 2),
    "MXFP6_E4M1": ("fp", dict(ebits=4, mbits=1, bias=7, maxval=384.0), 8),
    "MXINT6":     ("int", dict(frac_bits=4, kmax=31, kmin=-32), 0),   # D10: two's complement
    "BFP6":       ("bfp", dict(kmax=31), 4),   # integer magnitude M; 31 = 1.9375 * 2^4
}
# Variants are kept out of FORMATS so that code looping over FORMATS still sees
# exactly the five mandatory candidates.
VARIANTS = {
    "MXFP6_E4M1_v192": ("fp", dict(ebits=4, mbits=1, bias=7, maxval=192.0), 7),
    "MXFP6_E4M1_nan1": ("fp", dict(ebits=4, mbits=1, bias=7, maxval=256.0), 8),
    "BFP6_g32e8":      ("bfp", dict(kmax=31), 4),
    "BFP6_b32_ceil":   ("bfp", dict(kmax=31), 4),
    "BFP6_row":        ("bfp", dict(kmax=31), 4),
    "MXINT6_sym":      ("int", dict(frac_bits=4, kmax=31), 0),
}

# Block organisation and shared-exponent field. Anything not listed uses LAYOUT_DEFAULT.
#   block       values per block along the reduction axis ("row" = the whole row)
#   policy      default scale rule
#   scale_bits  width of the stored shared exponent
#   field       range of the stored exponent field (clamped)
LAYOUT_DEFAULT = dict(block=BLOCK, policy="ocp", scale_bits=8, field=(E8M0_MIN, E8M0_MAX))
LAYOUT = {
    "BFP6":          dict(block=16, scale_bits=4, field=(-8, 7)),          # D1: 16|4|5
    "BFP6_b32_ceil": dict(policy="ceil"),
    "BFP6_row":      dict(block="row", policy="ceil"),
}


def spec(name):
    """(kind, params, emax_elem) for a format or variant name."""
    if name in FORMATS:
        return FORMATS[name]
    if name in VARIANTS:
        return VARIANTS[name]
    raise KeyError(f"unknown format {name!r}; choose from {list(FORMATS) + list(VARIANTS)}")


def layout(name):
    """Block size, default policy, exponent width and exponent-field range of a format."""
    spec(name)                                   # raises on an unknown name
    return {**LAYOUT_DEFAULT, **LAYOUT.get(name, {})}


def field_offset(kind, emax):
    """stored field - scale exponent: 0 for MX (E8M0 stores e), emax_elem (= 4) for BFP."""
    return emax if kind == "bfp" else 0


def vmax_elem(kind, p):
    """Largest positive element magnitude before scaling."""
    if kind == "fp":
        return p["maxval"]
    if kind == "int":
        return p["kmax"] * 2.0 ** -p["frac_bits"]
    return float(p["kmax"])


def vmin_elem(kind, p):
    """Most negative element value before scaling (-vmax except for two's complement)."""
    if kind == "int" and p.get("kmin") is not None:
        return p["kmin"] * 2.0 ** -p["frac_bits"]
    return -vmax_elem(kind, p)


def elem_round(kind, p, v):
    if kind == "fp":
        return fp_round(v, **p)
    if kind == "int":
        return int_round(v, **p)
    if kind == "bfp":
        return np.clip(np.round(v), -p["kmax"], p["kmax"])
    raise ValueError(kind)


def elem_round_unsaturated(kind, p, v):
    """The same rounding with no saturation, to count class-style 'saturated' elements."""
    if kind == "fp":
        return fp_round(v, **{**p, "maxval": np.inf})
    if kind == "int":
        step = 2.0 ** -p["frac_bits"]
        return np.round(v / step) * step
    return np.round(v)


# --------------------------------------------------------------------------- blocks + scale
def blocks(n, block=BLOCK):
    """Column slices of the blocks of a row of length n ("row" or None = one block)."""
    if block in (None, "row"):
        block = n
    return [slice(s, min(s + block, n)) for s in range(0, n, block)]


def default_exp(kind, p, emax, amax, policy=None, field=(E8M0_MIN, E8M0_MAX), clamp=True):
    """Scale exponent e (scale = 2^e) for blocks with max |w| = amax.

    policy None -> "ocp" (v1 used "ceil" for BFP; D1 makes the floor rule the BFP default).
    field: range of the STORED exponent; for BFP the stored value is e + emax_elem.
    All-zero blocks store 0. clamp=False returns the unclamped exponent.
    """
    amax = np.asarray(amax, dtype=np.float64)
    policy = policy or "ocp"
    off = field_offset(kind, emax)
    pos = amax > 0
    safe = np.where(pos, amax, 1.0)
    if policy == "ocp":
        e = floor_log2(safe) - emax
    elif policy == "ceil":
        # smallest e with amax / 2^e <= vmax. log2 of the ratio can be off by one
        # near a power of two, so correct it with exact power-of-two divisions.
        vmax = vmax_elem(kind, p)
        e = np.ceil(np.log2(safe / vmax))
        e = np.where(safe / 2.0 ** e > vmax, e + 1, e)
        e = np.where(safe / 2.0 ** (e - 1) <= vmax, e - 1, e)
    else:
        raise ValueError(f"default_exp needs policy 'ocp' or 'ceil', got {policy!r}")
    e = np.where(pos, e, -off)
    if clamp:
        e = np.clip(e, field[0] - off, field[1] - off)
    return e


def storage_bits(n_rows, n_cols, block=BLOCK, elem_bits=6, scale_bits=8):
    """(total bits, bits per real element); every block, the tail included, has one scale."""
    nblk = n_rows * len(blocks(n_cols, block))
    total = n_rows * n_cols * elem_bits + nblk * scale_bits
    return total, total / (n_rows * n_cols)


def format_storage(name, n_rows, n_cols, block=None, elem_bits=6):
    """True storage of a format on an n_rows x n_cols matrix (D11): (total bits, bits/elem)."""
    lay = layout(name)
    return storage_bits(n_rows, n_cols, block or lay["block"], elem_bits, lay["scale_bits"])


# --------------------------------------------------------------------------- quantize
def quantize(W, name, block=None, shift_search=0, H=None, policy=None, return_stats=False):
    """Quantize W row-wise in blocks along its columns.

    name          a key of FORMATS or VARIANTS
    block         block size; None = the format's default (layout(name)["block"])
    policy        "ocp" | "ceil" | "opt"; None = the format's default
    shift_search  legacy option: also try base exponent +-k (ignored when policy="opt")
    H             if given (H = X X^T), the exponent search minimises the output-aware
                  error r^T H_bb r per block instead of the weight MSE. It uses only
                  the block's own diagonal slice of H, which is exact when X X^T is
                  diagonal and an approximation otherwise.
    Returns (Wq, exps), or (Wq, exps, stats) with return_stats=True;
    exps[i, b] = stored shared-exponent field of block b in row i (see module docstring).
    """
    kind, p, emax = spec(name)
    lay = layout(name)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be a 2-D matrix")
    policy = policy or lay["policy"]
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    block = block or lay["block"]
    if block == "row":
        block = W.shape[1]
    off = field_offset(kind, emax)
    e_lo, e_hi = lay["field"][0] - off, lay["field"][1] - off      # limits on the scale exponent

    if policy == "opt":
        base, offsets = "ocp", OPT_OFFSETS
    else:
        base, offsets = policy, range(-shift_search, shift_search + 1)
    offsets = sorted(offsets, key=lambda d: (abs(d), d))   # 0 first, so ties keep the base rule

    vpos, vneg = vmax_elem(kind, p), vmin_elem(kind, p)
    Wq = np.empty_like(W)
    exps = []
    n_range = n_sat = n_zero = n_sub = n_clamped = 0
    for sl in blocks(W.shape[1], block):
        Wb = W[:, sl]
        amax = np.abs(Wb).max(axis=1)
        e_raw = default_exp(kind, p, emax, amax, policy=base, field=lay["field"], clamp=False)
        e0 = np.clip(e_raw, e_lo, e_hi)
        best_err = np.full(W.shape[0], np.inf)
        best_q = np.zeros_like(Wb)
        best_e = e0.copy()
        for d in offsets:
            e = np.clip(e0 + d, e_lo, e_hi)
            s = (2.0 ** e)[:, None]
            q = elem_round(kind, p, Wb / s) * s
            r = Wb - q
            err = (r * r).sum(1) if H is None else np.einsum("ij,jk,ik->i", r, H[sl, sl], r)
            better = err < best_err
            best_err[better] = err[better]
            best_q[better] = q[better]
            best_e[better] = e[better]
        Wq[:, sl] = best_q
        exps.append(best_e + off)
        if return_stats:
            s = (2.0 ** best_e)[:, None]
            v = Wb / s
            n_range += int(((v > vpos) | (v < vneg)).sum())
            n_sat += int((elem_round_unsaturated(kind, p, v) != elem_round(kind, p, v)).sum())
            n_zero += int(((Wb != 0) & (best_q == 0)).sum())
            n_clamped += int((e_raw != e0).sum())
            if kind == "fp":
                small = np.abs(best_q) / s < 2.0 ** (1 - p["bias"])
                n_sub += int(((best_q != 0) & small).sum())
    exps = np.stack(exps, 1)
    if not return_stats:
        return Wq, exps

    total, per = storage_bits(W.shape[0], W.shape[1], block, 6, lay["scale_bits"])
    n_nonzero = int((W != 0).sum())
    stats = dict(
        format=name, policy=policy, block=int(block), scale_bits=lay["scale_bits"],
        n_elements=int(W.size), n_blocks=int(exps.size),
        n_range=n_range,                      # |w| / scale outside the range BEFORE rounding
        n_saturated=n_sat,                    # rounded value beyond vmax, clipped (class def.)
        n_zeroed=n_zero,                      # non-zero weights that became 0
        zeroed_frac=n_zero / max(n_nonzero, 1),
        n_subnormal=n_sub,                    # FP only: non-zero outputs below 2^emin
        n_scale_clamped=n_clamped,            # blocks whose base exponent hit the field limits
        bits_total=int(total), bits_per_elem=float(per),
        n_roundings=int(W.size * len(offsets)),   # element quantizations performed
        exp_min=int(exps.min()), exp_max=int(exps.max()),
    )
    return Wq, exps, stats


def quantize_X(X, name, **kw):
    """W+X mode (D2): quantize the input X of Y = W @ X along ITS reduction axis.

    The reduction axis of X is axis 0, so blocks run down each column of X
    (each column = one input vector, as the class notebook quantizes activations
    along their input-feature axis). Columns are therefore independent.
    Same keyword arguments as quantize(); exps has shape (n_blocks, n_columns).
    """
    out = quantize(np.asarray(X, dtype=np.float64).T, name, **kw)
    return (out[0].T, out[1].T) + tuple(out[2:])


# --------------------------------------------------------------------------- metrics
def metrics(W, Wq, X, Xq=None):
    """Errors on W and on Y = W @ X. Relative errors are Frobenius-norm ratios.

    Xq None: weights-only (Yq = Wq @ X), as in v1. Xq given: W+X mode (Yq = Wq @ Xq).
    """
    W = np.asarray(W, dtype=np.float64)
    Wq = np.asarray(Wq, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    dW = W - Wq
    Y = W @ X
    if Xq is None:
        dY = dW @ X
    else:
        Xq = np.asarray(Xq, dtype=np.float64)
        dY = Y - Wq @ Xq
    Yq = Y - dY
    with np.errstate(divide="ignore", invalid="ignore"):
        ny, nq = np.linalg.norm(Y, axis=1), np.linalg.norm(Yq, axis=1)
        den = ny * nq
        # both rows zero -> 1 (identical); exactly one zero -> 0 (nothing in common)
        cos = np.where(den > 0, (Y * Yq).sum(1) / np.where(den > 0, den, 1.0),
                       np.where((ny == 0) & (nq == 0), 1.0, 0.0))
        nz = W != 0
        out = dict(
            w_mse=float((dW ** 2).mean()),
            w_rel_err=float(np.linalg.norm(dW) / np.linalg.norm(W)),
            w_sqnr_db=float(10 * np.log10((W ** 2).sum() / (dW ** 2).sum())),
            y_rel_err=float(np.linalg.norm(dY) / np.linalg.norm(Y)),
            y_sqnr_db=float(10 * np.log10((Y ** 2).sum() / (dY ** 2).sum())),
            y_cos_row_mean=float(cos.mean()),
            y_cos_row_min=float(cos.min()),
            zeroed_frac=float((nz & (Wq == 0)).sum() / max(nz.sum(), 1)),
        )
        if Xq is not None:
            out["x_rel_err"] = float(np.linalg.norm(X - Xq) / np.linalg.norm(X))
        return out


def split_X(X):
    """Deterministic calibration split: (even columns to fit, odd columns held out)."""
    return X[:, ::2], X[:, 1::2]


# --------------------------------------------------------------------------- exhaustive
EXHAUSTIVE_POLICIES = ("ocp", "ceil", "opt", "opt_out")
MODES = ("w", "wx")
with open(os.path.abspath(__file__), "rb") as _f:      # any edit to this file invalidates checkpoints
    _CODE_HASH = hashlib.sha1(_f.read()).hexdigest()[:12]


def fingerprint(W, X, name, policy, mode, block=None):
    """Identifies one exhaustive row: data, format definition, policy, mode and code version."""
    h = hashlib.sha1()
    for a in (W, X):
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(repr((name, spec(name), layout(name), policy, mode, block, _CODE_HASH)).encode())
    return h.hexdigest()[:16]


def exhaustive(W, X, names=None, policies=EXHAUSTIVE_POLICIES, block=None,
               checkpoint=None, progress=True, mode="w"):
    """Ground truth for the selector: quantize W with every format x policy.

    policies  "ocp", "ceil", "opt" (exponent search on weight MSE, uses no X) and
              "opt_out" (exponent search on the output-aware error, H fitted on the
              even columns of X only).
    mode      "w"  weights only, X stays FP32 (primary result, D2)
              "wx" X is quantized too, with the same format and policy, blocks down its
                   columns. For "opt_out" X's search uses H_X = W^T W (the error of Y
                   caused by an error dx in one column is W dx).
    Each row reports errors on all of X and on the held-out odd columns, the
    quantizer statistics, the cost (element roundings + MACs) and the wall time.
    checkpoint  path of a .jsonl file: finished rows are appended as they complete
                and reused on a rerun only if their fingerprint (data, format, policy,
                mode, formats.py version) matches, so an interrupted run resumes and a
                changed run recomputes.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    names = list(names or FORMATS)
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X_fit, X_eval = split_X(X)
    H_fit = X_fit @ X_fit.T
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as f:
            for line in f:
                r = json.loads(line)
                if "fingerprint" in r:
                    done[r["fingerprint"]] = r
    jobs = [(n, pol) for n in names for pol in policies]
    rows = []
    for n, pol in tqdm(jobs, desc=f"exhaustive[{mode}]", disable=not progress):
        fp = fingerprint(W, X, n, pol, mode, block)
        if fp in done:
            rows.append(done[fp])
            continue
        t0 = time.perf_counter()
        qpol, H = ("opt", H_fit) if pol == "opt_out" else (pol, None)
        Wq, _, st = quantize(W, n, block=block, policy=qpol, H=H, return_stats=True)
        Xq = Xq_eval = None
        n_round = st["n_roundings"]
        if mode == "wx":
            HX = W.T @ W if pol == "opt_out" else None
            Xq, _, stx = quantize_X(X, n, block=block, policy=qpol, H=HX, return_stats=True)
            Xq_eval = Xq[:, 1::2]
            n_round += stx["n_roundings"]
        m_all, m_ho = metrics(W, Wq, X, Xq), metrics(W, Wq, X_eval, Xq_eval)
        row = dict(
            format=n, policy=pol, mode=mode,
            y_rel_err=m_all["y_rel_err"], y_rel_err_heldout=m_ho["y_rel_err"],
            y_sqnr_db=m_all["y_sqnr_db"], w_rel_err=m_all["w_rel_err"],
            w_sqnr_db=m_all["w_sqnr_db"], y_cos_row_min=m_all["y_cos_row_min"],
            x_rel_err=m_all.get("x_rel_err", 0.0),
            n_range=st["n_range"], n_saturated=st["n_saturated"], n_zeroed=st["n_zeroed"],
            n_subnormal=st["n_subnormal"], n_scale_clamped=st["n_scale_clamped"],
            bits_per_elem=st["bits_per_elem"],
            n_roundings=n_round,
            n_macs=int(W.shape[0] * W.shape[1] * X.shape[1]),   # one W x X-sized matmul
            seconds=time.perf_counter() - t0,
            fingerprint=fp,
        )
        rows.append(row)
        if checkpoint:
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
            with open(checkpoint, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows


def ranking(rows, metric="y_rel_err", policies=None):
    """Best policy per format, sorted best first: [(format, policy, value), ...]."""
    best = {}
    for r in rows:
        if policies and r["policy"] not in policies:
            continue
        if r["format"] not in best or r[metric] < best[r["format"]][1]:
            best[r["format"]] = (r["policy"], r[metric])
    return sorted(((f, p, v) for f, (p, v) in best.items()), key=lambda t: t[2])


def save_results(rows, path):
    """Write rows to <path>.csv and <path>.json."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(path + ".json", "w") as f:
        json.dump(rows, f, indent=1)
