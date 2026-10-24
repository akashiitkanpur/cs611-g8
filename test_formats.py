"""Unit tests for formats.py.  Run:  python test_formats.py   (or: pytest test_formats.py)

The FP checks compare against a grid built independently from the bit codes
(exponent field x mantissa field), not against fp_round's own arithmetic.
Hand examples (BFP6, saturation counts, W+X) are worked out in the comments.

v2 (22 Oct): two v1 tests changed because D1 changed what "BFP6" means
(test_bfp6_variants_vs_mxint6, test_exponent_field_clamped); every other v1 test
is unchanged. New tests are marked "v2".
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from formats import (FORMATS, VARIANTS, E8M0_MIN, E8M0_MAX, spec, layout, fp_round,  # noqa: E402
                     int_round, quantize, quantize_X, metrics, storage_bits, format_storage,
                     exhaustive, ranking, split_X, vmax_elem)

FP_NAMES = ["MXFP6_E3M2", "MXFP6_E2M3", "MXFP6_E4M1", "MXFP6_E4M1_v192", "MXFP6_E4M1_nan1"]
ALL = list(FORMATS) + list(VARIANTS)
SYMMETRIC = [n for n in ALL if n != "MXINT6"]             # two's complement is asymmetric (D10)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def bit_grid(ebits, mbits, bias, maxval):
    """Every non-negative value the format can encode, with its mantissa code."""
    vals, mant = [], []
    for E in range(2 ** ebits):
        for M in range(2 ** mbits):
            if E == 0:
                v = (M / 2 ** mbits) * 2.0 ** (1 - bias)          # subnormal
            else:
                v = (1 + M / 2 ** mbits) * 2.0 ** (E - bias)       # normal
            if v <= maxval:
                vals.append(v)
                mant.append(M)
    return np.array(vals), np.array(mant)


def ref_round(x, grid, mant):
    """Nearest grid value; ties go to the even mantissa code; saturate at the top."""
    out = np.empty_like(x)
    for i, v in enumerate(x):
        a = min(abs(v), grid.max())
        d = np.abs(grid - a)
        best = np.flatnonzero(d == d.min())
        j = best[0] if len(best) == 1 else [b for b in best if mant[b] % 2 == 0][0]
        out[i] = np.copysign(grid[j], v)
    return out


def row(*vals, n=32):
    w = np.zeros((1, n))
    w[0, :len(vals)] = vals
    return w


def load_C():
    path = os.path.join(HERE, "data", "W_C.npy")
    if not os.path.exists(path):
        return None, None
    W = np.load(path).astype(np.float64)
    X = np.load(os.path.join(HERE, "data", "X_cal.npy")).astype(np.float64)
    return W, X


def yerr(W, X, name, **kw):
    return 100 * metrics(W, quantize(W, name, **kw)[0], X)["y_rel_err"]


# ---------------------------------------------------------------- element level
def test_fp_grid_limits():
    expect = {"MXFP6_E3M2": (28, 0.0625, 4), "MXFP6_E2M3": (7.5, 0.125, 2),
              "MXFP6_E4M1": (384, 2 ** -7, 8), "MXFP6_E4M1_v192": (192, 2 ** -7, 7),
              "MXFP6_E4M1_nan1": (256, 2 ** -7, 8)}
    for n in FP_NAMES:
        _, p, emax = spec(n)
        g, _ = bit_grid(**p)
        vmax, min_sub, e = expect[n]
        assert g.max() == vmax == p["maxval"], n
        assert g[g > 0].min() == min_sub, n
        assert emax == e and g.max() < 2.0 ** (emax + 1) and g.max() >= 2.0 ** emax, n


def test_fp_round_matches_bit_grid():
    for n in FP_NAMES:
        _, p, _ = spec(n)
        g, mant = bit_grid(**p)
        mids = (g[:-1] + g[1:]) / 2                               # exact RNE ties
        x = np.concatenate([rng.uniform(-1.3 * p["maxval"], 1.3 * p["maxval"], 5000),
                            mids, -mids, g, -g, [g[1] / 2, g[1] / 2 * 0.999, 1e-30, 1e9, -1e9]])
        assert np.array_equal(fp_round(x, **p), ref_round(x, g, mant)), n


def test_saturation_no_inf_nan():
    for n in FP_NAMES:
        _, p, _ = spec(n)
        q = fp_round(np.array([1e30, -1e30]), **p)
        assert np.all(np.isfinite(q)) and q[0] == p["maxval"] and q[1] == -p["maxval"], n


def test_mxint6_codes_and_ties():
    x = np.concatenate([rng.uniform(-3, 3, 5000), (np.arange(-63, 64) + 0.5) / 16])
    k = int_round(x, 4, 31) * 16
    assert np.array_equal(k, np.round(k))
    assert k.min() == -31 and k.max() == 31                       # -32 never produced
    assert list(int_round(np.array([0.5, 1.5, 2.5, -0.5]) / 16, 4, 31) * 16) == [0, 2, 2, 0]


# ---------------------------------------------------------------- scale + blocks
def test_all_zero_block_scale_one():
    for n in ALL:
        Wq, e, st = quantize(np.zeros((3, 100)), n, return_stats=True)
        assert np.all(Wq == 0) and np.all(e == 0), n
        for pol in ("ceil", "opt"):
            assert np.all(quantize(np.zeros((2, 64)), n, policy=pol)[1] == 0), (n, pol)


def test_exponent_field_clamped():
    # v2: the stored shared-exponent field must stay in its own range:
    # E8M0 [-127, 127] for MX and 8-bit BFP, [-8, 7] for BFP6 16|4|5.
    W = np.array([[2.0 ** -200] * 32, [2.0 ** 200] * 32])
    for n in ALL:
        lo, hi = layout(n)["field"]
        for pol in ("ocp", "ceil", "opt"):
            e = quantize(W, n, policy=pol)[1]
            assert e.min() >= lo and e.max() <= hi, (n, pol, e.min(), e.max())
    assert layout("MXFP6_E2M3")["field"] == (E8M0_MIN, E8M0_MAX)


def test_power_of_two_block_max():
    # block max exactly 4 = 2^2: exactly representable under the class rule
    for n in ["MXFP6_E3M2", "MXFP6_E2M3", "MXFP6_E4M1", "MXINT6", "BFP6"]:
        Wq, _ = quantize(row(4.0, -4.0, 1.0), n)
        assert Wq[0, 0] == 4.0 and Wq[0, 1] == -4.0, n


def test_just_below_power_of_two():
    # max = 3.999: the "ocp" floor rule saturates it for E2M3 (7.5 * 2^-1 = 3.75);
    # "ceil" picks the next scale up and keeps it (rounds to 4.0). Known trade-off.
    assert quantize(row(3.999), "MXFP6_E2M3")[0][0, 0] == 3.75
    Wq, _, st = quantize(row(3.999), "MXFP6_E2M3", policy="ceil", return_stats=True)
    assert Wq[0, 0] == 4.0 and st["n_saturated"] == 0


def test_one_ulp_below_power_of_two():
    # np.floor(np.log2(x)) rounds up here for float64; the scale must still follow the rule
    for k in range(-10, 12):
        a = np.nextafter(2.0 ** k, 0)
        for n in ["MXFP6_E3M2", "MXFP6_E2M3", "MXFP6_E4M1", "MXINT6"]:
            kind, p, emax = spec(n)
            e = quantize(row(a), n, policy="ocp")[1][0, 0]
            assert e == (k - 1) - emax, (n, k, e)
    # E2M3, block max 8 - ulp: scale 1 (not 2), so the max saturates to 7.5
    assert quantize(row(np.nextafter(8.0, 0)), "MXFP6_E2M3")[0][0, 0] == 7.5
    # ceil: a max one ulp above vmax * 2^k must move to the next scale (nothing clipped)
    # v2: BFP6 16|4|5 has a 4-bit exponent (E <= 7), so its largest value is 31 * 2^(7-4) = 248;
    # beyond that even "ceil" must clip (k = 3 here needs E = 8). Checked separately below.
    for n in ["MXFP6_E2M3", "MXINT6", "BFP6"]:
        kind, p, _ = spec(n)
        vm = vmax_elem(kind, p)
        for k in range(-5, 3 if n == "BFP6" else 6):
            st = quantize(row(np.nextafter(vm * 2.0 ** k, np.inf)), n, policy="ceil", return_stats=True)[2]
            assert st["n_range"] == 0 and st["n_saturated"] == 0, (n, k)
    st = quantize(row(np.nextafter(248.0, np.inf)), "BFP6", policy="ceil", return_stats=True)[2]
    assert st["n_scale_clamped"] == 1 and st["n_range"] == 1


def test_rne_tie_inside_quantize():
    # E2M3, block max 7.5 -> scale 1. 1.0625 is a tie between 1.0 and 1.125 -> even code (1.0)
    Wq, e = quantize(row(7.5, 1.0625, 1.1875), "MXFP6_E2M3")
    assert e[0, 0] == 0 and Wq[0, 1] == 1.0 and Wq[0, 2] == 1.25


def test_tail_block_of_four():
    W = rng.standard_normal((5, 100))
    W[:, 96:] *= 0.01                                            # tiny tail
    Wq, e, st = quantize(W, "MXFP6_E2M3", return_stats=True)
    assert e.shape == (5, 4)                                     # 32 + 32 + 32 + 4
    expect = np.floor(np.log2(np.abs(W[:, 96:]).max(1))) - 2     # tail scale from its own 4 values
    assert np.array_equal(e[:, 3], expect)
    assert st["n_elements"] == 500                               # no padded elements counted
    assert abs(st["bits_per_elem"] - 6.32) < 1e-12               # tail pays its own 8-bit scale


def test_sign_symmetry():
    W = rng.laplace(size=(20, 100))
    for n in SYMMETRIC:
        for pol in ("ocp", "ceil", "opt"):
            assert np.array_equal(quantize(-W, n, policy=pol)[0], -quantize(W, n, policy=pol)[0]), (n, pol)


# ---------------------------------------------------------------- policies
def test_ceil_never_saturates_and_opt_never_worse():
    W = rng.standard_t(3, (30, 100))
    for n in ALL:
        assert quantize(W, n, policy="ceil", return_stats=True)[2]["n_range"] == 0, n
        e_ocp = ((W - quantize(W, n, policy="ocp")[0]) ** 2).sum(1)
        e_opt = ((W - quantize(W, n, policy="opt")[0]) ** 2).sum(1)
        assert np.all(e_opt <= e_ocp + 1e-12), n


def test_bfp6_variants_vs_mxint6():
    # v2 (D1): BFP6 is now 16|4|5 and is NOT MXINT6. The old block-32 "ceil" BFP6 is kept
    # as BFP6_b32_ceil and is still identical to MXINT6 under ceil (documented equivalence).
    W = rng.standard_normal((40, 100))
    assert np.array_equal(quantize(W, "BFP6_b32_ceil")[0], quantize(W, "MXINT6", policy="ceil")[0])
    assert not np.array_equal(quantize(W, "BFP6")[0], quantize(W, "MXINT6")[0])
    # the whole-row variant is genuinely different
    assert not np.array_equal(quantize(W, "BFP6_row")[0], quantize(W, "MXINT6")[0])
    assert quantize(W, "BFP6_row")[1].shape == (40, 1)


def test_storage():
    assert storage_bits(100, 128)[1] == 6.25
    assert abs(storage_bits(100, 100)[1] - 6.32) < 1e-12
    assert abs(storage_bits(100, 100, block="row")[1] - 6.08) < 1e-12


def test_metrics_basic():
    W = rng.standard_normal((10, 100)); X = rng.standard_normal((100, 50))
    m = metrics(W, W, X)
    assert m["y_rel_err"] == 0 and m["w_rel_err"] == 0 and abs(m["y_cos_row_min"] - 1) < 1e-12
    m = metrics(W, np.zeros_like(W), X)
    assert abs(m["y_rel_err"] - 1) < 1e-12 and m["zeroed_frac"] == 1 and m["y_cos_row_mean"] == 0


def test_exhaustive_and_checkpoint(tmp=None):
    W = rng.standard_normal((8, 64)); X = rng.standard_normal((64, 20))
    ck = os.path.join(tmp or tempfile.mkdtemp(), "ck.jsonl")
    rows = exhaustive(W, X, checkpoint=ck, progress=False)
    assert len(rows) == 5 * 4 and sum(1 for _ in open(ck)) == 20
    again = exhaustive(W, X, checkpoint=ck, progress=False)     # resumes from checkpoint
    assert [r["y_rel_err"] for r in again] == [r["y_rel_err"] for r in rows]
    assert len(ranking(rows)) == 5


# ---------------------------------------------------------------- v2: BFP6 = 16|4|5 (D1)
def test_v2_bfp6_layout_and_hand_example():
    lay = layout("BFP6")
    assert lay["block"] == 16 and lay["scale_bits"] == 4 and lay["field"] == (-8, 7) and lay["policy"] == "ocp"
    # Block 0 (cols 0-15): max 3.0 -> stored exponent E = floor(log2 3) = 1, step 2^(1-4) = 0.125.
    #   3.0    -> 24 * 0.125 = 3.0 (exact)
    #   1.0625 -> 8.5 steps, a tie -> even code 8 -> 1.0
    #   1.1875 -> 9.5 steps, a tie -> even code 10 -> 1.25
    #   0.05   -> 0.4 steps -> 0 (zeroed)
    # Block 1 (cols 16-31): max 3.99 -> E = 1, 31.92 steps -> 32 -> saturates to 31 -> 3.875
    #   (the class floor rule clips the block max, exactly as MXINT6 does)
    w = np.zeros((1, 100))
    w[0, :4] = [3.0, 1.0625, 1.1875, 0.05]
    w[0, 16] = -3.99
    w[0, 96:] = [0.5, 0.25, 0.0, -0.125]           # tail block of 4: E = -1, step 2^-5
    Wq, e, st = quantize(w, "BFP6", return_stats=True)
    assert e.shape == (1, 7)                       # 6 blocks of 16 + a tail of 4
    assert list(Wq[0, :4]) == [3.0, 1.0, 1.25, 0.0]
    assert Wq[0, 16] == -3.875
    assert list(Wq[0, 96:]) == [0.5, 0.25, 0.0, -0.125]
    assert e[0, 0] == 1 and e[0, 1] == 1 and e[0, 6] == -1 and e[0, 2] == 0   # all-zero block -> 0
    assert st["n_zeroed"] == 1 and st["n_range"] == 1 and st["n_saturated"] == 1
    k = Wq[0, :16] / 2.0 ** (e[0, 0] - 4)
    assert np.array_equal(k, np.round(k)) and np.abs(k).max() <= 31


def test_v2_bfp6_4bit_exponent_clamp():
    # block max 2^10 -> E = 10, clamped to 7 -> step 2^3, largest value 31 * 8 = 248 (saturates)
    # block max 2^-14 -> E = -14, clamped to -8 -> step 2^-12, 2^-14 is 0.25 steps -> 0
    w = np.zeros((2, 16))
    w[0, 0], w[0, 1] = 2.0 ** 10, 3.0
    w[1, 0] = 2.0 ** -14
    Wq, e, st = quantize(w, "BFP6", return_stats=True)
    assert e[0, 0] == 7 and e[1, 0] == -8
    assert Wq[0, 0] == 248.0 and Wq[0, 1] == 0.0   # 3 / 8 = 0.375 steps -> 0
    assert Wq[1, 0] == 0.0
    assert st["n_scale_clamped"] == 2
    # the 8-bit-exponent variants do not clamp here
    assert quantize(w, "BFP6_g32e8", return_stats=True)[2]["n_scale_clamped"] == 0


def test_v2_bfp_32_8_5_identical_to_mxint6_sym():
    # D1: BFP(32 values, 8-bit exponent, floor, RNE) == symmetric MXINT6, including the stored exponent
    W = rng.standard_t(4, (30, 100))
    W[0, :32] = 2.0 ** -200
    W[1, 32:64] = 2.0 ** 200
    for pol in ("ocp", "ceil", "opt"):
        a, ea = quantize(W, "BFP6_g32e8", policy=pol)
        b, eb = quantize(W, "MXINT6_sym", policy=pol)
        assert np.array_equal(a, b) and np.array_equal(ea, eb), pol


def test_v2_bfp6_on_dataset_C():
    W, X = load_C()
    if W is None:
        return
    assert abs(yerr(W, X, "BFP6") - 3.46) < 0.006                # plan D1 prototype value
    assert abs(yerr(W, X, "BFP6_g32e8") - yerr(W, X, "MXINT6_sym")) == 0


# ---------------------------------------------------------------- v2: E4M1 one-NaN (D3)
def test_v2_e4m1_one_nan_variant():
    kind, p, emax = spec("MXFP6_E4M1_nan1")
    assert p["maxval"] == 256.0 and emax == 8 and p["bias"] == 7
    # 300: binade 2^8, step 128 -> 2.34 steps -> 256;  330 -> 2.58 -> 384, not a code -> saturate 256
    assert list(fp_round(np.array([300.0, 330.0, -1e9]), **p)) == [256.0, 256.0, -256.0]
    W, X = load_C()
    if W is None:
        return
    assert abs(yerr(W, X, "MXFP6_E4M1_nan1") - 20.7) < 0.06       # plan D3 (class rule)
    c = [yerr(W, X, n, policy="ceil") for n in ("MXFP6_E4M1", "MXFP6_E4M1_nan1", "MXFP6_E4M1_v192")]
    # with ceil all three agree (plan D3). Not bit-identical: the scaled grids differ only at
    # the subnormal end, which moves the error by ~1e-6 percentage points.
    assert max(c) - min(c) < 1e-5 and abs(c[0] - 10.18) < 0.006


# ---------------------------------------------------------------- v2: class saturation counts
def test_v2_range_vs_saturated_counts():
    # Lecture 19: "range" = |w|/scale > vmax BEFORE rounding; "saturated" = the ROUNDED value
    # exceeds vmax. E2M3, block max 3.8 -> scale 2^-1 -> 7.6 (> 7.5, out of range), but in the
    # binade 2^2 the step is 0.5 and 7.6 rounds to 7.5, which is representable: not saturated.
    st = quantize(row(3.8), "MXFP6_E2M3", return_stats=True)[2]
    assert st["n_range"] == 1 and st["n_saturated"] == 0
    st = quantize(row(3.999), "MXFP6_E2M3", return_stats=True)[2]   # 7.998 rounds to 8.0 > 7.5
    assert st["n_range"] == 1 and st["n_saturated"] == 1


# ---------------------------------------------------------------- v2: MXINT6 two's complement (D10)
def test_v2_mxint6_twos_complement():
    # block [-1.99]: E = 0, -1.99 * 16 = -31.84 -> -32 is a code in two's complement (-2.0),
    # symmetric MXINT6 clips to -31 (-1.9375)
    assert quantize(row(-1.99), "MXINT6")[0][0, 0] == -2.0
    assert quantize(row(-1.99), "MXINT6_sym")[0][0, 0] == -1.9375
    assert quantize(row(1.99), "MXINT6")[0][0, 0] == 1.9375       # +32 is not a code


# ---------------------------------------------------------------- v2: storage (D11)
def test_v2_true_storage_per_format():
    expect = {n: 6.32 for n in FORMATS}
    expect.update({"BFP6": 6.28, "BFP6_g32e8": 6.32, "BFP6_b32_ceil": 6.32, "BFP6_row": 6.08,
                   "MXFP6_E4M1_v192": 6.32, "MXFP6_E4M1_nan1": 6.32, "MXINT6_sym": 6.32})
    for n in ALL:
        total, per = format_storage(n, 100, 100)
        assert abs(per - expect[n]) < 1e-12, (n, per)
        st = quantize(rng.standard_normal((100, 100)), n, return_stats=True)[2]
        assert abs(st["bits_per_elem"] - expect[n]) < 1e-12, n
    # BFP6 16|4|5 by hand: 7 blocks per row x 4 bits = 28, (600 + 28) / 100 = 6.28
    assert format_storage("BFP6", 1, 100)[0] == 628


# ---------------------------------------------------------------- v2: W+X mode (D2)
def test_v2_quantize_X_blocks_run_down_columns():
    X = rng.standard_normal((100, 7))
    Xq, e = quantize_X(X, "MXFP6_E2M3")
    assert Xq.shape == X.shape and e.shape == (4, 7)             # 32+32+32+4 down each column
    assert np.array_equal(Xq, quantize(X.T, "MXFP6_E2M3")[0].T)
    # columns are independent: quantizing a slice of columns = slicing the quantized X
    assert np.array_equal(quantize_X(X[:, 1::2], "MXINT6")[0], quantize_X(X, "MXINT6")[0][:, 1::2])
    # one large value in column 0 changes only column 0's first block
    X2 = X.copy(); X2[0, 0] = 1e3
    d = quantize_X(X2, "MXINT6")[0] != quantize_X(X, "MXINT6")[0]
    assert d[:32, 0].any() and not d[32:, 0].any() and not d[:, 1:].any()


def test_v2_metrics_with_quantized_X():
    W = rng.standard_normal((10, 100)); X = rng.standard_normal((100, 50))
    Wq = quantize(W, "MXFP6_E2M3")[0]; Xq = quantize_X(X, "MXFP6_E2M3")[0]
    m_w = metrics(W, Wq, X)
    assert m_w == metrics(W, Wq, X, Xq=None)                     # old call unchanged
    m_wx = metrics(W, Wq, X, Xq=Xq)
    Y = W @ X
    ref = np.linalg.norm(Y - Wq @ Xq) / np.linalg.norm(Y)
    assert abs(m_wx["y_rel_err"] - ref) < 1e-15
    assert m_wx["w_rel_err"] == m_w["w_rel_err"] and "x_rel_err" in m_wx


def test_v2_exhaustive_wx_mode():
    W = rng.standard_normal((8, 64)); X = rng.standard_normal((64, 20))
    rows = exhaustive(W, X, policies=("ocp",), mode="wx", progress=False)
    assert all(r["mode"] == "wx" for r in rows) and len(rows) == 5
    Xq = quantize_X(X, "MXINT6")[0]
    Wq = quantize(W, "MXINT6")[0]
    r = [r for r in rows if r["format"] == "MXINT6"][0]
    assert abs(r["y_rel_err"] - np.linalg.norm(W @ X - Wq @ Xq) / np.linalg.norm(W @ X)) < 1e-15
    _, Xe = split_X(X)
    ho = np.linalg.norm(W @ Xe - Wq @ Xq[:, 1::2]) / np.linalg.norm(W @ Xe)
    assert abs(r["y_rel_err_heldout"] - ho) < 1e-15


def test_v2_checkpoint_rejects_stale_rows():
    # v1 keyed the checkpoint by (format, policy) only, so a rerun after W, X or a format
    # definition changed silently returned old numbers. v2 stores a fingerprint per row.
    d = tempfile.mkdtemp()
    ck = os.path.join(d, "ck.jsonl")
    W = rng.standard_normal((8, 64)); X = rng.standard_normal((64, 20))
    a = exhaustive(W, X, names=["MXINT6"], policies=("ocp",), checkpoint=ck, progress=False)
    b = exhaustive(2 * W, X, names=["MXINT6"], policies=("ocp",), checkpoint=ck, progress=False)
    assert b[0]["fingerprint"] != a[0]["fingerprint"]
    c = exhaustive(W, X, names=["MXINT6"], policies=("ocp",), mode="wx", checkpoint=ck, progress=False)
    assert c[0]["mode"] == "wx" and c[0]["y_rel_err"] != a[0]["y_rel_err"]


# ---------------------------------------------------------------- regression on dataset C
def test_dataset_C_matches_prototype():
    W, X = load_C()
    if W is None:
        print("  (skipped: data/W_C.npy not found)")
        return
    proto = {  # % Y error from the team plan (ocp, ceil, block-optimal)
        "MXFP6_E2M3": (3.08, 3.10, 3.06), "MXINT6_sym": (4.05, 4.17, 3.90),
        "MXINT6": (4.05, 4.17, 3.85),                  # two's complement (D10): only opt moves
        "MXFP6_E3M2": (5.44, 5.24, 5.24), "MXFP6_E4M1": (11.09, 10.18, 10.18),
    }
    for n, vals in proto.items():
        for pol, v in zip(("ocp", "ceil", "opt"), vals):
            got = yerr(W, X, n, policy=pol)
            assert abs(got - v) < 0.006, (n, pol, round(got, 3), v)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
