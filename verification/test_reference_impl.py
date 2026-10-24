"""
Golden + adversarial tests for reference_impl.py.  Run:  python test_reference_impl.py   (or pytest)
Every expected value below is worked out by hand from the conventions, not from code.
"""
from fractions import Fraction as F
import math
import numpy as np
import reference_impl as R

E3M2, E2M3, E4M1 = (R.FORMATS[k]["elem"] for k in ("MXFP_E3M2", "MXFP_E2M3", "MXFP_E4M1"))
E4M1r = R.FORMATS["MXFP_E4M1_reserved"]["elem"]
INT6, BFP6 = R.FORMATS["MXINT6"]["elem"], R.FORMATS["BFP6"]["elem"]


def q(elem, x, rnd="rne"):
    code, flag = elem.encode(F(x), rnd)
    return elem.decode(code)[0], flag, code


# ---------------- format parameters (class table + derived E4M1) ----------------
def test_params():
    assert (E3M2.emin, E3M2.emax, E3M2.vmax) == (-2, 4, 28)
    assert (E2M3.emin, E2M3.emax, E2M3.vmax) == (0, 2, F(15, 2))
    assert (E4M1.emin, E4M1.emax, E4M1.vmax) == (-6, 8, 384)
    assert (E4M1r.emax, E4M1r.vmax) == (7, 192)
    assert INT6.vmax == F(31, 16) and BFP6.vmax == F(31, 16)


def test_value_tables():
    for el, smallest in ((E3M2, F(1, 16)), (E2M3, F(1, 8)), (E4M1, F(1, 128))):
        vals = el.all_values()
        pos = [v for v in vals if v > 0]
        assert len(pos) == 2 ** (el.e + el.m) - 1          # 31 positive magnitudes
        assert min(pos) == smallest and max(pos) == el.vmax
        assert vals == sorted(set(-v for v in vals))       # sign-symmetric
    assert len([v for v in E4M1r.all_values() if v > 0]) == 29


def test_roundtrip_every_value():
    for el in (E3M2, E2M3, E4M1, E4M1r):
        for v in el.all_values():
            assert q(el, v)[0] == v, (el.name, v)
    for k in range(-31, 32):
        for el in (INT6, BFP6):
            assert q(el, F(k, 16))[0] == F(k, 16)


# ---------------- rounding: ties to even, carries, saturation ----------------
def test_rne_ties_fp():
    # E2M3 in [1,2): step 1/8
    assert q(E2M3, 1.0625)[0] == 1        # tie between 1.0 (mant 0) and 1.125 (mant 1) -> even 1.0
    assert q(E2M3, 1.1875)[0] == F(5, 4)  # tie between 1.125 (1) and 1.25 (2) -> 1.25
    assert q(E2M3, 1.9375)[0] == 2        # tie at top of binade -> carries into next exponent
    # subnormals (step 1/8 below 1)
    assert q(E2M3, 0.0625)[0] == 0 and q(E2M3, 0.0625)[1] == "zeroed"   # half of min subnormal -> 0
    assert q(E2M3, 0.0626)[0] == F(1, 8)
    assert q(E2M3, 0.1875)[0] == F(1, 4)  # n = 1.5 -> 2
    assert q(E2M3, 0.9375)[0] == 1        # top subnormal tie (7.5 -> 8) carries to min normal
    # E3M2 tie in [1,2): step 1/4
    assert q(E3M2, 1.125)[0] == 1 and q(E3M2, 1.375)[0] == F(3, 2)


def test_saturation_fp():
    assert q(E2M3, 7.6) == (F(15, 2), None, q(E2M3, 7.6)[2])
    assert q(E2M3, 7.75)[:2] == (F(15, 2), "sat")      # rounds up to 8 -> saturate
    assert q(E2M3, 1e9)[:2] == (F(15, 2), "sat")
    assert q(E3M2, -30)[:2] == (-28, "sat")
    assert q(E4M1, 400)[:2] == (384, None)             # 400/128 = 3.125 -> 3: ordinary rounding
    assert q(E4M1, 448)[:2] == (384, "sat")            # 3.5 tie -> 4 -> exponent overflow -> saturate
    assert q(E4M1r, 300)[:2] == (192, "sat")


def test_int_codes():
    assert q(INT6, F(31, 32) / 16 * 16)[0] == 1          # 0.96875*16 = 15.5 -> 16 (even)
    v, flag, code = q(INT6, F(63, 32))                   # k = 31.5 -> 32 -> saturate
    assert (v, flag) == (F(31, 16), "sat")
    v, flag, code = q(INT6, -F(63, 32))
    assert v == -F(31, 16) and code == 0b100001          # -31, never 0b100000 (-32)
    assert q(INT6, F(1, 32))[:2] == (0, "zeroed")        # k = 0.5 -> 0
    assert q(INT6, F(3, 32))[0] == F(2, 16)              # k = 1.5 -> 2
    # BFP sign-magnitude code layout and truncation mode
    assert q(BFP6, -F(5, 16))[2] == 0b100101
    assert q(BFP6, F(47, 32), "trunc")[0] == F(23, 16)   # k = 23.5 -> 23 (toward 0)


def test_signed_zero():
    assert E2M3.encode(F(0), neg=True)[0] == 0b100000   # -0.0 input keeps its sign
    assert q(E2M3, -0.01)[2] == 0b100000                 # negative value rounding to zero -> -0
    assert q(INT6, -0.001)[2] == 0                      # two's complement: no -0 code


# ---------------- shared exponent ----------------
def test_shared_exp():
    se = R.shared_exp
    z = [F(0)] * 32
    assert se(z, E2M3, "floor") == 0                    # all-zero block: scale 1, no log2(0)
    assert se([F(4)], E2M3, "floor") == 0               # floor(log2 4)=2, emax=2
    below4 = F(float(np.nextafter(4.0, 0)))
    assert se([below4], E2M3, "floor") == -1            # just below a power of two
    assert se([F(4)], INT6, "ceil_emax") == 2 and se([below4], INT6, "ceil_emax") == 2
    assert se([F(5)], INT6, "floor") == 2 and se([F(5)], INT6, "ceil_emax") == 3
    # team 'ceil' = smallest s with amax <= vmax*2^s
    assert se([F(5)], INT6, "ceil") == 2                # 5 <= 31/16*4 = 7.75
    assert se([F(1.9)], INT6, "ceil") == 0              # 1.9 <= 1.9375
    assert se([F(1.95)], INT6, "ceil") == 1
    assert se([F(31, 16)], INT6, "ceil") == 0           # exactly vmax: fits, no extra shift
    assert se([F(28)], E3M2, "ceil") == 0 and se([F(28.5)], E3M2, "ceil") == 1
    assert se([F(2.0 ** -200)], E2M3, "floor") == -127  # clamp
    assert se([F(2.0 ** 200)], E2M3, "floor") == 127
    # 'fit': INT6 block max 1.99 saturates under floor (k=32) -> fit moves to s=1
    assert se([F(1.99)], INT6, "floor") == 0 and se([F(1.99)], INT6, "fit") == 1
    assert se([F(1.9)], INT6, "fit") == 0               # k = 30.4 -> 30, fits


def test_floor_log2_exact():
    for p in range(-140, 140):
        x = 2.0 ** p
        assert R.floor_log2(x) == p
        assert R.floor_log2(float(np.nextafter(x, 0))) == p - 1
        assert R.floor_log2(float(np.nextafter(x, np.inf))) == p


# ---------------- whole-matrix behaviour ----------------
def test_partial_block_and_padding():
    W = np.zeros((1, 100)); W[0, 96:] = [1.0, -0.5, 0.25, 3.0]   # partial block of 4
    W[0, :96] = 100.0                                          # big values elsewhere must not leak
    Wh, st, (codes, scales) = R.quantize_matrix(W, "MXFP_E2M3", return_codes=True)
    assert st["n"] == 100 and len(scales[0]) == 4              # padding not counted, 4 blocks
    assert np.array_equal(Wh[0, 96:], [1.0, -0.5, 0.25, 3.0])  # max 3 -> s=-1: exact
    Wh2, st2 = R.quantize_matrix(W, "BFP6_row100")
    assert st2["n"] == 100


def test_outlier_crushes_int_block():
    # one outlier at 32 forces s=5 for INT6: step 2^(5-4) = 2, so 0.9 -> 0
    W = np.full((1, 32), 0.9); W[0, 0] = 32.0
    Wi, si = R.quantize_matrix(W, "MXINT6")
    Wf, sf = R.quantize_matrix(W, "MXFP_E2M3")
    assert si["zeroed"] == 31 and np.all(Wi[0, 1:] == 0)
    assert sf["zeroed"] == 0                                   # FP element exponent absorbs it


def test_e4m1_one_nan():
    el = R.FORMATS["MXFP_E4M1_onenan"]["elem"]
    assert (el.emax, el.vmax) == (8, 256)
    pos = [v for v in el.all_values() if v > 0]
    assert len(pos) == 30 and max(pos) == 256                     # 384 is the NaN code
    assert q(el, 300)[:2] == (256, None)                            # 300/128 = 2.34 -> 2 -> 256
    assert q(el, 320)[:2] == (256, None)                            # tie 2.5 -> even 2 -> 256, no clipping
    assert q(el, 330)[:2] == (256, "sat")                           # 2.58 -> 3 = 384 = NaN code -> saturate
    assert q(el, 1e9)[:2] == (256, "sat")
    # class ocp rule with emax 8: block max 7 -> s = 2-8 = -6 -> 7*64 = 448 -> saturates to 256
    W = np.zeros((1, 32)); W[0, 0] = 7.0
    assert R.quantize_matrix(W, "MXFP_E4M1_onenan")[0][0, 0] == 4.0


def test_bfp6_16_4_5():
    f = R.FORMATS["BFP6"]
    assert (f["block"], f["exp_lo"], f["exp_hi"], f["rule"]) == (16, -8, 7, "floor")
    W = np.zeros((1, 100)); W[0, :100] = 0.0
    W[0, 0] = 1.5; W[0, 1] = 0.1                                   # group 0: e = 0, step 1/16
    W[0, 16] = 2.0 ** -14; W[0, 17] = 2.0 ** -12                   # group 1: e = -12 -> clamped to -8, step 2^-12
    W[0, 32] = 300.0                                                # group 2: e = 8 -> clamped to 7: 31/16*128 = 248
    W[0, 97] = -5.0                                                 # tail group of 4 (96..99): e = 2
    Wh, st, (codes, scales) = R.quantize_matrix(W, "BFP6", return_codes=True)
    assert len(scales[0]) == 7 and st["n"] == 100
    assert [s if s < 8 else s - 16 for s in scales[0][:3]] == [0, -8, 7]   # 4-bit two's complement codes
    assert Wh[0, 0] == 1.5 and Wh[0, 1] == F(2, 16)                 # 0.1*16 = 1.6 -> 2
    assert Wh[0, 16] == 0.0 and Wh[0, 17] == 2.0 ** -12             # below step/2 -> 0; exactly one step -> kept
    assert Wh[0, 32] == 248.0 and st["sat"] >= 1
    assert Wh[0, 97] == -5.0
    # BFP 32|8|5 with the floor rule is numerically MXINT6 (D1 sentence in the report)
    Wr = np.random.default_rng(3).laplace(size=(10, 100))
    assert np.array_equal(R.quantize_matrix(Wr, "BFP6_32_8_floor")[0], R.quantize_matrix(Wr, "MXINT6")[0])


def test_matrix_signed_zero():
    W = np.array([[-0.0, -1e-30, 1.0] + [0.0] * 29])
    Wh, _ = R.quantize_matrix(W, "MXFP_E2M3")
    assert np.signbit(Wh[0, 0]) and np.signbit(Wh[0, 1])
    Wi, _ = R.quantize_matrix(W, "MXINT6")
    assert not np.signbit(Wi[0, 1])                      # INT6 has no -0


def test_opt_policy():
    # block max 1.99 in INT6: floor (s=0) saturates 1.99 -> 1.9375 (err 0.0525^2),
    # s=+1 gives step 1/8: 1.99 -> 2.0 (err 0.01^2) but 0.03 -> 0 (err 0.03^2); exact sums decide
    W = np.zeros((1, 32)); W[0, 0] = 1.99; W[0, 1] = 0.03
    f = dict(R.FORMATS["MXINT6"]); f["rule"] = "opt"
    Wh, _, (_, scales) = R.quantize_matrix(W, f, return_codes=True)
    e_floor = (F(1.99) - F(31, 16)) ** 2 + (F(0.03) - F(1, 32)) ** 2      # 0.03 -> 0.5/16 = 1/32
    e_up = (F(1.99) - 2) ** 2 + F(0.03) ** 2
    assert scales[0][0] - 127 == (0 if e_floor <= e_up else 1)
    Wz, _ = R.quantize_matrix(np.zeros((2, 100)), f)                       # all-zero: offset 0 kept
    assert not Wz.any()


def test_bits_per_element():
    b = R.bits_per_element("MXINT6")
    assert b["nominal"] == 6.25 and b["d11"] == 6.32 and b["padded"] == 8.0
    b = R.bits_per_element("BFP6")                                  # 16|4|5: 7 groups per row
    assert b["nominal"] == 6.25 and b["d11"] == 6.28
    assert R.bits_per_element("BFP6_row100")["d11"] == 6.08


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"all {len(tests)} tests passed")
