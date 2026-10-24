"""
CS611 Assignment 1, Group 8: INDEPENDENT REFERENCE IMPLEMENTATION (Siddhant)

Written from the conventions (Lecture 19 OCP-MX notebook + plan v2 decisions),
NOT from formats.py. Deliberately slow and scalar:
  * every value is handled as an exact rational (fractions.Fraction),
  * the shared exponent comes from math.frexp (exact), never from log2,
  * every element is encoded to an actual bit pattern and decoded back.
So any disagreement with the vectorized library points at a real bug.

Conventions implemented (all configurable, see FORMATS / make_bfp):
  Block       : 32 along the last axis (rows of W); last partial block zero-padded.
                Padding never contributes to statistics.
  Scale       : power of two, E8M0 (biased 127), exponent clamped to [-127, 127];
                all-zero block -> scale 2^0.
  Scale rule  : 'floor' (class `ocp`): floor(log2(max)) - emax_elem
                'ceil'               : ceil(log2(max))  - emax_elem
                'fit'                : smallest exponent with no saturation after rounding
  Rounding    : round-to-nearest, ties-to-even ('rne') or truncation toward 0 ('trunc');
                then saturate at vmax (FP6/INT6 have no Inf/NaN).
  MXFP ExMy   : bias = 2^(e-1)-1, top exponent code finite,
                emin = 1-bias, emax = 2^e-1-bias, vmax = 2^emax*(2-2^-m),
                subnormal spacing 2^(emin-m). Sign of zero kept (like IEEE).
  MXINT6      : 6-bit two's complement k in [-31, 31] (code -32 never produced),
                value = k * 2^-4 * scale, emax_elem = 0.
  BFP6        : sign + 5-bit magnitude M in [0, 31], value = M * 2^-4 * scale,
                emax_elem = 0. Block size / scale rule / rounding per decision D1.
"""
from fractions import Fraction
import math
import numpy as np

# --------------------------------------------------------------------------
# exact helpers
# --------------------------------------------------------------------------

def pow2(n):
    """Exact 2^n as a Fraction (n may be negative)."""
    return Fraction(2) ** n


def floor_log2(x):
    """floor(log2 x) for x > 0, exact. x is a float or Fraction."""
    x = Fraction(x)
    assert x > 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    # now 2^(e-1) < x < 2^(e+1); fix up exactly
    while pow2(e) > x:
        e -= 1
    while pow2(e + 1) <= x:
        e += 1
    return e


def ceil_log2(x):
    e = floor_log2(x)
    return e if pow2(e) == Fraction(x) else e + 1


def round_int(q, mode):
    """Round a non-negative Fraction q to an integer."""
    assert q >= 0
    n = q.numerator // q.denominator          # floor
    if mode == "trunc":
        return n
    r = q - n
    if r > Fraction(1, 2):
        return n + 1
    if r < Fraction(1, 2):
        return n
    return n if n % 2 == 0 else n + 1         # tie -> even


# --------------------------------------------------------------------------
# element formats: encode(Fraction already divided by scale) -> code, decode(code) -> Fraction
# --------------------------------------------------------------------------

class MXFP:
    kind = "fp"

    def __init__(self, name, e, m, top_code_finite=True, one_nan=False):
        """top_code_finite=False: whole top exponent code reserved (IEEE-like).
        one_nan=True: only top exponent + all-ones mantissa is NaN (E4M3-style)."""
        self.name, self.e, self.m = name, e, m
        self.bits = 1 + e + m
        self.bias = 2 ** (e - 1) - 1
        self.emin = 1 - self.bias
        top = 2 ** e - 1 if top_code_finite else 2 ** e - 2
        self.emax = top - self.bias
        self.max_exp_field = top
        self.one_nan = one_nan
        self.max_mant = 2 ** m - 2 if one_nan else 2 ** m - 1      # largest usable mantissa at emax
        self.vmax = pow2(self.emax) * (1 + Fraction(self.max_mant, 2 ** m))
        self.emax_elem = self.emax

    def encode(self, x, rounding="rne", neg=None):
        """x: Fraction. Returns (code, flags). flags in {'sat','zeroed','sub', None}.
        neg: pass the float's sign bit so that an input of -0.0 keeps its sign."""
        sign = 1 if (x < 0 or neg) else 0
        a = abs(x)
        m = self.m
        flag = None
        if a == 0:
            ef, mant = 0, 0
        elif a < pow2(self.emin):                       # subnormal range
            n = round_int(a / pow2(self.emin - m), rounding)   # n in [0, 2^m]
            ef, mant = divmod(n, 2 ** m)                # n == 2^m -> carries into min normal
            if n == 0:
                flag = "zeroed"
            elif ef == 0:
                flag = "sub"
        else:
            E = floor_log2(a)
            n = round_int(a / pow2(E - m), rounding)    # n in [2^m, 2^(m+1)]
            if n == 2 ** (m + 1):
                E, n = E + 1, 2 ** m
            if E > self.emax or (E == self.emax and n - 2 ** m > self.max_mant):   # saturate
                ef, mant, flag = self.max_exp_field, self.max_mant, "sat"
            else:
                ef, mant = E + self.bias, n - 2 ** m
        code = (sign << (self.e + m)) | (ef << m) | mant
        return code, flag

    def decode(self, code):
        m = self.m
        sign = (code >> (self.e + m)) & 1
        ef = (code >> m) & (2 ** self.e - 1)
        mant = code & (2 ** m - 1)
        assert ef <= self.max_exp_field, f"{self.name}: reserved exponent code {ef}"
        assert not (ef == self.max_exp_field and mant > self.max_mant), f"{self.name}: NaN code"
        if ef == 0:
            mag = mant * pow2(self.emin - m)
        else:
            mag = (2 ** m + mant) * pow2(ef - self.bias - m)
        return -mag if sign else mag, sign

    def all_values(self):
        vals = set()
        for c in range(2 ** self.bits):
            ef, mant = (c >> self.m) & (2 ** self.e - 1), c & (2 ** self.m - 1)
            if ef > self.max_exp_field or (ef == self.max_exp_field and mant > self.max_mant):
                continue
            vals.add(self.decode(c)[0])
        return sorted(vals)


class IntSym:
    """Shared-scale integer element: value = k * 2^-frac, |k| <= kmax.
    signmag=False -> two's complement code (MXINT6); True -> sign + magnitude (BFP6)."""
    kind = "int"

    def __init__(self, name, bits=6, signmag=False, full_twos=False):
        """full_twos=True: two's complement with the -2^(n-1) code allowed (k in [-32, 31])."""
        self.name, self.bits, self.signmag = name, bits, signmag
        self.frac = bits - 2                      # implicit scale 2^-(n-2)
        self.kmax = 2 ** (bits - 1) - 1           # 31: symmetric
        self.full_twos = full_twos
        self.kneg = self.kmax + 1 if full_twos else self.kmax   # largest |k| for negatives
        self.vmax = Fraction(self.kmax, 2 ** self.frac)
        self.emax_elem = 0

    def encode(self, x, rounding="rne", neg=None):
        sign = 1 if (x < 0 or neg) else 0
        k = round_int(abs(x) * 2 ** self.frac, rounding)
        flag = None
        lim = self.kneg if sign else self.kmax
        if k > lim:
            k, flag = lim, "sat"
        elif k == 0 and x != 0:
            flag = "zeroed"
        if self.signmag:
            code = (sign << (self.bits - 1)) | k
        else:
            kk = -k if sign else k
            code = kk & (2 ** self.bits - 1)
            if sign and k == 0:
                code = 0                           # two's complement has no -0
        return code, flag

    def decode(self, code):
        if self.signmag:
            sign = code >> (self.bits - 1)
            k = code & (2 ** (self.bits - 1) - 1)
            v = Fraction(k, 2 ** self.frac)
            return (-v if sign else v), sign
        assert self.full_twos or code != 2 ** (self.bits - 1), "MXINT: forbidden code -2^(n-1) produced"
        k = code - 2 ** self.bits if code >= 2 ** (self.bits - 1) else code
        return Fraction(k, 2 ** self.frac), 0

    def all_values(self):
        return [Fraction(k, 2 ** self.frac) for k in range(-self.kmax, self.kmax + 1)]


# --------------------------------------------------------------------------
# format registry: element + block organization
# --------------------------------------------------------------------------

def make_fmt(elem, block=32, rule="floor", rounding="rne", exp_bits=8):
    """exp_bits: width of the shared exponent. 8 -> E8M0, clamp [-127, 127];
    k bits -> two's-complement range [-2^(k-1), 2^(k-1)-1] (BFP 4-bit: [-8, 7])."""
    lo, hi = (-127, 127) if exp_bits == 8 else (-2 ** (exp_bits - 1), 2 ** (exp_bits - 1) - 1)
    return dict(elem=elem, block=block, rule=rule, rounding=rounding,
                exp_bits=exp_bits, exp_lo=lo, exp_hi=hi)


def make_bfp(block=16, exp_bits=4, rule="floor", rounding="rne"):
    """BFP6 per D1 (plan v3, locked 21 Oct): g|e|m = 16|4|5. Groups of 16, 4-bit shared
    exponent clamped to [-8, 7], sign + 5-bit magnitude, k in [-31, 31],
    exponent floor(log2 absmax), RNE. Value = k * 2^-4 * 2^e."""
    return make_fmt(IntSym("BFP6", 6, signmag=True), block, rule, rounding, exp_bits)


FORMATS = {
    "MXFP_E3M2": make_fmt(MXFP("E3M2", 3, 2)),
    "MXFP_E2M3": make_fmt(MXFP("E2M3", 2, 3)),
    "MXFP_E4M1": make_fmt(MXFP("E4M1", 4, 1)),
    "MXFP_E4M1_reserved": make_fmt(MXFP("E4M1r", 4, 1, top_code_finite=False)),   # vmax 192
    "MXFP_E4M1_onenan": make_fmt(MXFP("E4M1n", 4, 1, one_nan=True)),              # vmax 256
    "MXINT6": make_fmt(IntSym("MXINT6", 6, signmag=False)),
    "MXINT6_twos": make_fmt(IntSym("MXINT6t", 6, signmag=False, full_twos=True)),   # k in [-32, 31]
    "BFP6": make_bfp(),                                                   # 16|4|5 (D1)
    "BFP6_32_8_ceil": make_bfp(block=32, exp_bits=8, rule="ceil"),        # old repo BFP6
    "BFP6_32_8_floor": make_bfp(block=32, exp_bits=8, rule="floor"),      # == MXINT6 grid
    "BFP6_row100": make_bfp(block=100, exp_bits=8, rule="ceil"),
}


# --------------------------------------------------------------------------
# shared scale
# --------------------------------------------------------------------------

def shared_exp(block_vals, elem, rule, rounding="rne", lo=-127, hi=127):
    """block_vals: list of Fractions (real elements only). Returns integer exponent,
    clamped to [lo, hi] (E8M0 by default)."""
    amax = max((abs(v) for v in block_vals), default=Fraction(0))
    if amax == 0:
        return 0
    if rule == "floor":
        s = floor_log2(amax) - elem.emax_elem
    elif rule == "ceil":
        # team definition (= formats.py): smallest power-of-two scale that clips nothing,
        # i.e. smallest s with amax <= vmax * 2^s
        s = ceil_log2(amax / elem.vmax)
    elif rule == "ceil_emax":
        # alternative reading of "ceil": ceil(log2 amax) - emax_elem (wastes range; kept for comparison)
        s = ceil_log2(amax) - elem.emax_elem
    elif rule == "fit":
        # smallest exponent such that the block max does not saturate after rounding
        s = floor_log2(amax) - elem.emax_elem - 1
        while True:
            _, flag = elem.encode(amax / pow2(s), rounding)
            if flag != "sat":
                break
            s += 1
    else:
        raise ValueError(rule)
    return max(lo, min(hi, s))


# 'opt': try floor-rule exponent + d, keep the smallest EXACT squared error;
# ties keep the offset tried first (0, -1, +1, -2), i.e. closest to the base rule.
OPT_OFFSETS = (0, -1, 1, -2)


# --------------------------------------------------------------------------
# matrix quantizer
# --------------------------------------------------------------------------

def quantize_block(raw, vals, elem, s, rnd):
    """Quantize one block at shared exponent s. Returns list of (code, flag, value Fraction, float)."""
    sc = pow2(s)
    res = []
    for t, v in enumerate(vals):
        code, flag = elem.encode(v / sc, rnd, neg=math.copysign(1.0, raw[t]) < 0)
        dv, sign = elem.decode(code)
        q = float(dv * sc)
        if sign and q == 0.0:
            q = -0.0
        res.append((code, flag, dv * sc, q))
    return res


def quantize_matrix(W, fmt_name_or_dict, return_codes=False):
    """Blocks along the last axis. Returns (W_hat float64, stats dict[, codes])."""
    fmt = FORMATS[fmt_name_or_dict] if isinstance(fmt_name_or_dict, str) else fmt_name_or_dict
    elem, B, rule, rnd = fmt["elem"], fmt["block"], fmt["rule"], fmt["rounding"]
    lo, hi = fmt.get("exp_lo", -127), fmt.get("exp_hi", 127)
    W = np.asarray(W, dtype=np.float64)
    R, K = W.shape
    out = np.zeros_like(W)
    stats = dict(n=0, sat=0, zeroed=0, sub=0)
    codes, scales = [], []
    for i in range(R):
        rc, rs = [], []
        for j0 in range(0, K, B):
            raw = W[i, j0:j0 + B]                     # real elements only (padding is implicit zeros)
            vals = [Fraction(float(v)) for v in raw]
            if rule == "opt":
                base = shared_exp(vals, elem, "floor", rnd, lo, hi)
                best = None
                for d in OPT_OFFSETS:
                    s_try = max(lo, min(hi, base + d))
                    res_try = quantize_block(raw, vals, elem, s_try, rnd)
                    err = sum((v - r[2]) ** 2 for v, r in zip(vals, res_try))
                    if best is None or err < best[0]:
                        best = (err, s_try, res_try)
                _, s, res = best
            else:
                s = shared_exp(vals, elem, rule, rnd, lo, hi)
                res = quantize_block(raw, vals, elem, s, rnd)
            rs.append(s + 127 if fmt.get("exp_bits", 8) == 8 else s & (2 ** fmt["exp_bits"] - 1))
            for t, (code, flag, _, q) in enumerate(res):
                out[i, j0 + t] = q
                rc.append(code)
                stats["n"] += 1
                if flag:
                    stats[flag] += 1
        codes.append(rc)
        scales.append(rs)
    if return_codes:
        return out, stats, (codes, scales)
    return out, stats


def rel_err(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def y_error(W, W_hat, X):
    return rel_err(W @ X, W_hat @ X)


def bits_per_element(fmt_name, K=100):
    """Storage per real element including the shared exponent.
      nominal  : b + e/B (long rows, no tail)
      d11      : team D11, tail block stored with only its real values + its own exponent
      padded   : tail block stored in full (zero padding kept in memory)"""
    fmt = FORMATS[fmt_name]
    B, b, eb = fmt["block"], fmt["elem"].bits, fmt.get("exp_bits", 8)
    nblk = math.ceil(K / B)
    return dict(nominal=b + eb / B, d11=(K * b + nblk * eb) / K, padded=(nblk * B * b + nblk * eb) / K)
