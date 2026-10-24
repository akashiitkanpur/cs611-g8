"""
Bit-exact cross-check: reference implementation (reference_impl.py) vs any other quantizer
(Akash's formats.py, Aryan's prototype, ...).

Usage
  python crosscheck.py --selftest                       # proves the harness catches a 1-ulp bug
  python crosscheck.py --impl formats:quantize          # module:function to test
  python crosscheck.py --impl formats:quantize --data W_C.npy --x Xcal_C.npy

The function under test must have the signature
    fn(W: np.ndarray, fmt: str, rule: str) -> W_hat: np.ndarray (same shape, dequantized)
with fmt in FORMAT_MAP keys and rule in {'floor','ceil'}. If Akash's API differs, write a
3-line adapter function and point --impl at it; do NOT modify reference_impl.py to match his code.
"""
import argparse, importlib, json, math, sys
import numpy as np
import reference_impl as R

# formats compared, with the rules each one is checked under
FORMAT_MAP = {
    "MXFP_E3M2": ["floor", "ceil", "opt"],
    "MXFP_E2M3": ["floor", "ceil", "opt"],
    "MXFP_E4M1": ["floor", "ceil", "opt"],
    "MXFP_E4M1_reserved": ["floor", "ceil", "opt"],
    "MXFP_E4M1_onenan": ["floor", "ceil", "opt"],
    "MXINT6": ["floor", "ceil", "opt"],
    "MXINT6_twos": ["floor", "ceil", "opt"],
    "BFP6_32_8_floor": ["floor", "ceil", "opt"],
    "BFP6": ["floor", "ceil", "opt"],            # D1: 16|4|5, 4-bit exponent [-8, 7], floor, RNE
    "BFP6_32_8_ceil": ["ceil"],   # old repo BFP6
    "BFP6_row100": ["ceil", "floor", "opt"],
}


# --------------------------------------------------------------------------
# adversarial inputs (100 columns, like the assignment's W)
# --------------------------------------------------------------------------

def grid_midpoints(elem):
    vals = elem.all_values()
    return [float((a + b) / 2) for a, b in zip(vals, vals[1:])]


def adversarial_matrix(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    K = 100

    def row(vals):
        v = np.zeros(K)
        v[:len(vals)] = vals[:K]
        rows.append(v)

    row([])                                                    # all-zero row (4 all-zero blocks)
    for p in (-3, 0, 2, 5):                                    # block max exactly 2^p and just below
        r = rng.uniform(-1, 1, K) * 2.0 ** p * 0.9
        r[0] = 2.0 ** p; r[32] = np.nextafter(2.0 ** p, 0); r[64] = -2.0 ** p; r[96] = -np.nextafter(2.0 ** p, 0)
        rows.append(r)
    for name in ("MXFP_E2M3", "MXFP_E3M2", "MXFP_E4M1", "MXINT6"):   # exact RNE ties, anchor gives s = 0
        el = R.FORMATS[name]["elem"]
        mids = [m for m in grid_midpoints(el) if m > 0]
        anchor = float(2.0 ** el.emax_elem)
        for start in range(0, len(mids), 31):
            chunk = mids[start:start + 31]
            blk = [anchor] + chunk + [-m for m in chunk]
            row(blk[:32] + [anchor] + blk[32:63] + [anchor] + [-x for x in chunk][:31])
    row([1e30, -1e30, 1e-30, -1e-30, 2.0 ** -140, 3.0, 0.0, -0.0])  # extreme magnitudes + clamp
    row([2.0 ** -133] * 32 + [-(2.0 ** -126)] * 32)            # scale near E8M0 lower clamp
    r = np.full(K, 0.9); r[::32] = 32.0; rows.append(r)        # one outlier per block
    r = rng.standard_normal(K); r[rng.random(K) < 0.6] = 0.0; rows.append(r)   # sparse like E
    r = np.where(rng.random(K) < 0.5, -0.0, 0.0); r[5] = 1.0; rows.append(r)   # signed zeros
    rows.append(np.linspace(-7.9, 7.9, K))                     # sweeps through every binade
    # BFP 16|4|5: groups (16 wide) whose exponent hits both ends of the 4-bit field [-8, 7]
    r = np.zeros(K)
    r[0:16] = rng.uniform(-1, 1, 16) * 2.0 ** -10; r[0] = 2.0 ** -10          # e = -10 -> clamped to -8
    r[16:32] = rng.uniform(-1, 1, 16) * 2.0 ** -12                            # tiny group, mostly zeroed
    r[32:48] = rng.uniform(-1, 1, 16) * 300; r[32] = 1000.0                   # e = 9 -> clamped to 7, saturates at 248
    r[48:64] = rng.uniform(-1, 1, 16) * 200; r[48] = np.nextafter(256.0, 0)  # e = 7 exactly at the top
    r[64:80] = rng.uniform(-1, 1, 16) * 2.0 ** -8; r[64] = 2.0 ** -8          # e = -8 exactly at the bottom
    rows.append(r)
    # negative block maxima: two's complement may use code -32, symmetric may not
    r = rng.uniform(-1, 1, K)
    r[0] = -1.99; r[16] = -1.97; r[32] = -1.96875; r[48] = 1.99; r[64] = -3.95; r[96] = -1.999
    rows.append(r)
    return np.array(rows)


def random_matrices(seed=1):
    rng = np.random.default_rng(seed)
    yield "gauss", rng.standard_normal((50, 100))
    yield "laplace", rng.laplace(size=(50, 100))
    yield "student_t2", rng.standard_t(2, size=(50, 100))
    yield "uniform", rng.uniform(-1.7, 1.7, (50, 100))
    yield "wide_scale", rng.standard_normal((50, 100)) * np.exp(rng.uniform(-8, 8, (50, 1)))


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def ref_quantize(W, fmt, rule):
    f = dict(R.FORMATS[fmt]); f["rule"] = rule
    return R.quantize_matrix(W, f)[0]


def compare(W, fn, fmt, rule, max_show=5):
    ref = ref_quantize(W, fmt, rule)
    try:
        got = np.asarray(fn(W, fmt, rule), dtype=np.float64)
    except KeyError as e:                          # format not implemented on the other side yet
        return dict(fmt=fmt, rule=rule, n=int(W.size), mismatches=None, missing=str(e))
    assert got.shape == ref.shape, f"shape {got.shape} vs {ref.shape}"
    bad = ~((ref == got) | (np.isnan(ref) & np.isnan(got)))
    sign0 = (ref == 0) & (got == 0) & (np.signbit(ref) != np.signbit(got))
    idx = np.argwhere(bad)
    return dict(
        fmt=fmt, rule=rule, n=int(W.size), mismatches=int(bad.sum()),
        signed_zero_diffs=int(sign0.sum()),
        max_abs_diff=float(np.max(np.abs(ref - got))) if bad.any() else 0.0,
        examples=[dict(row=int(i), col=int(j), w=float(W[i, j]), ref=float(ref[i, j]), got=float(got[i, j]))
                  for i, j in idx[:max_show]],
    )


def run(fn, data=None, x=None):
    cases = [("adversarial", adversarial_matrix())] + list(random_matrices())
    if data is not None:
        cases.append(("dataset", np.load(data)))
    report, ok = [], True
    for cname, W in cases:
        for fmt, rules in FORMAT_MAP.items():
            for rule in rules:
                r = compare(W, fn, fmt, rule); r["case"] = cname
                if r["mismatches"] is None:
                    print(f"MISS {cname:12s} {fmt:18s} {rule:5s} not in the implementation under test")
                    report.append(r)
                    continue
                if cname == "dataset" and x is not None:
                    X = np.load(x)
                    r["y_err_ref"] = R.y_error(W, ref_quantize(W, fmt, rule), X)
                    r["y_err_impl"] = R.y_error(W, np.asarray(fn(W, fmt, rule)), X)
                ok &= r["mismatches"] == 0
                report.append(r)
                status = "OK  " if r["mismatches"] == 0 else "FAIL"
                extra = f"  Y-err ref {r['y_err_ref']*100:.3f}% impl {r['y_err_impl']*100:.3f}%" if "y_err_ref" in r else ""
                print(f"{status} {cname:12s} {fmt:18s} {rule:5s} mismatches {r['mismatches']:5d}/{r['n']}"
                      f"  (-0 diffs {r['signed_zero_diffs']}){extra}")
                for e in r["examples"]:
                    print("       ", e)
    import os
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "crosscheck_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nBIT-EXACT AGREEMENT" if ok else "\nDISAGREEMENT FOUND: see crosscheck_report.json")
    return ok


def selftest():
    """The harness must (a) pass the reference against itself and (b) catch a 1-ulp perturbation."""
    W = adversarial_matrix()
    assert compare(W, ref_quantize, "MXFP_E2M3", "floor")["mismatches"] == 0

    def buggy(W, fmt, rule):                       # typical vectorized bug: log2 instead of exact exponent
        out = ref_quantize(W, fmt, rule).copy()
        out[1, 0] = np.nextafter(out[1, 0], np.inf)
        return out
    assert compare(W, buggy, "MXFP_E2M3", "floor")["mismatches"] == 1
    print(f"selftest OK: adversarial matrix {W.shape}, harness detects a single-ulp difference")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl"); ap.add_argument("--data"); ap.add_argument("--x")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.impl:
        selftest(); sys.exit(0)
    mod, fn = a.impl.split(":")
    sys.exit(0 if run(getattr(importlib.import_module(mod), fn), a.data, a.x) else 1)
