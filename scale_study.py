"""Scale-optimization study (Anuj): how much does the choice of shared exponent matter?

  python scale_study.py                  # datasets A-E if the course file is found, else C from data/
  python scale_study.py --npz assignment_data.npz    # the course file: W_A ... W_E and X_cal
  python scale_study.py --data DIR       # every W_<tag>.npy in DIR (with X_<tag>.npy or X_cal.npy)
  python scale_study.py --no-figure      # skip matplotlib

The course file is looked up like run_all.py does: --npz / --data, then $CS611_DATA, then
data/assignment_data.npz; without it only dataset C (data/W_C.npy) is studied.

Writes results/scale_study.md (every table; generated, do not edit by hand),
results/scale_study_<tag>.csv and figures/fig5_scale_tradeoff.png.
Seeds are fixed; the numbers in notes/anuj.md come from these files.
"""
import argparse
import csv
import glob
import os

import numpy as np

import formats as F
import scale_opt as S

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = list(S.FMTS)
EXTRA = list(S.VARIANTS)
SHORT = {"MXFP6_E3M2": "E3M2", "MXFP6_E2M3": "E2M3", "MXFP6_E4M1": "E4M1", "MXINT6": "MXINT6",
         "BFP6": "BFP6 16|4|5", "MXFP6_E4M1_v192": "E4M1 v192", "MXFP6_E4M1_nan1": "E4M1 one-NaN",
         "BFP6_g32e8": "BFP6 32|8|5 (= MXINT6_sym)", "BFP6_b32_ceil": "BFP6 32 ceil (v1)",
         "MXINT6_sym": "MXINT6_sym [-31, 31]"}
POLICIES = [  # name, how, roundings per weight, what it is
    ("ocp", "class floor rule", 1),
    ("ceil", "smallest scale that clips nothing", 1),
    ("opt", "search -2..+1, weight SSE (repo window)", 4),
    ("opt_wide", "search -3..+2, weight SSE", 6),
    ("oracle_w", "search -8..+8, weight SSE (per-block oracle)", 17),
    ("model", "search -2..+1, predicted SSE, no rounding", 0),
    ("opt_out", "search -2..+1, r^T H_bb r, H from fit columns", 4),
    ("oracle_y", "joint per row -3..+2, r^T H r with full H, fit columns", 6),
    ("bound_y", "as oracle_y but fitted ON the held-out columns: a bound, not a usable policy", 6),
]


# --------------------------------------------------------------------------- data
def load_datasets(data_dir):
    """W_<tag> matrices with their X: from a folder of .npy files or one .npz (the course file)."""
    out = {}
    if os.path.isfile(data_dir) and data_dir.endswith(".npz"):
        d = np.load(data_dir)
        for key in sorted(k for k in d.files if k.startswith("W_")):
            tag = key[2:]
            xk = next((x for x in (f"X_{tag}", f"X_cal_{tag}", "X_cal") if x in d.files), None)
            if xk:
                out[tag] = (d[key].astype(np.float64), d[xk].astype(np.float64))
        if not out:
            raise SystemExit(f"no W_<tag> array with an X_cal found in {data_dir}")
        return out
    for path in sorted(glob.glob(os.path.join(data_dir, "W_*.npy"))):
        tag = os.path.basename(path)[2:-4]
        for xn in (f"X_{tag}.npy", f"X_cal_{tag}.npy", "X_cal.npy"):
            xp = os.path.join(data_dir, xn)
            if os.path.exists(xp):
                out[tag] = (np.load(path).astype(np.float64), np.load(xp).astype(np.float64))
                break
    if not out:
        raise SystemExit(f"no W_<tag>.npy with an X_cal.npy found in {data_dir}")
    return out


def run_policy(W, name, policy, H_fit, H_eval):
    if policy in ("ocp", "ceil"):
        return S.quantize(W, name, policy)
    if policy == "opt":
        return S.quantize(W, name, "opt", S.OPT_WINDOW)
    if policy == "opt_wide":
        return S.quantize(W, name, "opt", S.WIDE_WINDOW)
    if policy == "oracle_w":
        return S.quantize(W, name, "opt", S.ORACLE_WINDOW)
    if policy == "model":
        return S.quantize(W, name, "model", S.OPT_WINDOW)
    if policy == "opt_out":
        return S.quantize(W, name, "opt", S.OPT_WINDOW, H=H_fit)
    if policy in ("oracle_y", "bound_y"):
        Wq, e, _ = S.joint_search(W, name, H_fit if policy == "oracle_y" else H_eval, S.WIDE_WINDOW)
        return Wq, e
    raise ValueError(policy)


def block_max_signed(W, block):
    """The value with the largest magnitude in each block, with its sign: shape (rows, blocks)."""
    rows = np.arange(W.shape[0])
    return np.stack([W[rows, sl.start + np.abs(W[:, sl]).argmax(1)] for sl in S.block_slices(W.shape[1], block)], 1)


def col_energy(W, Wq, X):
    """Per-column squared error of Y and per-column energy of Y (for the bootstrap)."""
    return (((W - Wq) @ X) ** 2).sum(0), ((W @ X) ** 2).sum(0)


def boot_diff(ea, eb, y, n=2000, seed=0):
    """Paired bootstrap over held-out columns of rel_err(a) - rel_err(b), in % points."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = 100 * (np.sqrt(ea[idx].sum(1) / y[idx].sum(1)) - np.sqrt(eb[idx].sum(1) / y[idx].sum(1)))
    point = 100 * (np.sqrt(ea.sum() / y.sum()) - np.sqrt(eb.sum() / y.sum()))
    return point, np.percentile(d, 2.5), np.percentile(d, 97.5)


# --------------------------------------------------------------------------- 1. cross-check
def crosscheck(datasets, md):
    """Bit-exact comparison with formats.py on the datasets, random and adversarial matrices."""
    rng = np.random.default_rng(0)
    mats = {f"W_{t}": W for t, (W, _) in datasets.items()}
    mats["gauss"] = rng.standard_normal((64, 100))
    mats["laplace"] = rng.laplace(size=(64, 100))
    mats["student_t3"] = rng.standard_t(3, (64, 100))
    mats["sparse70"] = rng.standard_normal((64, 100)) * (rng.random((64, 100)) > 0.7)
    mats["rows_1e-6_1e6"] = rng.standard_normal((64, 100)) * 10.0 ** rng.uniform(-6, 6, (64, 1))
    adv = rng.standard_normal((64, 100))                       # block maxima at / just below 2^k (FP32)
    for i in range(64):
        for s in range(0, 100, 16):
            k = int(rng.integers(-6, 7))
            v = 2.0 ** k if i % 2 else float(np.nextafter(np.float32(2.0 ** k), np.float32(0)))
            adv[i, s] = v * (1 if i % 4 < 2 else -1)
            adv[i, s + 1:s + 16] *= 2.0 ** k / 8
    mats["pow2_maxima_fp32"] = adv
    for k in list(mats):
        if not k.startswith("W_"):
            mats[k] = mats[k].astype(np.float32).astype(np.float64)   # the course data is FP32
    Xr = rng.standard_normal((100, 3)) @ rng.standard_normal((3, 80)) + 0.5 * rng.standard_normal((100, 80))
    H_rand = Xr @ Xr.T
    H_of = {f"W_{t}": X @ X.T for t, (_, X) in datasets.items()}

    # (mine, formats.py name, block override, exponents compared up to a constant offset).
    # Formats are matched by definition, so the check works with formats.py v1, v2 and after
    # the revised D10 (MXINT6 two's complement, symmetric code = MXINT6_sym) lands there.
    v2 = getattr(F, "FORMATS_VERSION", "v1") != "v1"
    repo_names = set(F.FORMATS) | set(getattr(F, "VARIANTS", {}))
    repo_tc = F.spec("MXINT6")[1].get("kmin") == -32             # formats.py MXINT6 = two's complement?
    repo_of = {n: n for n in CANDIDATES + EXTRA if n in repo_names}
    repo_of["MXINT6"] = "MXINT6" if repo_tc else ("MXINT6_tc" if "MXINT6_tc" in repo_names else None)
    repo_of["MXINT6_sym"] = "MXINT6_sym" if "MXINT6_sym" in repo_names else (None if repo_tc else "MXINT6")
    if not v2:
        repo_of["BFP6_b32_ceil"], repo_of["BFP6"] = "BFP6", None    # v1 called the old BFP6 "BFP6"
    pairs = [(n, repo_of[n], None, n.startswith("BFP")) for n in CANDIDATES + EXTRA if repo_of.get(n)]
    if repo_of.get("MXINT6_sym"):                       # D1 BFP6 = symmetric MXINT6 grid in blocks of 16
        pairs.append(("BFP6", repo_of["MXINT6_sym"], 16, False))
    checks = [("ocp", dict(policy="ocp"), dict(policy="ocp")),
              ("ceil", dict(policy="ceil"), dict(policy="ceil")),
              ("opt", dict(policy="opt"), dict(policy="opt")),
              ("opt + H", dict(policy="opt", H=True), dict(policy="opt", H=True)),
              ("search -3..+3", dict(policy="opt", window=tuple(range(-3, 4))),
               dict(policy="ocp", shift_search=3))]
    md.append("## 1. Cross-check against formats.py (bit-exact)\n")
    md.append("Matrices: " + ", ".join(f"`{k}` {v.shape[0]}x{v.shape[1]}" for k, v in mats.items())
              + ". Random matrices are cast to FP32 first, like the course data. `opt + H` uses X_cal X_cal^T for "
              "the datasets and a rank-3 correlated H for the random matrices.\n")
    md.append("| mine | formats.py | policy | matrices | identical Wq | identical exponents | elements that differ |")
    md.append("|---|---|---|---|---|---|---|")
    n_bad = 0
    for mine, repo, blk, shifted in pairs:
        f = S.spec(mine)
        for label, km, kr in checks:
            same_q = same_e = n_mat = n_diff = 0
            for mk, M in mats.items():
                H = (H_of.get(mk, H_rand)) if km.get("H") else None
                amax = np.stack([np.abs(M[:, sl]).max(1) for sl in S.block_slices(M.shape[1], f.block)], 1)
                live = amax > 0
                if blk is not None and f.e_range != S.E8M0:   # formats.py MXINT6 has no 4-bit clamp
                    e0 = S.floor_log2(amax[live]) - f.emax
                    if e0.min() - 3 < f.e_range[0] or e0.max() + 3 > f.e_range[1]:
                        continue
                a, ea = S.quantize(M, mine, km["policy"], km.get("window", S.OPT_WINDOW), H=H)
                b, eb = F.quantize(M, repo, block=blk, policy=kr["policy"],
                                   shift_search=kr.get("shift_search", 0), H=H)
                n_mat += 1
                same_q += np.array_equal(a, b)
                if ea.shape == eb.shape:
                    d = (ea - eb)[live]
                    same_e += bool(d.size == 0 or (np.all(d == d[0]) if shifted else np.all(d == 0)))
                n_diff += int((a != b).sum())
            if not (S.spec(mine).grid_neg is not None and label == "ceil"):   # explained below the table
                n_bad += n_diff
            md.append(f"| {mine} | {repo}{'' if blk is None else f' block={blk}'} | {label} | {n_mat} | "
                      f"{same_q}/{n_mat} | {same_e}/{n_mat} | {n_diff} |")
    md.append("")
    missing = [n for n in CANDIDATES + EXTRA if not repo_of.get(n)]
    md.append(f"formats.py version: {getattr(F, 'FORMATS_VERSION', 'v1')}; its MXINT6 is "
              f"{'two' + chr(39) + 's complement' if repo_tc else 'symmetric [-31, 31]'}. Formats are matched by "
              "definition (mine `MXINT6` = two's complement, D10 of 23 Oct)"
              + (f"; not in this formats.py: {', '.join(missing)}" if missing else "") + ". Exponents are compared "
              "on non-zero blocks (an all-zero block gets exponent 0 in both); for the BFP rows only up to a "
              "constant offset (formats.py v1 stored the exponent of the integer code, 4 lower than mine; v2 stores "
              "the same E). The `BFP6 | ... block=16` rows compare D1 BFP6 with the symmetric MXINT6 grid in blocks "
              "of 16 and skip matrices where the 4-bit exponent clamp [-8, 7] could bind (every block exponent must "
              "be at least 3 from the limits). Signed zeros are not counted: an integer code has one zero, so "
              "`scale_opt.py` decodes MXINT6 to +0 where formats.py returns -0.\n")
    tc_ceil = [r for r in md if r.startswith("| MXINT6 | ") and " | ceil | " in r]
    if tc_ceil and not tc_ceil[0].endswith("| 0 |"):
        md.append("**Two's-complement MXINT6 under `ceil` differs on purpose (open decision).** formats.py's `ceil` "
                  "(and Siddhant's reference) moves the scale up whenever amax > (31/16) 2^e, whatever the sign of "
                  "the maximum. For two's complement a negative maximum down to -2 * 2^e is representable (code -32), "
                  "so nothing clips and `scale_opt.py` keeps the lower scale, which is the literal 'smallest scale "
                  "that clips nothing'. The class-rule and `opt` rows agree. This row is left out of the 'differing "
                  "elements' total the script prints.\n")

    # the counters (Lecture 19 definitions), only in formats.py v2
    if v2:
        Wc = mats["W_C"] if "W_C" in mats else next(iter(mats.values()))
        agree, total = 0, 0
        bad = []
        for mine, repo, blk, _ in pairs:
            if blk is not None:
                continue
            for pol in ("ocp", "ceil", "opt"):
                a, ea = S.quantize(Wc, mine, pol)
                st = F.quantize(Wc, repo, policy=pol, return_stats=True)[2]
                p = S.error_parts(Wc, a, ea, mine)
                mine_c = (p["n_range"], p["n_saturated"], p["n_zero"])
                repo_c = (st["n_range"], st["n_saturated"], st["n_zeroed"])
                total += 1
                agree += mine_c == repo_c
                if mine_c != repo_c:
                    bad.append(f"{mine} {pol}: mine {mine_c} vs formats.py {repo_c}")
        md.append(f"Counters on `{'W_C' if 'W_C' in mats else 'the first matrix'}` (n_range = out of range before "
                  "rounding, n_saturated = rounded value beyond vmax, n_zeroed), compared with formats.py v2's "
                  f"stats for every pair above under ocp, ceil and opt: {agree}/{total} identical"
                  + (". Differences: " + "; ".join(bad) if bad else "") + ".\n")
    reference_check(mats.get("W_C"), md)
    log2_edge_case(md)
    return n_bad


REF_NAME = {"MXFP6_E3M2": "MXFP_E3M2", "MXFP6_E2M3": "MXFP_E2M3", "MXFP6_E4M1": "MXFP_E4M1",
            "MXINT6": "MXINT6_twos", "BFP6": "BFP6", "MXFP6_E4M1_v192": "MXFP_E4M1_reserved",
            "MXFP6_E4M1_nan1": "MXFP_E4M1_onenan", "MXINT6_sym": "MXINT6",
            "BFP6_g32e8": "BFP6_32_8_floor", "BFP6_b32_ceil": "BFP6_32_8_ceil"}


def reference_check(W, md):
    """Third implementation: Siddhant's scalar reference (verification/reference_impl.py), on W_C."""
    path = os.path.join(HERE, "verification", "reference_impl.py")
    if W is None or not os.path.exists(path):
        return
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("reference_impl", path)
    R = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(R)
    rule = {"ocp": "floor", "ceil": "ceil", "opt": "opt"}
    md.append("Third implementation, Siddhant's scalar reference (`verification/reference_impl.py`, exact "
              "fractions, written without reading formats.py), on W_C. Differing elements per policy, and signed "
              "zeros (-0 vs +0) in brackets:\n")
    md.append("| mine | reference | ocp | ceil | opt |")
    md.append("|---|---|---|---|---|")
    tc_ceil_diff = 0
    for mine, ref in REF_NAME.items():
        if ref not in R.FORMATS:
            continue
        cells = []
        for pol in ("ocp", "ceil", "opt"):
            f = dict(R.FORMATS[ref])
            f["rule"] = rule[pol]
            b = R.quantize_matrix(W, f)[0]
            a = S.quantize(W, mine, pol)[0]
            n = int((a != b).sum())
            sz = int(((a == 0) & (b == 0) & (np.signbit(a) != np.signbit(b))).sum())
            cells.append(f"{n} ({sz})")
            if S.spec(mine).grid_neg is not None and pol == "ceil":
                tc_ceil_diff += n
        md.append(f"| {mine} | {ref} | " + " | ".join(cells) + " |")
    md.append("")
    if tc_ceil_diff:
        md.append("The two's-complement MXINT6 `ceil` difference is a definition, not a rounding error: the reference "
                  "(like formats.py) moves the scale up when amax > (31/16) 2^e whatever the sign of the maximum; "
                  "`scale_opt.py` keeps the lower scale when a negative maximum fits the code -32, which clips "
                  "nothing. Open decision for the team.\n")


def log2_edge_case(md):
    """The np.log2 edge case found on 21 Oct: float64 values a few ulps below 2^k."""
    x = float(np.nextafter(8.0, 0))
    w = np.zeros((1, 32)); w[0, 0] = x; w[0, 1] = 1.0
    eF = F.quantize(w, "MXFP6_E2M3")[1][0, 0]
    eS = S.quantize(w, "MXFP6_E2M3")[1][0, 0]
    ks = [k for k in range(-126, 127) if np.floor(np.log2(np.nextafter(2.0 ** k, 0))) != k - 1]
    f32 = [k for k in range(-126, 127)
           if np.floor(np.log2(np.float64(np.nextafter(np.float32(2.0 ** k), np.float32(0))))) != k - 1]
    pred_src = open(os.path.join(HERE, "predictor.py"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(HERE, "predictor.py")) else ""
    pred_note = (" predictor.py still uses `np.floor(np.log2(...))` (harmless on FP32 data)."
                 if "np.floor(np.log2(" in pred_src else "")
    md.append("**Edge case (float64 input only).** `np.floor(np.log2(a))` rounds up when `a` is a float64 a few ulps "
              f"below a power of two; this happens for {len(ks)} of 253 powers of two in float64 and for {len(f32)} "
              "when the value is the FP32 number just below 2^k, so it cannot change any result on the (FP32) course "
              "data. `np.frexp(a)[1] - 1` is exact. ")
    if eF == eS:
        md[-1] += (f"formats.py now handles it: for a = nextafter(8, 0) = {x!r} it gives the E2M3 block exponent "
                   f"{eF:.0f}, as the class rule floor(log2 a) - 2 does." + pred_note + "\n")
    else:
        md[-1] += (f"For a = nextafter(8, 0) = {x!r}, formats.py gives the E2M3 block exponent {eF:.0f} where the "
                   f"class rule floor(log2 a) - 2 gives {eS}." + pred_note + "\n")


# --------------------------------------------------------------------------- 2. per-dataset study
def study(tag, W, X, md, csv_rows):
    X_fit, X_eval = F.split_X(X)
    H_fit, H_eval = X_fit @ X_fit.T, X_eval @ X_eval.T
    res = {}
    for name in CANDIDATES + EXTRA:
        for pol, _, _ in POLICIES:
            if name in EXTRA and pol not in ("ocp", "ceil", "opt"):
                continue
            Wq, e = run_policy(W, name, pol, H_fit, H_eval)
            parts = S.error_parts(W, Wq, e, name)
            res[name, pol] = dict(Wq=Wq, e=e, y_ho=S.rel_err(W, Wq, X_eval), y_fit=S.rel_err(W, Wq, X_fit),
                                  y_all=S.rel_err(W, Wq, X), w=S.rel_err(W, Wq), **parts)

    w = W.ravel()
    kurt = ((w - w.mean()) ** 4).mean() / w.var() ** 2
    md.append(f"## 2. Dataset {tag}\n")
    md.append(f"W {W.shape[0]}x{W.shape[1]}, X_cal {X.shape[0]}x{X.shape[1]} (fit on the {X_fit.shape[1]} even "
              f"columns, held out: the {X_eval.shape[1]} odd columns). std {w.std():.3f}, kurtosis {kurt:.2f}, "
              f"zeros {np.mean(w == 0):.1%}, max |w| {np.abs(w).max():.3f}.\n")

    # 2a. main table
    md.append(f"### 2a. Relative error of Y = W X on the held-out columns (%), dataset {tag}\n")
    md.append("Lower is better; **bold** = best format in each column. Gain = (ocp - policy) / ocp.\n")
    md.append("| format | bits/elem | " + " | ".join(p for p, _, _ in POLICIES) + " |")
    md.append("|---|---|" + "---|" * len(POLICIES))
    best = {p: min(CANDIDATES, key=lambda n: res[n, p]["y_ho"]) for p, _, _ in POLICIES}
    for n in CANDIDATES:
        cells = []
        for p, _, _ in POLICIES:
            v = 100 * res[n, p]["y_ho"]
            cells.append(f"**{v:.3f}**" if best[p] == n else f"{v:.3f}")
        md.append(f"| {SHORT[n]} | {S.bits_per_elem(n, W.shape[1]):.2f} | " + " | ".join(cells) + " |")
    md.append("| *gain vs ocp, best format* | | " + " | ".join(
        f"{100 * (1 - res[best[p], p]['y_ho'] / res[best[p], 'ocp']['y_ho']):+.1f}%" for p, _, _ in POLICIES) + " |")
    md.append("")
    md.append("| policy | what it does | element roundings per weight |")
    md.append("|---|---|---|")
    for p, how, cost in POLICIES:
        md.append(f"| {p} | {how} | {cost} |")
    md.append("")
    md.append(f"Same table on all 256 columns of X_cal / on W (relative Frobenius error of W), %:\n")
    md.append("| format | " + " | ".join(f"{p} (Y all / W)" for p, _, _ in POLICIES[:5]) + " |")
    md.append("|---|" + "---|" * 5)
    for n in CANDIDATES:
        md.append(f"| {SHORT[n]} | " + " | ".join(
            f"{100 * res[n, p]['y_all']:.3f} / {100 * res[n, p]['w']:.3f}" for p, _, _ in POLICIES[:5]) + " |")
    md.append("")
    md.append("Does fitting the exponents to X_cal overfit? Y error (%) on the even columns / on the odd "
              "(held-out) columns. opt uses no X; opt_out and oracle_y are fitted on the even columns; bound_y is "
              "fitted on the odd columns, so its odd-column number is the best any exponent choice can do there.\n")
    over = ("opt", "opt_out", "oracle_y", "bound_y")
    md.append("| format | " + " | ".join(f"{p} (even / odd)" for p in over) + " |")
    md.append("|---|" + "---|" * len(over))
    for n in CANDIDATES:
        md.append(f"| {SHORT[n]} | " + " | ".join(
            f"{100 * res[n, p]['y_fit']:.3f} / {100 * res[n, p]['y_ho']:.3f}" for p in over) + " |")
    md.append("")
    md.append("Variants (sensitivity rows, not candidates), held-out Y error %:\n")
    md.append("| variant | bits/elem | ocp | ceil | opt |")
    md.append("|---|---|---|---|---|")
    for n in EXTRA:
        md.append(f"| {SHORT[n]} | {S.bits_per_elem(n, W.shape[1]):.2f} | " + " | ".join(
            f"{100 * res[n, p]['y_ho']:.3f}" for p in ("ocp", "ceil", "opt")) + " |")
    md.append("")
    if "MXINT6_sym" in EXTRA:
        # two's complement only matters when a NEGATIVE value reaches the extra code -32
        sign = {}
        mx = block_max_signed(W, 32)
        for n in ("MXINT6", "MXINT6_sym"):
            d = res[n, "opt"]["e"] - res[n, "ocp"]["e"]
            sign[n] = (int((d == -1).sum()), int(((d == -1) & (mx < 0)).sum()))
        md.append("MXINT6 two's complement (the candidate since D10 was revised on 23 Oct) vs MXINT6_sym [-31, 31]: "
                  "under the class rule the extra code -32 is almost never reached (held-out "
                  f"{100 * res['MXINT6', 'ocp']['y_ho']:.3f} vs {100 * res['MXINT6_sym', 'ocp']['y_ho']:.3f}). "
                  f"Under `opt` it matters ({100 * res['MXINT6', 'opt']['y_ho']:.3f} vs "
                  f"{100 * res['MXINT6_sym', 'opt']['y_ho']:.3f}): at d = -1 the clip level for a negative maximum "
                  "is -2^E instead of -(31/32) 2^E, so shrinking the scale costs less. Blocks with d = -1 under opt: "
                  f"two's complement {sign['MXINT6'][0]} ({sign['MXINT6'][1]} with a negative maximum), symmetric "
                  f"{sign['MXINT6_sym'][0]} ({sign['MXINT6_sym'][1]} with a negative maximum).\n")

    # 2b. ranking + bootstrap
    yev = col_energy(W, W, X_eval)[1]
    md.append(f"### 2b. Does the ranking change with the scale policy? (dataset {tag})\n")
    md.append("Gap = runner-up minus winner, held-out Y error in % points, with a paired bootstrap 95% interval "
              "over the held-out columns (2000 resamples, seed 0). Tie rule (D6): if the interval contains 0 the "
              "top two are a tie; 'identical' = both formats produce exactly the same quantized W.\n")
    md.append("| policy | ranking (best first) | gap winner vs runner-up [95% CI] | verdict |")
    md.append("|---|---|---|---|")
    for p, _, _ in POLICIES:
        order = sorted(CANDIDATES, key=lambda n: res[n, p]["y_ho"])
        ea = col_energy(W, res[order[1], p]["Wq"], X_eval)[0]
        eb = col_energy(W, res[order[0], p]["Wq"], X_eval)[0]
        g, lo, hi = boot_diff(ea, eb, yev)
        if np.array_equal(res[order[0], p]["Wq"], res[order[1], p]["Wq"]):
            verdict = f"tie, identical ({SHORT[order[0]]} = {SHORT[order[1]]})"
        elif lo <= 0:
            verdict = f"tie ({SHORT[order[0]]} / {SHORT[order[1]]})"
        else:
            verdict = SHORT[order[0]]
        res["_verdict", p] = verdict
        md.append(f"| {p} | {' < '.join(SHORT[n] for n in order)} | {g:.3f} [{lo:.3f}, {hi:.3f}] | {verdict} |")
    md.append("")
    md.append("Gain of each search over the class rule, per format: ocp minus policy, held-out Y error in "
              "% points [95% CI]. Positive = the search helps.\n")
    md.append("| format | ceil | opt | opt_wide | model | opt_out | oracle_y | bound_y |")
    md.append("|---|---|---|---|---|---|---|---|")
    for n in CANDIDATES:
        e0 = col_energy(W, res[n, "ocp"]["Wq"], X_eval)[0]
        cells = []
        for p in ("ceil", "opt", "opt_wide", "model", "opt_out", "oracle_y", "bound_y"):
            g, lo, hi = boot_diff(e0, col_energy(W, res[n, p]["Wq"], X_eval)[0], yev)
            cells.append(f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        md.append(f"| {SHORT[n]} | " + " | ".join(cells) + " |")
    md.append("")
    for split, key in (("held-out", "y_ho"), ("all X_cal", "y_all")):
        md.append(f"Relative gain of opt over ocp ({split}): " + ", ".join(
            f"{SHORT[n]} {100 * (1 - res[n, 'opt'][key] / res[n, 'ocp'][key]):.1f}%" for n in CANDIDATES) + ".  ")
    md.append("")

    # 2c. clipping vs zeroing vs rounding
    md.append(f"### 2c. Where the weight error comes from (dataset {tag})\n")
    md.append("Share of the squared weight error from out-of-range (clipped) elements, from non-zero weights "
              "flushed to 0, and from ordinary rounding. Counts out of "
              f"{W.size} weights, with the Lecture 19 definitions used by formats.py v2: 'range' = |w|/scale "
              "beyond vmax before rounding; 'saturated' = the rounded value is beyond vmax, so saturation "
              "changed it (a value just above vmax that rounds back to vmax is in the first count only).\n")
    md.append("| format | policy | clip | zero | round | n range | n saturated | n zeroed | W rel err % |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for n in CANDIDATES:
        for p in ("ocp", "ceil", "opt"):
            r = res[n, p]
            tot = r["sse_clip"] + r["sse_zero"] + r["sse_round"]
            md.append(f"| {SHORT[n]} | {p} | {r['sse_clip'] / tot:.1%} | {r['sse_zero'] / tot:.1%} | "
                      f"{r['sse_round'] / tot:.1%} | {r['n_range']} | {r['n_saturated']} | {r['n_zero']} | "
                      f"{100 * r['w']:.3f} |")
    md.append("")

    # 2d. clip-zone blocks: bump or not
    md.append(f"### 2d. Blocks the class rule clips: is one exponent up worth it? (dataset {tag})\n")
    md.append("A block is clipped by the floor rule when its max mantissa m = amax / 2^floor(log2 amax) exceeds "
              "r = vmax / 2^emax for the sign of the maximum. 'Predicted' = log2(2/r), the clip-zone share if "
              "log2(amax) is spread evenly over a binade; for two's-complement MXINT6, r = 2 for a negative maximum "
              "(code -32), so only blocks with a positive maximum can clip and the prediction is weighted by the "
              "share of those. For the clipped blocks, compare the weight SSE at e_ocp (clip part + rest) with "
              "e_ocp + 1.\n")
    md.append("| format | r | blocks clipped | share | predicted | clipped weights: clip SSE at e_ocp | same weights: "
              "rounding SSE at e_ocp+1 | other weights: SSE change | other weights whose step doubles | bump helps in "
              "| net SSE change from bumping |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in CANDIDATES:
        f = S.spec(n)
        r_pos = f.grid[-1] / 2.0 ** f.emax
        r_neg = (f.grid if f.grid_neg is None else f.grid_neg)[-1] / 2.0 ** f.emax
        clip_sse = clip_after = others = net = 0.0
        n_clip = n_help = n_blk = n_dbl = n_el = n_neg = 0
        for sl in S.block_slices(W.shape[1], f.block):
            Wb = W[:, sl]
            amax = np.abs(Wb).max(1)
            live = amax > 0
            n_blk += int(live.sum())
            n_neg += int((live & (block_max_signed(Wb, f.block)[:, 0] < 0)).sum())
            e0 = np.clip(S.base_exp(f, amax, "ocp"), *f.e_range)
            out = S.out_of_range(f, Wb / np.ldexp(1.0, e0)[:, None])
            clipped = out.any(1) & live
            if not clipped.any():
                continue
            Wb, e0, sat = Wb[clipped], e0[clipped], out[clipped]
            s0 = np.ldexp(1.0, e0)[:, None]
            r0 = (Wb - S.qblock(f, Wb, e0)) ** 2
            r1 = (Wb - S.qblock(f, Wb, np.clip(e0 + 1, *f.e_range))) ** 2
            clip_sse += r0[sat].sum()
            clip_after += r1[sat].sum()
            others += r1[~sat].sum() - r0[~sat].sum()
            net += r1.sum() - r0.sum()
            n_clip += int(clipped.sum())
            n_help += int((r1.sum(1) < r0.sum(1)).sum())
            step0 = S.local_step(f, np.abs(Wb) / s0) * s0
            step1 = S.local_step(f, np.abs(Wb) / (2 * s0)) * (2 * s0)
            keep = ~sat & (Wb != 0)
            n_dbl += int((step1 > step0)[keep].sum())
            n_el += int(keep.sum())
        pred = ((n_blk - n_neg) * np.log2(2 / r_pos) + n_neg * np.log2(2 / r_neg)) / max(n_blk, 1)
        r_txt = f"{r_pos:.4f}" if r_neg == r_pos else f"{r_pos:.4f} (+) / {r_neg:.4f} (-)"
        md.append(f"| {SHORT[n]} | {r_txt} | {n_clip}/{n_blk} | {n_clip / max(n_blk, 1):.1%} | "
                  f"{pred:.1%} | {clip_sse:.4f} | {clip_after:.4f} | {others:+.4f} | "
                  f"{n_dbl / max(n_el, 1):.1%} | {n_help}/{n_clip} | {net:+.4f} |")
    md.append(f"\nFor scale: total weight energy ||W||^2 = {np.sum(W ** 2):.1f}. Net = (same weights at e_ocp+1 - "
              "clip SSE) + other weights' change. 'Other weights whose step doubles' = share of the unclipped "
              "non-zero weights in those blocks whose rounding step is larger at e_ocp + 1: for FP only the "
              "weights that drop into the subnormal range, for INT/BFP every weight.\n")

    # 2d'. why the integer formats like a SMALLER scale
    md.append(f"Why MXINT6 and BFP6 pick d = -1 (dataset {tag}). At d = -1 the step halves for every weight but the "
              "clip level drops to c 2^E, E = floor(log2 amax): c = 31/32 for a positive maximum (and for the "
              "sign-magnitude BFP6), c = 32/32 for a negative maximum in two's-complement MXINT6 (code -32). With "
              "one weight above that level and uniform rounding noise, d = -1 wins when (m - c)^2 < "
              "2^-8 ((N-1)/16 + 1/12), m = amax / 2^E, N = block size: m < m* = c + sqrt(...).\n")
    md.append("| format | sign of the block max | N | m* | predicted share log2(m*) | blocks with d = -1 under opt "
              "| median m of those blocks |")
    md.append("|---|---|---|---|---|---|---|")
    for n in ("MXINT6", "BFP6"):
        f = S.spec(n)
        N = f.block
        e_ocp, e_opt = res[n, "ocp"]["e"], res[n, "opt"]["e"]
        amax = np.stack([np.abs(W[:, sl]).max(1) for sl in S.block_slices(W.shape[1], N)], 1)
        neg = block_max_signed(W, N) < 0
        groups = [("+", ~neg, 31 / 32), ("-", neg, 1.0)] if f.grid_neg is not None else [("any", neg | ~neg, 31 / 32)]
        for label, sel, c in groups:
            sel = sel & (amax > 0)
            m_star = c + np.sqrt(2.0 ** -8 * ((N - 1) / 16 + 1 / 12))
            m = amax[sel] / np.ldexp(1.0, S.floor_log2(amax[sel]))
            down = (e_opt - e_ocp)[sel] == -1
            med = f"{np.median(m[down]):.3f}" if down.any() else "-"
            md.append(f"| {SHORT[n]} | {label} | {N} | {m_star:.3f} | {np.log2(m_star):.1%} of {sel.sum()} = "
                      f"{np.log2(m_star) * sel.sum():.0f} | {down.sum()} | {med} |")
    md.append("")

    # 2e. which offsets do the searches pick?
    md.append(f"### 2e. Exponent offsets chosen (relative to the class rule), dataset {tag}\n")
    offs = list(range(-4, 4))
    md.append("| format | policy | " + " | ".join(f"{d:+d}" for d in offs) + " | other |")
    md.append("|---|---|" + "---|" * (len(offs) + 1))
    for n in CANDIDATES:
        e_ocp = res[n, "ocp"]["e"]
        nz = np.stack([np.abs(W[:, sl]).max(1) > 0 for sl in S.block_slices(W.shape[1], S.spec(n).block)], 1)
        for p in ("ceil", "opt", "opt_wide", "oracle_w", "model", "opt_out", "oracle_y"):
            d = (res[n, p]["e"] - e_ocp)[nz]
            cnt = [int((d == k).sum()) for k in offs]
            md.append(f"| {SHORT[n]} | {p} | " + " | ".join(str(c) for c in cnt) + f" | {d.size - sum(cnt)} |")
    md.append("")
    md.append("Agreement of the no-rounding `model` with `opt` (same exponent, share of blocks):  " + ", ".join(
        f"{SHORT[n]} {np.mean(res[n, 'model']['e'] == res[n, 'opt']['e']):.1%}" for n in CANDIDATES) + ".\n")

    # 2f. is the coordinate-descent oracle the true optimum? (exact enumeration where it is cheap)
    lines = []
    for n in CANDIDATES:
        nb = len(S.block_slices(W.shape[1], S.spec(n).block))
        if len(S.WIDE_WINDOW) ** nb > 5000:
            lines.append(f"{SHORT[n]}: not enumerated ({len(S.WIDE_WINDOW)}^{nb} combinations per row)")
            continue
        _, _, cd = S.joint_search(W, n, H_fit, S.WIDE_WINDOW)
        _, _, ex = S.joint_search(W, n, H_fit, S.WIDE_WINDOW, exact=True)
        lines.append(f"{SHORT[n]}: rows where enumeration beats descent {int((ex < cd * (1 - 1e-12)).sum())}"
                     f"/{W.shape[0]}, objective gap {100 * (cd.sum() / ex.sum() - 1):.4f}%")
    md.append("`oracle_y` check (coordinate descent vs enumerating every combination of the 6 offsets per row): "
              + "; ".join(lines) + ".\n")

    for n in CANDIDATES + EXTRA:
        for p, _, cost in POLICIES:
            if (n, p) not in res:
                continue
            r = res[n, p]
            csv_rows.append(dict(dataset=tag, format=n, policy=p, y_rel_err_heldout=r["y_ho"],
                                 y_rel_err_fit=r["y_fit"], y_rel_err_all=r["y_all"], w_rel_err=r["w"],
                                 n_range=r["n_range"], n_saturated=r["n_saturated"], n_zeroed=r["n_zero"], sse_clip=r["sse_clip"], sse_zero=r["sse_zero"],
                                 sse_round=r["sse_round"], roundings_per_weight=cost,
                                 bits_per_elem=S.bits_per_elem(n, W.shape[1])))
    return res


# --------------------------------------------------------------------------- figure
def figure(tag, W, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    offsets = list(range(-3, 4))
    fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(13, 3.1), sharey=False)
    W2 = np.sum(W ** 2)
    for ax, n in zip(axes, CANDIDATES):
        parts = {k: [] for k in ("sse_clip", "sse_zero", "sse_round")}
        for d in offsets:
            Wq, e = S.quantize(W, n, "opt", (d,))
            pr = S.error_parts(W, Wq, e, n)
            for k in parts:
                parts[k].append(pr[k] / W2)
        tot = np.sum([parts[k] for k in parts], 0)
        ax.plot(offsets, tot, "-o", color="#0b0b0b", lw=2, ms=4, label="total")
        ax.plot(offsets, parts["sse_clip"], "--", color="#eb6834", lw=1.5, label="clipped")
        ax.plot(offsets, parts["sse_round"], "-.", color="#2a78d6", lw=1.5, label="rounded")
        ax.plot(offsets, parts["sse_zero"], ":", color="#1baf7a", lw=1.8, label="flushed to 0")
        r_opt = res[n, "opt"]
        ax.axhline((r_opt["w"] ** 2), color="#e87ba4", lw=1.2, label="opt (per block)")
        ax.set_yscale("log")
        ax.set_xticks(offsets)
        ax.set_xlabel("offset d from class rule (all blocks)")
        lo = min(min(v for v in tot if v > 0), r_opt["w"] ** 2) / 30
        ax.set_ylim(lo, max(tot) * 2)
        ax.set_title(SHORT[n], fontsize=10, loc="left")
        ax.grid(True, color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("weight SSE / ||W||^2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Dataset {tag}: clipping vs resolution as the shared exponent moves (0 = class floor rule, "
                 f"+1 = one scale up)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None, help="folder of W_<tag>.npy files, or the course .npz")
    ap.add_argument("--npz", default=None, help="course assignment_data.npz (W_A ... W_E, X_cal)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    source = args.npz or args.data
    if source is None:
        for p in (os.environ.get("CS611_DATA"), os.path.join(HERE, "data", "assignment_data.npz")):
            if p and os.path.exists(p):
                source = p
                break
    datasets = load_datasets(source or os.path.join(HERE, "data"))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

    md = ["# Scale-optimization study (generated by `python scale_study.py`; do not edit by hand)\n",
          f"Datasets: {', '.join(datasets)}. Implementation: `scale_opt.py` (independent of formats.py). "
          "Primary metric: relative Frobenius error of Y = W X on the held-out (odd) columns of X_cal, FP64; "
          "only W is quantized. Policies that use X are fitted on the even columns only.\n"]
    n_bad = crosscheck(datasets, md)
    allres, csv_rows = {}, []
    for tag, (W, X) in datasets.items():
        allres[tag] = study(tag, W, X, md, csv_rows)
        if not args.no_figure and tag == ("C" if "C" in datasets else tag):
            figure(tag, W, allres[tag], os.path.join(HERE, "figures", "fig5_scale_tradeoff.png"))

    if len(datasets) > 1:
        md.append("## 3. Format x policy x dataset (held-out Y error %)\n")
        pols = ("ocp", "ceil", "opt", "opt_wide", "model", "opt_out", "oracle_y")
        md.append("| format | policy | " + " | ".join(datasets) + " |")
        md.append("|---|---|" + "---|" * len(datasets))
        for n in CANDIDATES:
            for p in pols:
                md.append(f"| {SHORT[n]} | {p} | " + " | ".join(
                    f"{100 * allres[t][n, p]['y_ho']:.3f}" for t in datasets) + " |")
        md.append("\nWinner per dataset and policy (ties by the D6 rule, see each dataset's section 2b):\n")
        md.append("| policy | " + " | ".join(datasets) + " |")
        md.append("|---|" + "---|" * len(datasets))
        for p in pols:
            md.append(f"| {p} | " + " | ".join(allres[t]["_verdict", p] for t in datasets) + " |")
        md.append("")

    out = os.path.join(HERE, "results", "scale_study.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    for tag in datasets:
        rows = [r for r in csv_rows if r["dataset"] == tag]
        with open(os.path.join(HERE, "results", f"scale_study_{tag}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(f"wrote {out}; cross-check: {n_bad} differing elements")


if __name__ == "__main__":
    main()
