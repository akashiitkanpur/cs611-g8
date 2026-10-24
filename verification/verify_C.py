"""Reference-only numbers on dataset C (Siddhant). Run: python verify_C.py [data_dir]"""
import sys, numpy as np, reference_impl as R

import os
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
W = np.load(f"{d}/W_C.npy").astype(np.float64)
X = np.load(f"{d}/X_cal.npy").astype(np.float64)
Xho = X[:, 1::2]                                         # held-out odd columns (D6)


def run(name, rule):
    f = dict(R.FORMATS[name]); f["rule"] = rule
    Wq, st = R.quantize_matrix(W, f)
    return Wq, st


rows = [("MXFP_E2M3", "floor"), ("MXFP_E2M3", "opt"), ("MXINT6", "floor"), ("MXINT6", "opt"),
        ("BFP6", "floor"), ("BFP6", "opt"), ("MXFP_E3M2", "floor"), ("MXFP_E3M2", "opt"),
        ("MXFP_E4M1", "floor"), ("MXFP_E4M1", "ceil"), ("MXFP_E4M1_reserved", "floor"),
        ("MXFP_E4M1_onenan", "floor"), ("MXFP_E4M1_onenan", "ceil"), ("BFP6_32_8_floor", "floor")]
print(f"{'format':20s}{'rule':6s}{'Y all %':>9s}{'Y held-out %':>14s}{'W %':>8s}{'sat':>6s}{'zeroed':>8s}"
      f"{'bits/el':>9s}{'Y fp32-fp64 rel diff':>23s}")
for n, rule in rows:
    Wq, st = run(n, rule)
    y_all = R.y_error(W, Wq, X)
    y_ho = R.y_error(W, Wq, Xho)
    w = R.rel_err(W, Wq)
    # (d) same relative Y error computed with float32 matmul
    W32, Wq32, X32 = W.astype(np.float32), Wq.astype(np.float32), X.astype(np.float32)
    y32 = float(np.linalg.norm(W32 @ X32 - Wq32 @ X32) / np.linalg.norm(W32 @ X32))
    print(f"{n:20s}{rule:6s}{100*y_all:9.3f}{100*y_ho:14.3f}{100*w:8.3f}{st['sat']:6d}{st['zeroed']:8d}"
          f"{R.bits_per_element(n)['d11']:9.2f}{abs(y32 - y_all) / y_all:23.1e}")
