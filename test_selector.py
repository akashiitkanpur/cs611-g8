"""Unit tests for selector.py.  Run:  python test_selector.py   (or: pytest test_selector.py)

The key property: the 64-bin sketch reproduces the class-rule error of every candidate exactly
(to float round-off), weighted or not, on matrices built to hit every corner of the formats:
ties, saturation, subnormals, flush to zero, both ends of BFP's 4-bit exponent, the
two's-complement -32 code, zero blocks and the tail block of 4.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selector as S                                            # noqa: E402
import selector_suite as T                                      # noqa: E402
from formats import quantize                                    # noqa: E402

rng = np.random.default_rng(11)


def corner_matrices():
    yield "t3", rng.standard_t(3, (48, 100))
    yield "sparse", rng.standard_normal((48, 100)) * (rng.random((48, 100)) > 0.85)
    yield "16 octaves", rng.choice([-1, 1], (48, 100)) * 2.0 ** rng.uniform(-16, 0, (48, 100))
    yield "bfp clamp high", 1e4 * rng.standard_normal((16, 100))
    yield "bfp clamp low", 1e-5 * rng.standard_normal((16, 100))
    yield "dyadic ties", rng.choice([-1, 1], (32, 100)) * rng.integers(1, 64, (32, 100)) / 32.0
    yield "exact outliers", np.where(rng.random((48, 100)) < 0.02, 20.0, rng.standard_normal((48, 100)))
    W = rng.standard_normal((24, 100))
    W[::3, :32] = 0                                              # zero blocks
    W[1::3, 96:] = -1.99 * np.abs(W[1::3, 96:]).max()            # tail block: -32 code of MXINT6
    yield "zero + tail", W
    yield "fp32 data", rng.standard_normal((48, 100)).astype(np.float32).astype(np.float64)


def test_sketch_is_exact_for_the_class_rule():
    for name, W in corner_matrices():
        for h in (None, rng.uniform(0.01, 5, W.shape[1])):
            ex = S.exhaustive(W, h)
            sk = S.sketch_select(W, h)
            for n in S.CANDIDATES:
                assert abs(sk[n] - ex[n]) <= 1e-9 * max(ex[n], 1e-300), (name, n, sk[n], ex[n])


def test_unit_errors_match_and_certified_uses_everything_when_needed():
    W = rng.standard_normal((40, 100))
    ex = S.exhaustive(W)
    full = S.certified(W, n0=10 ** 9)
    assert S.certified.last[0] == S.certified.last[1]
    for n in S.CANDIDATES:
        assert abs(full[n] - ex[n]) <= 1e-9 * ex[n], n


def test_sketch_is_additive():
    # sketch(top rows) + sketch(bottom rows) == sketch(all rows): mergeable across shards
    W = rng.laplace(size=(60, 100))
    a, b, c = S.sketch(W[:25]), S.sketch(W[25:]), S.sketch(W)
    for fam in c:
        assert np.allclose(a[fam] + b[fam], c[fam], rtol=1e-12, atol=0)


def test_sampled_sketch_is_unbiased():
    W = rng.standard_normal((64, 100))
    ex = S.exhaustive(W)["MXFP6_E2M3"]
    est = [S.sketch_select(W, frac=0.25, rng=np.random.default_rng(s))["MXFP6_E2M3"] for s in range(200)]
    m, se = np.mean(est), np.std(est) / np.sqrt(len(est))
    assert abs(m - ex) < 4 * se, (m, ex, se)


def test_coarse_sketch_converges_to_exact():
    W = rng.standard_t(4, (64, 100))
    ex = S.exhaustive(W)
    err = [max(abs(S.sketch_select(W, nb=nb)[n] / ex[n] - 1) for n in S.CANDIDATES) for nb in (1, 4, 16, 64)]
    assert err[-1] < 1e-9 and err[0] > err[-1], err


def test_closed_form_matches_rho_on_dataset_C():
    # the worked example: rho = 4.9786 on W_C gives MXINT6 NSR 1.6206e-3 (true 4.025% error)
    path = os.path.join(HERE, "data", "W_C.npy")
    if not os.path.exists(path):
        print("  (skipped: data/W_C.npy not found)")
        return
    W = np.load(path).astype(np.float64)
    nsr = S.closed_form(W)["MXINT6"] / (W ** 2).sum()
    assert abs(nsr - 1.6206e-3) < 2e-7, nsr
    true = S.exhaustive(W)["MXINT6"] / (W ** 2).sum()
    assert abs(np.sqrt(true) - 0.04025) < 1e-4, np.sqrt(true)


def test_m1_two_complement_negative_max_does_not_clip():
    W = np.zeros((1, 32))
    W[0, 0] = -1.99                                              # fits the -32 code: no clipping
    W[0, 1] = 0.5
    assert S.m1(W)["MXINT6"] < S.m1(-W)["MXINT6"]


def test_registry_runs_every_approach():
    W = rng.standard_normal((32, 100))
    h = rng.uniform(0.5, 2, 100)
    tree = S.fit_tree(np.array([S.features(rng.standard_normal((16, 64))) for _ in range(30)]),
                      ["MXFP6_E2M3"] * 15 + ["BFP6"] * 15, depth=1, min_leaf=5)
    for name in S.APPROACHES:
        sc, sec = S.run(name, W, h, **({"tree": tree} if name == "learned_tree" else {}))
        assert set(sc) == set(S.CANDIDATES) and sec >= 0, name


def test_suite_is_deterministic():
    a = [W.sum() for _, W, _ in T.random_suite(5)]
    b = [W.sum() for _, W, _ in T.random_suite(5)]
    assert a == b
    assert len(list(T.adversarial())) == 28


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
