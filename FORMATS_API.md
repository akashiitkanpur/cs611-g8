# formats.py v2: how to use it (Akash)

v1 calls still work: `quantize(W, name)`, `quantize(W, name, policy=..., H=...)`, `metrics(W, Wq, X)` and `exhaustive(W, X)`.
**What changed in v2:** `"BFP6"` is now the D1 candidate 16|4|5, so its numbers changed. The v1 BFP6 is `"BFP6_b32_ceil"`. `n_saturated` now uses the Lecture 19 definition (see Stats).

## Formats

```python
from formats import FORMATS, VARIANTS, spec, layout
list(FORMATS)    # the 5 candidates: MXFP6_E3M2, MXFP6_E2M3, MXFP6_E4M1, MXINT6, BFP6
layout("BFP6")   # {'block': 16, 'policy': 'ocp', 'scale_bits': 4, 'field': (-8, 7)}
```

| name | role | definition | bits/elem (100x100) |
|---|---|---|---|
| MXFP6_E3M2 / E2M3 / E4M1 | candidates | class FP6; E4M1 vmax 384 (D3) | 6.32 |
| MXINT6 | candidate | k·2^-4, two's complement k in [-32, 31] (D10, revised 23 Oct) | 6.32 |
| BFP6 | candidate | 16\|4\|5: 4-bit E = floor(log2 max) in [-8, 7], ±M·2^(E-4), M ≤ 31 (D1) | 6.28 |
| MXFP6_E4M1_v192 | variant | top exponent reserved, vmax 192 | 6.32 |
| MXFP6_E4M1_nan1 | variant | one NaN code, vmax 256 | 6.32 |
| BFP6_g32e8 | variant | BFP 32\|8\|5, floor: identical to MXINT6_sym | 6.32 |
| BFP6_b32_ceil | variant | v1's BFP6 (block 32, ceil) | 6.32 |
| BFP6_row | variant | one block per row, ceil | 6.08 |
| MXINT6_sym | variant | symmetric k in [-31, 31] (the MXINT6 of v1/v2) | 6.32 |

Code that loops over `FORMATS` sees only the five candidates.

## Quantize

```python
from formats import quantize, quantize_X, metrics

Wq, exps = quantize(W, "MXFP6_E2M3")                    # class rule ("ocp"), the format's own block
Wq, exps = quantize(W, "MXINT6", policy="ceil")         # "ocp" | "ceil" | "opt"
Wq, exps = quantize(W, "MXFP6_E3M2", policy="opt", H=X @ X.T)   # output-aware exponent search
Wq, exps, st = quantize(W, "BFP6", return_stats=True)   # + counts and true bits/elem
Wq, exps = quantize(W, "MXINT6", block=16)              # override the block size
```

- `exps[i, b]` is the **stored exponent field** of block `b` in row `i`. For MX formats this is the E8M0 value (scale = 2^exps). For BFP it is E, and the step of the integer magnitude is 2^(E−4). An all-zero block stores 0.
- Policies: `ocp` = floor(log2 max) − emax (class), `ceil` = smallest scale that clips nothing, `opt` = try ocp + {−2, −1, 0, +1} per block and keep the lowest error. The search stays inside the format's exponent field.

## W+X mode (D2, secondary experiment)

```python
Xq, ex = quantize_X(X, "MXFP6_E2M3")     # blocks run DOWN the columns of X (its reduction axis)
m = metrics(W, Wq, X, Xq=Xq)              # Y error of Wq @ Xq; also x_rel_err
```

## Stats (`return_stats=True`)

`n_range` (|w|/scale > vmax before rounding), `n_saturated` (rounded value clipped: the Lecture 19 definition; **v1 reported n_range under this name**), `n_zeroed`, `n_subnormal`, `n_scale_clamped` (blocks whose exponent hit the field limits), `bits_per_elem` (true, tail block included), `n_roundings`.

## Ground truth for the selector (Aryan)

```python
from formats import exhaustive, ranking
rows = exhaustive(W, X)                 # 5 formats x {ocp, ceil, opt, opt_out}; tqdm bar; held-out split
rows = exhaustive(W, X, mode="wx")      # X quantized too
ranking(rows, policies=("ocp",))        # [(format, policy, y_rel_err), ...], best first
```

Each row has `y_rel_err_heldout` (primary, D6), `n_roundings` and `n_macs` (for the cost model) and a `fingerprint`. A checkpoint row is reused only if its fingerprint (data, format, policy, mode, formats.py version) matches.

## Regenerate everything

`python run_all.py` runs the tests, the exhaustive baselines (W and W+X), all tables (`results/tables/T1..T8`), the team scripts and the figures. It also writes `results/RUN_MANIFEST.md`. Datasets A–E are included if `data/assignment_data.npz` exists (or pass `--npz`).

## Tests

`python test_formats.py` runs 30 tests: the 18 from v1 (two adapted to D1) and 12 new ones. The new tests cover the BFP6 hand example and 4-bit clamp, 32|8|5 = MXINT6_sym, BFP6 = 3.46% on C, E4M1 one-NaN (20.7%), range vs saturated, two's complement, storage per D11, W+X blocking, metrics and exhaustive in W+X, and stale checkpoints.
