# Format selector: method, approaches compared, results (Aryan, 24 Oct)

Code: `selector.py` (all approaches, one interface), `selector_suite.py` (permanent test matrices), `test_selector.py` (9 tests), `bench_selector.py` (accuracy vs cost), `bench_scaling.py` (cost vs matrix size), `make_selector_figures.py` (fig6, fig7). Everything uses `formats.py` v2.1 (five candidates, MXINT6 two's complement, BFP6 16|4|5).

```
python test_selector.py
python bench_selector.py --npz assignment_data.npz    # about 4 min; results/selector_bench.{csv,md}
python bench_scaling.py                               # results/selector_scaling.md
python make_selector_figures.py
```

## 1. The two "sites of optimization" (TA)
- **Way 1, from the weights:** pick the format with the smallest weight error Σ(W − Wq)².
- **Way 2, from the activations as well:** the output error is ‖(W − Wq)X‖² = Σᵢ Δᵢᵀ H Δᵢ with H = XXᵀ. With independent zero-mean rounding errors its expectation is Σᵢⱼ H_jj Δᵢⱼ². So each weight is weighted by the energy of its input channel. This is the same reason AWQ (Lin et al. 2023) finds salient weights from the activation distribution rather than from the weights themselves.

## 2. Main result: a format-independent sketch gives the exact error of every format
For a block with reference exponent R (= floor(log2 max|w|); for BFP, clamped to its 4-bit field), write each non-zero weight as |w| = μ·2ᵗ with μ ∈ [1, 2), and let d = R − t be its octave offset below the block peak.

Under the class rule every candidate's scale is 2^(R − emax_f). So the quantized magnitude is c_f(d, μ, sign)·2ᵗ, and c_f depends on the format alone. The squared error of one weight is

  4ᵗ(μ − c)² = m2 − 2c·m1 + c²·m0,  with m0 = 4ᵗ, m1 = 4ᵗμ, m2 = w².

Bin the weights by (sign, d, mantissa bin of μ) and keep the three sums, weighted by H_jj for way 2. That is the **sketch**: 2 × 49 × 64 × 3 numbers per block layout (MX blocks of 32; BFP blocks of 16), and it is the same for every format.

Each format's error is then a fixed linear function of the sketch, error_f = ⟨kernel_f, sketch⟩, where the kernel comes from the format's definition and needs no data.

**It is exact, not an approximation.** For 6-bit formats every rounding midpoint and saturation point lies on a multiple of 1/32 of a binade. So with 64 mantissa bins c_f is constant on each bin, and the moments give the error exactly. `test_selector.py` checks this to 1e-9 for all five candidates, weighted and unweighted, on 9 corner-case matrices:
- ties at exact grid values
- the −32 code of MXINT6
- both ends of BFP's exponent clamp
- zero blocks and the tail block of 4
- 16 octaves of dynamic range
- FP32 data

**Properties:**
- One pass, shared by all candidates. Scoring a format costs O(#bins), whatever the size of W. Adding a format, or a different bit width, means writing its kernel, not touching the data.
- Additive: sketches of shards (rows, layers, devices) add up, so it streams and merges.
- Sampling a fraction of blocks gives an unbiased estimate with an error bar (tested).
- Coarser sketches form a hierarchy:

  | Level | What is kept | Top-1 on the random suite |
  |---|---|---|
  | 64 bins | exact | 92.3% |
  | 16 bins | | 91.2% |
  | 4 bins | | 87.1% |
  | 1 bin per octave | | 79.7% |
  | Two-statistic closed form | ρ (block peaks), φ (element binades) | about 40% |

  The closed form is the coarsest level of the same framework. It fails because it has no flush and clip terms.

The weighted-histogram idea is Akash's `predictor.py`. This version makes it exact and adds BFP's 16-blocks and exponent clamp and the two's-complement sign asymmetry.

## 3. Every approach we tried (kept for the accuracy-vs-cost figure)
| Approach | Cost | Idea |
|---|---|---|
| `learned_tree` | features | CART (depth 4) on sparsity, kurtosis, ρ32, ρ16, φ and the flush fraction, trained on 600 separate random matrices |
| `closed_form(_h)` | 2 statistics | INT/BFP: 4⁻⁴/12·Σ4^R; FP: 4^−m/12·Σ4ᵗ |
| `sketch_nb{1,4,16}_h`, `sketch(_h)` | 1 pass, no rounding | §2 |
| `sketch_s10_h` | 10% of blocks | §2 on a uniform sample |
| `certified_h` | adaptive sample | uniform sampling, doubled until the winner leads every other format by 3 standard errors |
| `certified_is_h` | block maxima + adaptive sample | importance sampling ∝ Σ h·4^F per block, Hansen–Hurwitz estimate, same 3 s.e. stopping rule |
| `akash_hist_h` | 1 pass | `predictor.py`: expected error per histogram bin, block maxima exact; BFP adapted to 16-blocks, no clamp |
| `m1(_h)` | 1 pass | per-element expected error: step²/12, flush z², clip (z − vmax)² |
| `hybrid_h` | 1 pass + rounding of about 6% | M1 with the top octave rounded exactly |
| `exh_rows10` | quantize 10% of rows | sampled brute force |
| `exhaustive_w/_h/_y` | quantize all 5 | brute force: weight error, diagonal-H error, full Y error on the fit columns |

## 4. Results (`results/selector_bench.md`, `figures/fig6`, `figures/fig7`)
Truth is the held-out Y error (odd X columns) under the class rule. The test sets are:
- **Datasets:** A–E.
- **Adversarial:** 28 hand-made families.
- **Random:** 1000 matrices, 128×128. Each is a mix of heavy tails, sparsity, random, channel or block outliers with continuous magnitudes, and row and column gains. Each has one of four X families: white, channel gains, outlier channels, or low rank.

| Approach (random suite) | Top-1 | Mean regret | Max regret | ms / matrix |
|---|---|---|---|---|
| closed form | 41% | 2.1 dB | 21 dB | 0.44 |
| learned tree | 73% | 0.67 dB | 17 dB | 0.85 |
| brute force, weights only (way 1) | 78.8% | 0.55 dB | 17 dB | 1.8 |
| exact sketch, weights only (way 1) | 78.8% (identical picks) | 0.55 dB | 17 dB | 0.77 |
| M1, H-weighted | 88.1% | 0.10 dB | 8.3 dB | 1.7 |
| `predictor.py`, H-weighted | 89.5% | 0.08 dB | 8.3 dB | 2.9 |
| hybrid, H-weighted | 90.0% | 0.08 dB | 8.3 dB | 3.5 |
| **exact sketch, H-weighted (way 2)** | **92.3%** | **0.051 dB** | 3.0 dB | **0.77** |
| brute force, H-weighted | 92.3% (identical picks) | 0.051 dB | 3.0 dB | 1.8 |
| certified, importance-sampled | 92.0% (reads 17% of blocks, median) | 0.055 dB | 3.1 dB | 3.0 |
| brute force, full Y on the fit columns | 95.3% | 0.012 dB | 2.2 dB | 5.6 |

- **Way 2 matters.** With non-white activations, choosing by weight error alone (even exactly) picks the wrong format in 21% of matrices: 61% top-1 on outlier channels, 77% on channel gains. H-weighting raises these to 95% and 96%.
- **On white X, and on A–E (X_cal is nearly white), way 1 and way 2 agree.** On A–E every exact method picks the winner, or a tie within 0.000 dB: B is BFP6 vs MXINT6, 1.781 vs 1.782.
- **The remaining gap is correlated X.** With low-rank X the diagonal of H is not enough: 83% vs 97% for brute force with the full H. The realized error then depends on the projections of ΔW onto a few directions, which an expectation cannot predict. This is the limit of any rounding-free method.
- **Scaling** (`results/selector_scaling.md`, LLM-like 4096×4096 = 16.8M weights): brute force 2.0 s, exact sketch 2.2 s (memory-bound in NumPy), certified 0.17 s while rounding 0.10% of blocks, with the same pick. The sample the certifier needs depends on the margin between formats, not on the matrix size: 0.10% of blocks at 16.8M weights, 2.8% when the margin is only 0.34 dB.
- **Winners in the random suite:** BFP6 671, E2M3 238, E3M2 68, MXINT6 23. BFP6 16|4|5 has the same grid as MXINT6 on blocks half as long, at fewer bits (6.28 vs 6.32), so it wins whenever its 4-bit exponent does not clamp.
- **Class rule vs `opt`:** the class-rule winner equals the `opt` winner in 94% of matrices.

## 5. Per-block mixed formats (excluded by D4; only to answer "would it help?")
Using the per-block errors from §2, an oracle that picks the best of the four MX formats in every 32-block lowers the weight error only a little, and costs +2 bits per block (+0.06 bits/element):

| | A | B | C | D | E |
|---|---|---|---|---|---|
| Relative reduction | 4.1% | 0.3% | 3.1% (3.082% → 2.986%) | 3.4% | −2.3% (worse than BFP6 alone) |

MixFP4 (Zou et al. 2026) picks E2M1 or E1M2 per block and stores the choice in the unused sign bit of the FP8 E4M3 scale. Our E8M0 scale has no free bit.

## 6. Ideas from the papers (checked against the abstracts)
- **AWQ** (Lin et al. 2023): salience comes from activations, not weights. This supports way 2.
- **Linearity theorem / HIGGS** (Malinovskii et al. 2024): per-layer ℓ2 reconstruction error relates directly to the perplexity increase, and non-uniform per-layer levels under a size budget are solved optimally by dynamic programming. With exact per-format errors per matrix (§2), choosing one format per matrix across a whole model under a bit budget becomes the same dynamic program. This is a natural extension, still inside D4 (one format per matrix).
- **Kuzmin et al. 2022 (FP8):** the best exponent/mantissa split is "driven by the severity of outliers". Our kernel is the data-exact version of their analytic MSE.
- **van Baalen et al. 2023 (FP8 vs INT8):** a theoretical INT vs FP comparison.
- **ANT** (MICRO 2022), **MoFQ** (Zhang et al. 2023): per-tensor or per-layer type selection. The abstracts do not say how the type is chosen. Check the full papers before citing a mechanism.

## 7. Limitations
- **Class rule only.** `opt` is not modelled exactly. The per-block sketch can score each offset, but this is not implemented yet.
- **Diagonal H only.** See the low-rank result in §4.
- The 4-bit BFP exponent clamp is modelled for d ≥ −8; blocks more than 8 octaves above the clamp are not.
- **Timing:** NumPy on one laptop. The 128×128 timings include Python overhead, which dominates for the sampled methods at that size.
- **The random families are our own.** The test matrices are synthetic. Only A–E are course data, and those are synthetic too.
