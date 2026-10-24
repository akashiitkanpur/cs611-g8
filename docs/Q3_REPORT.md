# Q3: how outliers, sparsity and variation in scale affect the selection (dataset C, 6-bit)

Rework of Nikita's draft, following the review (`docs/REVIEW_Q3_REPORT.md`) and issue #5. Every number comes from `python q3_study.py` (about 20 s), which writes `results/q3_study.md` and `figures/fig8`–`fig10`.

## Setup
| Item | Convention |
|---|---|
| MX blocks | 32 consecutive values along each row of W (the reduction axis); K = 100 gives 32+32+32+4, and the tail block has its own scale. E8M0 power-of-two scale. Storage 6.32 bits/element |
| BFP6 | 16\|4\|5: blocks of 16, 4-bit shared exponent E = floor(log2 max) clamped to [−8, 7], sign + 5-bit magnitude. 6.28 bits/element |
| Elements | E3M2 (vmax 28), E2M3 (vmax 7.5), E4M1 (no reserved code, vmax 384), MXINT6 two's complement k ∈ [−32, 31]. Round to nearest even, saturate, no Inf/NaN |
| Scale rules | class rule `ocp`: e = floor(log2 max) − emax_elem. `opt`: best of e_ocp + {−2, …, +1} per block |
| What is quantized | weights only; X stays FP32; evaluation in FP64 |
| Metric | relative Frobenius error of Y = W·X on the held-out odd columns of X_cal |
| Seeds | 10 per point. Margin = runner-up − winner in percentage points (pp), mean ± sd over seeds |

Unmodified W_C: E2M3 3.101% < BFP6 3.484% < MXINT6 4.054% < E3M2 5.469% < E4M1 11.149%. E4M1 never wins any experiment below.

## 1. Outliers: E3M2 never wins; very large outliers favour BFP6
We replaced n weights of W_C by ±k·σ, with continuous magnitudes k·σ·U(0.7, 1.3).

- **Up to 20σ, E2M3 wins every cell in 10/10 seeds.** This holds even with 400 outliers touching 58% of the blocks. Its margin over BFP6 grows from 0.38 to 0.67 pp.
- **At 50σ it is a tie** between E2M3 and BFP6 (for example 100 outliers: +0.01 ± 0.11 pp, E2M3 in 5/10 seeds).
- **At 100σ, BFP6 wins for every n** (7–10/10 seeds, +0.2 to +0.7 pp). `opt` gives the same winners.
- **Mechanism:** an outlier sets its block's scale, so the other values in that block get a coarse grid.
  - With 100 outliers at 20σ, E2M3 flushes 1,066 values to zero against 461 without outliers; MXINT6 flushes 1,904 and BFP6 1,289.
  - At 100σ the outlier's neighbours are flushed in every format, so the error is roughly the energy of the neighbours that share its block. A BFP6 block has 15 neighbours, an MX block 31. Our reading is that this is why BFP6's half-size blocks win there.
- **E3M2 keeps the most neighbours** (160 flushed), but its top binade is twice as coarse (2 mantissa bits instead of 3). With continuous magnitudes the outliers themselves carry most of the energy, so it never wins.
- **The earlier E3M2 result is an artifact.** Nikita's protocol used exact magnitudes k·σ. It reproduces her E3M2 win (100 × 20σ: E3M2 2.39% vs E2M3 3.45%), but only because 20σ is exactly representable in E3M2. At 50σ the outliers sit exactly on an E2M3 rounding midpoint, its worst case (issue #5).
- **A better x-axis** than the outlier count is the share of blocks containing an outlier: 1 outlier touches 0.2% of the blocks, 100 touch 21%, 400 touch 58%.

## 2. Sparsity: the integer grids overtake E2M3, BFP6 already at about 40% zeros
- **E2M3 stays almost constant** (3.10% → 2.90%) as random weights are set to zero.
- **BFP6 falls steadily** (3.48% → 1.68% at 95%). It ties E2M3 at 40% zeros (+0.01 ± 0.05 pp, 5/10 seeds) and wins in 10/10 seeds from 50% on, by up to 0.49 pp at 75%. MXINT6 crosses E2M3 only at about 75% (2.88% vs 2.88%), which is the crossover Nikita reported. It belongs to MXINT6; with D1's BFP6 the integer side wins much earlier.
- **Mechanism:**
  - FP error is relative to each value's own binade, so its statistic φ = Σ4^floor(log2|w|)/Σw² stays at 0.545 at every sparsity level.
  - Integer error is set by each block's maximum: NSR = ρ·4⁻⁴/12, with ρ = Σ n_nz·4^F/Σw². The maximum of fewer non-zero values is smaller relative to their energy, so ρ falls with sparsity: for 16-blocks from 3.60 (0%) to 2.74 (40%) and 1.51 (80%).
  - The closed form predicts the crossover where ρ16·4⁻⁴ equals E2M3's φ·4⁻³ times its subnormal factor of about 1.34, that is at ρ16 ≈ 2.9. This is reached between 30% and 40% zeros, as observed. BFP6 crosses before MXINT6 because 16-value blocks have smaller maxima than 32-value blocks (ρ16 < ρ32).

## 3. Variation in scale: the global winner holds; weak channels favour wide-range formats
- **Gains per output row** (10^±0.5 up to 10^±2, max/min up to 10⁴): every block lies inside one row, so its scale absorbs the gain. All errors stay flat and E2M3 wins, including on the weakest 25% of rows.
- **The exception is BFP6 at 10⁴.** Rows with gain near 100 have block maxima above BFP6's 4-bit exponent range (2⁷·31/16 = 248). 8 of 700 blocks clamp and saturate, and BFP6's error jumps to 9.6%. This is a real limit of a 4-bit shared exponent: BFP6 is not scale-invariant, while the MX formats with their 8-bit scale are.
- **Gains per input column** (inside every block): the global error still picks E2M3 at every spread. It is a near-tie at 10³ (+0.09 ± 0.10 pp) and BFP6 clamps again at 10⁴.
- **Judged on the part of Y that comes from the weakest 25% of channels, the ranking flips:**

  | Spread (max/min) | E3M2 | E2M3 | BFP6 | E4M1 |
  |---|---|---|---|---|
  | 10² | 7.2% | 36.7% | 47.7% | |
  | 10³ | 25.5% | 80.8% | | **10.5%** |
  | 10⁴ | 74.6% | | | **10.2%** |

  The block scale follows the strongest channel in each block, and only the formats with many exponent bits keep the weak channels representable.
- **With a global metric and a nearly white X**, the weak channels contribute almost nothing to Y, so E2M3 stays best. If the activations amplified the weak channels, the H-weighted selector (way 2, `docs/SELECTOR.md`) would see it: weighting by (XXᵀ)_jj is exactly what moves the objective toward those channels.

## 4. What this means for the selector
The winner moves with three measurable quantities:
- block-peak energy (ρ, which falls with sparsity and favours BFP6 and MXINT6)
- energy flushed to zero inside outlier blocks
- whether any block leaves BFP6's 4-bit exponent range

The exact sketch selector (`selector.py`, H from the even columns) picks the held-out winner in every cell except near-ties: 80–90% of seeds where the margin is smaller than its seed-to-seed sd. All three quantities are in its sketch.

## Changes from the first draft
- Uses D1's BFP6 (16|4|5) and D10's MXINT6; the old BFP6 was numerically MXINT6.
- Held-out metric, with both `ocp` and `opt`.
- Continuous outlier magnitudes; the exact-magnitude protocol kept only to show the artifact.
- Sparsity crossover on a finer grid with 10 seeds.
- Weak-group metric alongside the global one; the row-gain experiment explained as a near-null control.
- Mechanism counts and closed-form statistics; three figures; one script regenerates everything.

**Not redone here:** the block-size study (review §3.3). The scale-rule study is Anuj's `results/scale_study.md`.
