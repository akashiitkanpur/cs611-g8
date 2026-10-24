# Report sections: Conventions and Storage/Overhead (draft, Siddhant)

Source tags: [CLASS] Lecture 19 notebook / board notes, [SPEC] OCP MX v1.0, [OUR DECISION] plan v3 D1–D12.
**Checked against the Lecture 19 notebook on 23 Oct:** ✅ = the notebook says exactly this; ⚠️ = the notebook differs (see the storage note); "board notes" = not in the notebook (MXINT6 and BFP come from the board notes, which have not been checked here).

## A. Conventions

| Item | Convention | Source |
|---|---|---|
| Block axis | Consecutive values along each row of W, the reduction axis of Y = W·X (notebook: "along the LAST axis … consecutive input features within each output row") | [CLASS] ✅ |
| MX block size | 32. K = 100 → 3 blocks of 32 + a tail of 4 real values. The notebook zero-pads the tail to 32; the quantized values are identical; padding is excluded from all statistics | [CLASS] ✅ |
| MX shared scale | 2^e, e stored as E8M0, clamped to [−127, 127]; all-zero block: e = 0 | [CLASS] ✅ |
| Scale rule `ocp` (reported in every table) | e = floor(log2 max\|w\|) − emax_elem; the notebook computes floor(log2) exactly with `frexp`, not `log2` | [CLASS] ✅ |
| Scale rule `ceil` | smallest e with max\|w\| ≤ vmax·2^e (nothing clipped). Notebook: start from `ocp`, add 1 if max/scale > vmax; identical to ours (checked on all 1,600 blocks of W_C × 4 formats) | [CLASS] ✅ (suggested follow-up) |
| Scale rule `opt` | per block try e_ocp + {−2, −1, 0, +1}, keep lowest squared error; ties: offset closest to 0, then −1, +1, −2 | [OUR DECISION D5] |
| Element rounding | round to nearest, ties to even; then saturate at ±vmax | [CLASS] ✅ |
| Special values | none: no Inf/NaN in FP6 / INT6 / BFP6 elements (notebook FP6/FP4 entries use every code; its FP8 entries do reserve codes: E4M3 vmax 448, E5M2 57344) | [CLASS] ✅ for FP6 |
| FP subnormals | below 2^emin, fixed spacing 2^(emin−m) | [CLASS] ✅ |
| FP bias | 2^(e−1) − 1, top exponent code finite | [CLASS] ✅ for E3M2/E2M3/E2M1; [OUR DECISION D3] for E4M1 |
| MXFP6 E3M2 | 1\|3\|2, bias 3, emin −2, emax 4, vmax 28, min subnormal 1/16 | [CLASS] ✅ |
| MXFP6 E2M3 | 1\|2\|3, bias 1, emin 0, emax 2, vmax 7.5, min subnormal 1/8 | [CLASS] ✅ |
| MXFP6 E4M1 | 1\|4\|1, bias 7, emin −6, emax 8, vmax 384, min subnormal 2⁻⁷; no reserved code. Not in the notebook. Support: the notebook's 6- and 4-bit formats reserve no code; its E4M3 (same 4-bit exponent, bias 7, emin −6) does reserve one NaN, which is why the one-NaN row (vmax 256) is reported | [OUR DECISION D3] |
| E4M1 sensitivity rows | top exponent reserved: vmax 192; one NaN (top exp + mantissa 1): vmax 256 | [OUR DECISION D3] |
| MXINT6 | S I F F F F: value k·2⁻⁴·2^e, k ∈ [−31, 31] (code −32 never produced), emax_elem = 0 | board notes (not in notebook), [OUR DECISION D10] range |
| BFP6 | g\|e\|m = 16\|4\|5: groups of 16 along the row (100 = 6·16 + 4), 4-bit shared exponent clamped to [−8, 7], sign + 5-bit magnitude, value k·2⁻⁴·2^e, k ∈ [−31, 31], e = floor(log2 max\|w\|), RNE | board notes structure (not in notebook), [OUR DECISION D1] parameters |
| BFP vs MXINT | BFP 32\|8\|5 with the floor rule is numerically identical to MXINT6 (verified on all 10,000 weights of C); our BFP6 differs through group size and exponent width | [OUR DECISION D1], verified |
| Quantized operands | W only; X stays FP32. W+X is a secondary experiment (the notebook runs both W-only and W+A, activations blocked along their last axis) | [OUR DECISION D2], consistent with notebook |
| Error arithmetic | FP64; FP32 accumulation changes relative Y error by ≤ 1.1·10⁻⁷ relative (notebook: FP32 reference with TF32 disabled, error sums in FP64) | [OUR DECISION D8], verified |
| Primary metric | relative Frobenius error of Y on held-out (odd) columns of X_cal | [OUR DECISION D6] |
| Selection granularity | one format for the whole matrix | [OUR DECISION D4] |

## B. Storage

Bits per element include the shared exponent. For a row of K = 100 values (D11: the tail block stores only its real values plus its own exponent):

> ⚠️ **The Lecture 19 notebook counts storage differently.** Its `ideal_weight_size` stores every block in full, padding included: `blocks = rows · ceil(cols/32)`, `bits += blocks · (32·bits + 8)`. For K = 100 that gives **8.00 bits/element for MX formats** (and 7.00 for BFP 16|4|5 by the same rule), not 6.32 / 6.28. D11 (6.32) is therefore [OUR DECISION], not [CLASS]. The report should give both and say which one the equal-budget claims use. Under either accounting all five candidates are compared at the same element width (6 bits) and the C ranking does not depend on it.

| Candidate | element bits | block | exponent bits | blocks/row | bits/row | **bits/element** | long-row limit |
|---|---|---|---|---|---|---|---|
| MXFP6 E3M2 / E2M3 / E4M1 | 6 | 32 | 8 | 4 | 600 + 32 = 632 | **6.32** | 6.25 |
| MXINT6 | 6 | 32 | 8 | 4 | 632 | **6.32** | 6.25 |
| BFP6 16\|4\|5 | 6 (1 sign + 5 magnitude) | 16 | 4 | 7 | 600 + 28 = 628 | **6.28** | 6.25 |
| BFP 32\|8\|5 (≡ MXINT6) | 6 | 32 | 8 | 4 | 632 | 6.32 | 6.25 |
| BFP6_row (one block/row) | 6 | 100 | 8 | 1 | 608 | 6.08 | – |

Whole matrix W (100×100): FP32 = 320,000 bits; MX formats = 63,200 bits (5.06× smaller); BFP6 16|4|5 = 62,800 bits (5.10× smaller).
If hardware stores the padded tail in full, MX costs (4·32·6 + 4·8)/100 = 8.00 and BFP6 16|4|5 costs (7·16·6 + 7·4)/100 = 7.00 bits/element; we report the D11 numbers and state this alternative in one sentence.

**Equal budget.** All five candidates are within 6.28–6.32 bits/element (0.6%), so the comparison is at equal budget. BFP6_row (6.08) and any block-size study (e.g. block 8: 6 + 8/8 = 7.0) are not, and are always quoted with their bits/element.

## C. Computational overhead of block scaling

Counts for Y = W·X with W of size M×K, X of size K×N, block size B (operation counts, not hardware claims [ASSUMPTION]).

- **Quantizing W (once, offline):** per block, B−1 comparisons for the maximum, one exponent extraction, and B element roundings. The scale is a power of two, so dividing by it is an exponent subtraction, not a multiplication.
- **Matrix multiply:** each dot product of length K is split into ⌈K/B⌉ partial sums of narrow products; each partial sum is multiplied by 2^(e_w) (an exponent add) and accumulated in wider precision. Extra work per output: ⌈K/B⌉ scale applications per K MACs, i.e. **4 per 100 MACs for MX (4%)** and **7 per 100 for BFP6 16|4|5 (7%)** at K = 100; 1/B in the long-row limit (3.1% and 6.3%).
- **Element MACs:** MXINT6 and BFP6 multiply integers (sign + 5 bits); MXFP6 multiplies small floats (add the element exponents, multiply mantissas of 2–4 bits), which needs a small shifter before accumulation [ASSUMPTION, from Lecture 12's framing].
- **Selection cost** is in Aryan's cost model, not here.
