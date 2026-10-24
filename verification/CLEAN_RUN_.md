# Saturday check of latest main (Siddhant), 24 Oct

Commit tested: `f4aafa9` (merge of PR #8, 24 Oct 06:30), includes formats.py v2.1 (revised D10: MXINT6 is two's complement), Aryan's selector and Anuj's scale study.

## 1. Results

| Check | Result |
|---|---|
| Bit-exact cross-check, `formats.py` v2.1 vs `verification/reference_impl.py` (11 formats/variants × policies × adversarial + 5 random families + W_C) | **217 / 217 bit-exact** |
| Clean environment: `git archive` copy, new venv, `pip install numpy matplotlib tqdm` (numpy 2.5.4, matplotlib 3.11.2, Python 3.13.16, Linux), `python run_all.py --fresh` | **exit 0, 22 s** |
| Tests | `test_formats.py` 30/30, `test_scale_opt.py` 16/16, `test_selector.py` 9/9 |
| Regenerated vs committed outputs | T1–T6, `scale_study.md`, `selector_bench.md`, `selector_scaling.md`: **identical**. `baseline_C*.csv`: same rows, every count identical; floats differ by at most **5.5·10⁻¹⁶** relative (committed run was macOS arm64 / numpy 2.5.1) |
| Revised MXINT6 (two's complement) on C, reference | held-out 4.054 (ocp) / 3.880 (opt), all-X 4.048 / 3.850, saturated 6, zeroed 891: **same as T1 and T6** |

**Conclusion: the code on `main` reproduces its own tables from a clean environment on a different OS and numpy version, and the quantizers agree bit for bit with an independent implementation.**

## 2. Problems to fix before submission

1. **T4 contradicts itself (stale after the D10 revision).** `run_all.py` lines 221 and 235 still say BFP 32|8|5 is "(= MXINT6)" and "numerically identical to MXINT6". Since D10 changed, it is identical to **MXINT6_sym**. T4 shows BFP6_g32e8 opt = 3.938 next to MXINT6 opt = 3.880. Fix: change the label and caption to `MXINT6_sym`, and add an `MXINT6_sym` reference row. Same stale claim in `run_baseline.py` lines 67–69 (compares `BFP6_g32e8` with `MXINT6`, so it now prints a misleading list), `notes/akash.md` lines 6, 12, 18, and the comment at `test_formats.py` line 210. (The test itself, line 300, already compares with `MXINT6_sym` and passes.)
2. **T7 is not regenerated without the course file.** With no `assignment_data.npz`, `run_all.py` skips A–E but leaves the old committed `T7_datasets_A_to_E.md` in place, so a reader cannot tell if it matches the current code (T8 *is* rewritten with C only). Either make `run_all.py` delete or mark T7 as "not regenerated", or ship the npz path in the README. I could not verify T7 (no npz here).
3. **Stray file** `crosscheck_report.json` in the repo root (added in `75ef94a` by running `verification/crosscheck.py` from the root). Delete it; the updated `crosscheck.py` now always writes its report next to itself.
4. **Manifest.** Committed `RUN_MANIFEST.md` says "6f31dfb (with uncommitted changes)". For the final zip, run `python run_all.py --fresh` on the final clean commit so the manifest names it.
5. **README** still says "(30 tests)" only, has no `requirements.txt`, no verification section, and does not mention `selector.py`, `scale_opt.py` or the three test files. Proposed text: `README_proposed.md` (needs a line for the selector and scale-study files).
6. **Report draft is stale** (`report/report_draft.docx`, 21 Oct). It still has the v1 BFP6 (block 32, ceil, 4.17%), MXINT6 as sign-magnitude symmetric, "[TO CONFIRM]" conventions, "assumptions until checked", 6.32 bits as the only storage number. See the checklist below.
7. **D8 wording.** `docs/DECISIONS.md` says FP32 matmul agrees "to ~1e-8"; measured maximum is 1.1·10⁻⁷ relative. Say "< 10⁻⁶".

## 3. Checklist for the final report (numbers must match these files)

| Claim in the report | Value | Source file |
|---|---|---|
| Winner on C | MXFP6 E2M3 | T1 |
| Held-out Y error, class rule | E2M3 3.101, BFP6 3.484, MXINT6 4.054, E3M2 5.469, E4M1 11.149 | T1 |
| Held-out Y error, opt | E2M3 3.076, BFP6 3.437, **MXINT6 3.880** (changed with D10), E3M2 5.257, E4M1 10.237 | T1 |
| BFP6 definition | 16\|4\|5, floor, RNE, exp [−8, 7], sign-magnitude | DECISIONS D1 |
| MXINT6 definition | two's complement k ∈ [−32, 31] | DECISIONS D10 (revised) |
| BFP 32\|8\|5 equals | **MXINT6_sym**, not MXINT6 | test_formats.py l.300 |
| Storage | 6.32 (MX) / 6.28 (BFP6) by D11; 8.00 / 7.00 if padding is stored (Lecture 19 notebook's accounting) | T2, CONVENTIONS_AND_STORAGE.md |
| Saturated / zeroed (ocp) | E2M3 20 / 461, MXINT6 6 / 891, BFP6 10 / 748, E3M2 43 / 66, E4M1 96 / 1 | T6 |
| E4M1 conventions | 384: 11.149, 192: 11.149, one-NaN 256: 20.662 (ocp); all 10.237 under ceil | T3 |
| W+X | E2M3 4.244, BFP6 4.558, **MXINT6 5.213** (changed with D10; was 5.218) (ocp, held-out) | T5 |
| Selector accuracy | as in `results/selector_bench.md` (5 datasets, 28 adversarial, 1000 random) | selector_bench.md |
| Conventions | every [CLASS] row checked against the Lecture 19 notebook | CONVENTIONS_AND_STORAGE.md |
| Independent verification | 217/217 bit-exact | AGREEMENT_REPORT.md, this file |

When Aryan's final report exists, send it to Siddhant: every number in it is checked against this table and the regenerated `results/`.
