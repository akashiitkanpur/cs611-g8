# Independent verification of `formats.py` v2 (Siddhant), 23 Oct

Tested: Akash's `formats_v2_changed_files.zip` (files dated 22 Oct 08:50) laid over GitHub commit `6290ea0`.
**v2 is not on GitHub yet**: once it is pushed, rerun the commands below on the pushed commit and put its hash here.

```
python verification/test_reference_impl.py     # 16 golden tests on the reference itself
python verification/crosscheck.py --selftest   # harness detects a single-ulp difference
python verification/crosscheck.py --impl akash_adapter:quantize --data data/W_C.npy --x data/X_cal.npy
python verification/verify_C.py                # reference-only numbers on C
```
Logs: `crosscheck_log.txt`, `crosscheck_report.json`, `verify_C_log.txt`, `clean_run_log.txt`.

## 1. Summary

| Check | Result |
|---|---|
| Bit-exact cross-check, v2 `formats.py` vs reference: 11 formats/variants × policies × 7 inputs (adversarial, 5 random families, W_C) | **217 / 217 bit-exact** |
| `log2` exponent bug from v1 | **fixed** (all adversarial `ocp` cases now agree; `test_one_ulp_below_power_of_two` added) |
| Every number in `T1`–`T6` and the C rows of `T8` | **confirmed** by the reference (same values; T1 table below) |
| Clean run: fresh venv, `pip install numpy matplotlib tqdm`, `python run_all.py --fresh` | **exit 0, 8 s, 30/30 tests**; all tables and both baseline CSVs identical to the zip (64 rows, every cell except wall-clock `seconds`), with numpy 2.5.4 instead of the zip's 2.5.3 |
| Hand example in `docs/QUANTIZERS.md` (block [6, 3.3, −1.06, 0.3, 0.07, 0], five formats) | correct for all five |

Formats covered: MXFP6 E3M2 / E2M3 / E4M1, E4M1_v192, E4M1_nan1, MXINT6, MXINT6_tc, BFP6 (16|4|5), BFP6_g32e8, BFP6_b32_ceil, BFP6_row; policies ocp, ceil, opt.

New adversarial rows (added for v2): BFP groups whose exponent hits both ends of the 4-bit field (2⁻¹⁰ → clamped to −8; 1000 → clamped to 7 and saturated to 248; exactly 2⁻⁸; just below 256), and negative block maxima that make two's-complement MXINT6 use code −32 (used 5 times; symmetric MXINT6 never does).

## 2. Dataset C (reference vs v2, Y error on all of X_cal, %)

| Format | ocp | ceil | opt |
|---|---|---|---|
| MXFP6 E2M3 | 3.079 | 3.103 | 3.058 |
| BFP6 16\|4\|5 | 3.464 | 3.574 | 3.403 |
| MXINT6 | 4.047 | 4.168 | 3.896 |
| MXFP6 E3M2 | 5.439 | 5.236 | 5.236 |
| MXFP6 E4M1 | 11.088 | 10.176 | 10.176 |
| E4M1 one-NaN (256) | 20.722 | 10.176 | 10.176 |
| MXINT6 two's complement | 4.048 | 4.168 | 3.850 |

Held-out (odd columns, D6), class rule: E2M3 3.101 < BFP6 3.484 < MXINT6 4.054 < E3M2 5.469 < E4M1 11.149, the same as `T1`.
Counts under ocp (`T6`): saturated after rounding E2M3 20, MXINT6 8, BFP6 10, E3M2 43, E4M1 96; zeroed 461 / 891 / 748 / 66 / 1: all equal to the reference.

## 3. Open points found during the review (none is a bit error)

1. **Push v2.** `RUN_MANIFEST.md` in the zip says "6290ea0 (with uncommitted changes)", so no commit hash identifies the code that made the numbers. After pushing, rerun `run_all.py --fresh` so the manifest names a clean commit.
2. **Datasets A–E are not reproducible from the repo.** `T7` and the A/B/D/E rows of `T8` need `assignment_data.npz`, which is in `.gitignore`. The clean run skips them (and says so). Either the README says where the file comes from (course zip) or the report marks T7/T8 A–E as "requires the course data file".
3. **Exact tie on dataset B.** `T7`: MXINT6 and BFP6 are both 1.781% with "margin 0.000", and the winner label comes from sort order. D6 says a tie must be called a tie; `T7` should print "tie" when the margin rounds to 0 (relevant to Aryan and Nikita).
4. **Manifest details.** It lists its own sha1 among the outputs (a file cannot contain its own hash), and in a non-git copy (e.g. the submission zip) "git commit:" is empty; print "not a git checkout" instead.
5. **Figures** differ byte-for-byte from the zip (PNG metadata / renderer), the numbers behind them do not. Fine; do not compare figures by hash.
6. **README.** Pin the tested versions (Python 3.13.16, numpy 2.5.3 and 2.5.4 both give identical outputs, matplotlib 3.11.2) in a `requirements.txt`; use `python` or `python3` consistently; add the `verification/` commands.
7. **Predictor (Aryan).** `validate_predictor.py` now reports 74/90 and BFP6 on C predicted 1.56 dB too low (27.65 vs 29.21 dB): the predictor still models the v1 BFP6. Already flagged by Akash.
8. **Two conventions sections.** `docs/QUANTIZERS.md` §B and `verification/CONVENTIONS_AND_STORAGE.md` overlap. Suggest one table in the report: Akash's §B plus the rows only mine has: `opt` tie order (0, −1, +1, −2), BFP exponent stored as 4-bit two's complement, the padded-storage alternative (8.00 / 7.00 bits/el) and the overhead section.

## 4. History
- v1 (21 Oct, `a185098`): 85/90 bit-exact; 5 failures, all one bug (`np.floor(np.log2(amax))` in the `ocp` exponent). Fixed in v2.
- Independence: the reference was written from the plan conventions. `formats.py` v0 was read once on 21 Oct to learn its API; since then `formats.py` is only run as a black box. The v2 change list was taken from the docs, tests, diffs of the other files and the API, not from reading `formats.py`.
