Selector math prototypes (Aryan, 23 Oct). Run from the repo root; NPZ = assignment_data.npz from the course zip.
Written against `formats.py` v1 (main on 23 Oct), so BFP6 is left out (v1's BFP6 was numerically MXINT6) and MXINT6 is symmetric. `models.py` holds the shared models. The maintained versions of these methods, on the v2 formats, are in `selector.py` and `bench_selector.py` at the repo root (see `docs/SELECTOR.md`).

- derive.py NPZ                          worked one-block example + step-by-step rho / phi numbers on W_C
- adversarial.py NPZ                     26 adversarial families: true NSR vs per-element model (M1) vs few-statistics formula
- adv2.py NPZ                            continuous outliers, X-weighted (H_jj) model on A-E, MXINT at 4/6/8 bits
- artifact.py NPZ                        shows Nikita's E3M2 outlier wins vanish when outlier magnitudes are jittered
- chan.py NPZ                            outlier channels (whole columns scaled)
- random_search_cascade.py NPZ           1000 random matrices: top-1 / regret of 2-statistic formula, per-element model, hybrid (exact top binade), cascade
- rho_closed_form.py                     first version of the block-peak statistic (needs assignment_data.npz in cwd)
