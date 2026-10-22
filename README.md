# CS611 Assignment 1, Group 8 (Dataset C, 6-bit)

Picks the best 6-bit format (MXFP6 E3M2 / E2M3 / E4M1, MXINT6, BFP6) for W_C,
using a histogram-based predictor instead of brute-force quantization.

## Requirements
Python 3.8+ and numpy (`pip install numpy`). Nothing else.

## Data
Put `W_C.npy` and `X_cal.npy` in `data/` (already included), or pass
`--data <folder>` to any script.

## Run everything (one command)
    python3 run_all.py              # tests, baselines, all tables, figures, manifest (~5 s)
    python3 run_all.py --npz path/assignment_data.npz   # + datasets A-E context tables

Outputs: `results/tables/T1..T8.md/.csv`, `results/baseline_C*.csv/.json`, `figures/`,
`results/logs/`, `results/RUN_MANIFEST.md`. Steps are checkpointed (`--fresh` recomputes).

## Run one piece
    python3 test_formats.py         # unit tests for the quantizers (30 tests)
    python3 run_baseline.py         # exhaustive baseline -> results/baseline_C.csv/.json (--variants, --mode wx)
    python3 validate_predictor.py   # main result: predictor vs brute force, accuracy and cost
    python3 explore.py              # brute-force table, scale search, X_cal split, sensitivity
    python3 make_figures.py         # report figures into figures/ (needs: pip install matplotlib)

## Files
- `formats.py`: the five quantizers (+ 6 sensitivity variants), scale policies ocp/ceil/opt, W+X mode, metrics, exhaustive harness (API: `FORMATS_API.md`; how it works: `docs/QUANTIZERS.md`)
- `run_all.py`: regenerates every table and figure
- `test_formats.py`: unit tests; `run_baseline.py`: exhaustive ground truth for the selector
- `predictor.py`: the format-selection algorithm (histogram + XX^T metric)
- `validate_predictor.py`: checks the predictor on W_C and 90 synthetic matrices
- `explore.py`: exhaustive baseline and the outlier / sparsity / scale experiments
- `make_figures.py`: generates the four report figures
- `report/`: draft report (`report_draft.docx`) and its builder script
- `data.py`: data loading; `scipy_free_stats.py`: rank-correlation helper

Format conventions follow the class (Lecture 19 notebook, board notes) and the team
decisions D1-D11 (BFP6 = 16|4|5, E4M1 vmax 384, ...). They are listed with sources in
`docs/QUANTIZERS.md` and the docstring at the top of `formats.py`.

## Use the predictor on your own matrices
    from predictor import select
    best, scores = select(W, X)
