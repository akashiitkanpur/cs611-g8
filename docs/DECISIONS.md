# Decision ledger

Append a row when a `decision` issue is resolved. Newest revisions win; do not delete old rows, mark them **superseded**.
Details and numbers are in [`TEAM_PLAN.md`](TEAM_PLAN.md) section 1.

| ID | Decision | Date | Status | Issue |
|---|---|---|---|---|
| D1 | BFP6 = `g\|e\|m` = 16\|4\|5, sign separate + 5-bit magnitude (k ∈ [−31, 31]), floor exponent, RNE, exponent clamp [−8, 7] | 21 Oct | locked | |
| D2 | Weights-only is primary; W+X is a secondary experiment | 21 Oct | locked | |
| D3 | E4M1: no reserved codes, vmax 384; `v192` and one-NaN (vmax 256) as sensitivity rows | 21 Oct | locked | |
| D4 | One format for the whole matrix | 21 Oct | locked | |
| D5 | Develop with optimized scale (`opt`) first; class floor rule (`ocp`) in every final table | 21 Oct | locked | |
| D6 | Primary metric: relative Frobenius error of Y on held-out X columns; tie rule via bootstrap interval | 21 Oct | proposed default | |
| D7 | Round-to-nearest-even everywhere | 21 Oct | locked | |
| D8 | Evaluate in FP64 (FP32 matmul agrees to ~1e-8) | 21 Oct | locked | |
| D9 | MX block 32 (class); BFP group size is free (default 16) | 21 Oct | locked | |
| D10 | MXINT6: two's complement k ∈ [−32, 31]; `MXINT6_sym` [−31, 31] as sensitivity variant; BFP6 stays sign-magnitude | 23 Oct | locked (supersedes symmetric) | |
| D11 | Storage per real bits/element incl. the tail block (MX32 6.32, BFP 16\|4\|5 6.28) | 21 Oct | locked | |
| D12 | Python + NumPy only, repo is source of truth, communication via GitHub Issues (`docs/CHANNEL.md`) | 23 Oct | locked | |
