# WS4A — cross-modal integration: statistics + regularised ML

Morphology × gene expression × **chemical structure**, on the 119 matched HepG2
compounds in `DATA_MAIN/`. Config-driven, containerised, and every number reported
next to the same number computed on scrambled labels.

**Start here:** [for-dummies.md](for-dummies.md) if you have not trained a model;
[results.md](results.md) for what was found; [presentation.md](presentation.md) if you
are presenting it.

## The headline

| | honest gap (real − scrambled labels) |
|---|---|
| chemistry (ECFP) alone | +0.175 |
| **morphology alone** | **+0.267** |
| chemistry + morphology | **+0.318** |
| expression | +0.339 |

Morphology adds **+0.143 over chemical structure** for mechanism of action, with all
four models agreeing — and **nothing** for the five toxicity endpoints, where it is
neutral or harmful. Chemistry is the control because it is free.

## Run it

```bash
# 1. build the image once (~10 min).  The .sif is NOT in git — it is 767 MB.
bash scripts/build_ws4a_container.sh          # or copy container/ws4a.sif in

# 2. check the vendored algorithms still match their call sites (~25 s)
bash scripts/ws4a.sh selftest                 # expect 23/23

# 3. Tier 1 — matrix agreement, ~4 min
sbatch --time=00:30:00 slurm/ws4a_pipeline.sbatch --stage stats --cell-line hepg2

# 4. Tier 2 — the ML grid, as a 30-task array (~20 min), then merge
sbatch slurm/ws4a_ml_array.sbatch --cell-line hepg2
sbatch --dependency=afterany:<ID> slurm/ws4a_ml_merge.sbatch --cell-line hepg2

# tuned run: same, plus --tuned --n-trials 20, writes to Results/ws4a_tuned/
```

Edit the `#SBATCH --account` / `--partition` lines, or pass `-A` / `-p`.
Full command list: [runbook.md](runbook.md).

## Layout

```
scripts/ws4a/        common.py (config, loading, guards), tuning.py (Optuna),
                     and 5 VENDORED algorithms with no working py3.12 install:
                     ajive, matrix_agreement, stabsel, splsda, permcca
scripts/ws4a_*.py    ml · stats · xai · plots · compare · report · crossmodal ·
                     hvg_experiment · merge · selftest
configs/             ws4a.yaml, and ws4a_tuned.yaml (an OVERLAY via `extends:`)
container/           ws4a.def + pinned requirements — the RECIPE, not the image
slurm/               pipeline, ML array, merge
Results/ws4a*/       the merged tables behind every claim (CSV, small)
WS4A_presentation_pack/   figures, slides, and how to present them
```

## Five things that will bite you

1. **`rv_adj`, never plain RV.** Plain RV scores *random noise of this shape* as high
   as the real data — higher on two of three modality pairs.
2. **`gap_vs_permuted`, never `score_mean`.** n is 64–70. Best-of-many model selection
   on permuted high-dimensional data gives 31–41 % error against 50 % chance.
3. **Exactly 0.500 ± 0.000 is a broken model, not a null result** — it predicted one
   class every fold. Such rows are flagged `degenerate`.
4. **The A549 morphology in the delivered MuData is contaminated** — 44 of 615 features
   exceed |500|, max 1.5e19. `hygiene.on_contamination: abort` is the default. HepG2 is
   clean.
5. **`--mem` is rejected on Sofia.** Ask for cores; memory follows at ~1.9 GB/core.

## Where this came from

Developed in a separate prep workspace and ported here. `configs/ws4a.yaml` has
`root: .` because in *this* repository the stream folder is the repository root.
