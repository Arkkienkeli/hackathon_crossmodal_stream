# WS4A runbook — every command, in order

*From an empty cluster directory to figures. Written against the VUB/VSC cluster this
project actually uses, and against the toolchain as it stands on **2026-09-03**.
Commands marked **yours** carry the real paths from job `1349279`; change them for
any other machine.*

!!! abstract "The short version"
    ```bash
    # once, on the workstation
    bash scripts/build_ws4a_container.sh && bash scripts/ws4a.sh selftest

    # once, to the cluster
    rsync -avP --partial container/ws4a.sif <cluster>:$P/container/ws4a.sif

    # every time
    ssh <cluster> && cd $P && git pull origin main
    mkdir -p logs
    sbatch -A <account> -p zen5_dense slurm/ws4a_pipeline.sbatch
    ```

## 0. What you are running, and on what

Three stages, all CPU-bound. **Do not request a GPU** — see [§7](#7-why-no-gpu).

| Stage | Does | Cost at n=119 |
|---|---|---|
| `stats` | Mantel, RV, PROTEST, AJIVE, permutation CCA + destructive controls | minutes, once parallel |
| `ml` | elastic net, sparse PLS-DA, linear SVM, XGBoost, nested CV | the long pole (`saga`) |
| `plots` | integration figures | seconds |

!!! warning "Memory is not requestable on Sofia"
    It is preselected per node and allocated in proportion to `--cpus-per-task`;
    passing `--mem` is **rejected**. Cores are the only sizing knob, and asking for
    more of them is how you get more RAM. None of the job scripts in `slurm/` set
    `--mem` any more.

Target partition, from the cluster's own spec sheet:

| Partition | Nodes | CPUs/node | RAM | Use |
|---|---:|---|---|---|
| **`zen5_dense`** | 56 | **2× 192-core EPYC 9965 = 384** | 742 GB (~1.9 GB/core, **not requestable**) | **this pipeline** |
| `zen5_himem` | 16 | 2× 96-core EPYC 9655 | 1493 GB (7.7 GB/core) | if memory ever binds |
| `zen4_h200` | 22 | 2× 96-core EPYC 9654 + 8× H200 | 1493 GB | only if a cell-level tier is built |

## 1. Build the image (workstation, once)

```bash
cd /media/arka/ubuntu_only/Big_projects/scVesrseXcp      # yours
bash scripts/build_ws4a_container.sh                     # ~8 min -> 765 MB
bash scripts/ws4a.sh test                                # image + vendored modules
bash scripts/ws4a.sh selftest                            # 18 API-contract checks, ~15 s
```

Building needs root (or fakeroot); **running never does**. That is why the image is
built here and shipped.

!!! tip "Run `selftest` before every submission"
    It calls every vendored API exactly as the pipeline does and asserts the shape of
    what returns. Three contract mismatches reached real runs before it existed — the
    last one killed job `1349279` forty minutes in, *after* the expensive part had
    finished. It now also runs as stage 0 of the sbatch and aborts the job in seconds
    if anything has drifted.

## 2. Ship it (workstation → cluster)

```bash
export P=/sofia/projects/2026_084/Projects/scVesrseXcp    # yours

rsync -avP --partial --inplace container/ws4a.sif <cluster>:$P/container/ws4a.sif
sha256sum container/ws4a.sif       # compare both ends
```

No `-z`: a `.sif` is already compressed. `--partial --inplace` resumes a broken
transfer.

The code comes from git, not rsync:

```bash
ssh <cluster>
cd $P && git pull origin main
```

**Data** must already be under the layout `configs/ws4a.yaml` expects. On your
cluster `root:` resolves to the repo, so:

```text
$P/
├── scripts/ container/ configs/ slurm/     <- git
├── DATA_MAIN/
│   ├── MuData (morph, exp, chemical)-.../  <- a549_mdata.h5mu, hepg2_mdata.h5mu
│   └── Tahoe_singl_cell/                   <- only for a future cell-level tier
├── OpenScreen/data/                        <- the 3 per-site h5ad files
└── Results/ws4a/                           <- outputs land here
```

Data elsewhere? Bind it instead of moving it:

```bash
sbatch --export=ALL,WS4A_DATA=/scratch/$USER/stream slurm/ws4a_pipeline.sbatch
```

## 3. Verify what arrived (cluster)

```bash
cd $P
ls -la container/ws4a.sif                       # ~765 MB
sha256sum container/ws4a.sif                    # must match the workstation
git log --oneline -1                            # 4024f26 or later
module load apptainer 2>/dev/null || command -v apptainer

bash scripts/ws4a.sh test
bash scripts/ws4a.sh selftest                   # expect 18/18
```

If `apptainer` is not on PATH on the **compute** node, uncomment the `module load`
line near the top of `slurm/ws4a_pipeline.sbatch`.

## 4. Set your account and partition

Edit `slurm/ws4a_pipeline.sbatch` once, or pass them per submission. Discover them:

```bash
sacctmgr -n show assoc user=$USER format=account%20 | sort -u
sinfo -o "%20P %10G %8c %12m %a" | head
```

Then either uncomment in the file:

```bash
#SBATCH --account=2026_084
#SBATCH --partition=zen5_dense
```

or override at submit time, which wins over the file:

```bash
sbatch -A 2026_084 -p zen5_dense slurm/ws4a_pipeline.sbatch
```

## 5. Submit

!!! tip "There is a faster way to run the ML stage"
    The commands in this section run the ML stage as **one job** — ~5 h untuned,
    ~22 h tuned at 20 trials. [§12](#12-fast-path-the-ml-stage-as-an-array-across-nodes)
    runs the same work as a 30-task array across nodes in **~5 / ~20 minutes** and is
    the way to go whenever more than one node is available. Use this section for
    `stats`, `xai`, `plots` and `compare`, which are cheap.

```bash
mkdir -p logs        # #SBATCH --output writes here; Slurm rejects the job without it
```

=== "Everything (default: 96 cores, 4 h)"

    ```bash
    sbatch -A 2026_084 -p zen5_dense slurm/ws4a_pipeline.sbatch
    ```

    Runs pre-flight → `stats` → `ml --target all` → `plots`, in that order, because
    the figures read what the first two wrote.

=== "One stage"

    ```bash
    sbatch -A 2026_084 -p zen5_dense slurm/ws4a_pipeline.sbatch --stage stats
    sbatch -A 2026_084 -p zen5_dense slurm/ws4a_pipeline.sbatch --stage ml --target toxicity
    sbatch -A 2026_084 -p zen5_dense slurm/ws4a_pipeline.sbatch --stage plots
    ```

=== "Whole node, for the fastest run"

    ```bash
    sbatch -A 2026_084 -p zen5_dense \
        --cpus-per-task=384 \
        slurm/ws4a_pipeline.sbatch
    ```

    **No `--mem`.** On Sofia memory is not requestable — it is preselected per node
    and allocated in proportion to the cores, and passing `--mem` is rejected.
    Cores are the only sizing knob: 96 implies ~180 GB, 384 implies ~740 GB. This
    pipeline needs ~15 GB regardless, so ask for cores to go **faster**, never to
    get memory.

=== "Chained, so ml waits for stats"

    ```bash
    JID=$(sbatch --parsable -A 2026_084 -p zen5_dense \
              slurm/ws4a_pipeline.sbatch --stage stats)
    sbatch --dependency=afterok:$JID -A 2026_084 -p zen5_dense \
              slurm/ws4a_pipeline.sbatch --stage ml --target all
    ```

Everything after the script name is passed to the stage, so `--cell-line a549`,
`--permutations`, `--n-jobs`, `--quick` all work.

## 6. Watch it

```bash
squeue -u $USER
tail -f logs/ws4a-<jobid>.out
```

A healthy log, in order:

```text
job / repo / image / stage / cpus / args
gpu        : none — XGBoost runs on CPU (correct, just slower)
>>> pre-flight (vendored API contracts)          18/18 passed
>>> stats
hygiene    : hepg2/morphology clean (636 features, range -44.42 .. 145.4)
loaded     : hepg2  n=119  morphology (119, 636) | expression (119, 41780) | ecfp (119, 1024)
parallel   : 96 worker(s) x 1 BLAS thread(s)  (allocated 96 cpu)
CAP permutations             = 9999
parallel   : controls morphology~expression   40 task(s) in ...s on 40 worker(s)
ajive      : joint_rank=4 individual=[17, 16]
```

Two lines to check specifically:

- **`parallel : N worker(s)`** — if N is 1, the cores are idle; see [§8](#8-troubleshooting).
- **`hygiene : ... clean`** — `a549` will instead **refuse**, by design ([§8](#8-troubleshooting)).

## 7. Why no GPU

Benchmarked on the real shapes, not assumed:

| | CPU | GPU |
|---|---:|---:|
| morphology 119 × 636 | 0.16 s | 0.14 s |
| expression 119 × 41,780 | 9.31 s | **9.35 s** |
| 50,000 × 636 (hypothetical) | 4.07 s | 3.69 s |

XGBoost is the **only** GPU consumer. Mantel, RV, PROTEST, AJIVE, permutation CCA,
stability selection and SHAP have no GPU code path at all. At ~100 rows there is
nothing for a GPU to parallelise, so an H200 would idle through a CPU-bound run.

If a single-cell tier is ever built, 542,089 cells is where it would pay:

```bash
sbatch -p zen4_h200 --gres=gpu:1 slurm/ws4a_pipeline.sbatch
```

The job detects the GPU, passes `--nv`, and XGBoost uses it. Nothing else changes.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `a549/morphology: 44 of 615 features exceed \|500\|` | **Working as designed.** The delivered A549 MuData carries [the `drop_outliers` defect](finding.md), max 1.5e19 | `hygiene.on_contamination: clean` in the config to drop them and say so, or use `--cell-line hepg2`, which is clean |
| `PRE-FLIGHT FAILED` | a call site disagrees with a vendored module | run `bash scripts/ws4a.sh selftest` and read which check failed; do not submit until 23/23 |
| `FitFailedWarning: 1 fits failed out of 12` + `splsda.py: X-score collapsed to zero -- keep_x=10` | **Not fatal, and not a wrong number.** `sparse_plsda`'s sparsest grid point runs out of rank on the *second* component; GridSearchCV scores it `nan` and picks the best of the rest. See [§8.1](#81-sparse_plsda-x-score-collapsed-to-zero) | let the job run. Nothing else is affected |
| `SKIP tox_dermatological_toxicity` | **Working as designed.** 68/2 balance — a classifier scores 0.97 predicting one class | nothing; the reason is logged and written to `skipped_targets_*.csv` |
| `SKIP depmap_auc` on HepG2 | n=22 labelled | use `--cell-line a549` (n=53) once its morphology is cleaned |
| `parallel : 1 worker(s)` | `SLURM_CPUS_PER_TASK` unset or 1 | pass `--cpus-per-task`; or force with `--n-jobs 96` |
| Load average in the thousands, slower than serial | BLAS oversubscription | the sbatch exports `OMP_NUM_THREADS=1`; if running by hand, do the same |
| `apptainer: command not found` | not on PATH on the compute node | uncomment `module load apptainer` in the sbatch |
| `Slurm rejects the job, unwritable output` | `logs/` missing | `mkdir -p logs` |
| Job rejected for `--mem` | Sofia preselects memory per node; it is not requestable | drop `--mem` entirely and raise `--cpus-per-task` instead — RAM follows cores at ~1.9 GB/core |
| `RuntimeError: cannot cache function ... no locator available` | read-only image, unwritable cache | the sbatch sets `TMPDIR` to node-local SSD; check it exists |

### 8.1 `sparse_plsda`: "X-score collapsed to zero"

This appears in the array-task logs, often repeated, and looks alarming. It is a
warning, the task exits 0, and the merge accepts its part.

```text
FitFailedWarning: 1 fits failed out of a total of 12.
  RuntimeError: component 1: X-score collapsed to zero -- keep_x=10 selected
  only null-variance variables
UserWarning: One or more of the test scores are non-finite: [nan 0.769 0.866 0.810]
```

**What actually happens.** `sparse_plsda` is fitted over a grid of
`keep_x = [10, 25, 50, 100]` — how many variables each component may use. At
`keep_x=10` one inner fold raises, so GridSearchCV records `nan` for that grid point
and selects the best of the three that scored (0.866 above). The outer fold's score
comes from that refit model. **The reported score and honest gap are valid**; the
only loss is that the sparsest grid point is unavailable in that fold.

**What the message gets wrong.** `component 1` is the *second* component (0-indexed),
and the cause is not null-variance variables being wrongly selected. A constant
column is standardised to all-zeros, so its weight is exactly 0 and it is never
picked while any informative column remains. What happens is that the first
component's deflation (`Xh -= t cᵀ`) removes the variance from the 10 columns the
sparsity allows, leaving the second component nothing to extract: `tᵀt < 1e-12`.

That is sparse PLS legitimately running out of rank at high sparsity, not an error.
The correct response is to fit **one** component and say so, rather than to raise —
a rank-1 sparse PLS-DA is a valid model. **The vendored code still raises; the fix is
not applied.**

**Consequences worth knowing:**

- Only `sparse_plsda` is affected, and only its most-sparse settings. `elastic_net`,
  `linear_svm` and `xgboost` are untouched.
- **In a tuned run it matters more.** Optuna samples `keep_x` from 5 upward
  (log-uniform), so a larger share of trials hit this and are pruned. If every trial
  in a fold is pruned, `tune_fit` falls back to default settings and logs
  `every trial failed`. A `sparse_plsda` verdict of `unchanged` in the comparison can
  therefore be *this*, rather than a real null result. Check before reading anything
  into it:

  ```bash
  grep -l "every trial failed" logs/ws4a-ml-*.out
  ```

- If it is ever seen at `component 0`, that is different and genuine: fewer than
  `keep_x` columns in the block carry any signal at all.

## 9. What comes back

Under `Results/ws4a/` (whatever `paths.outputs` says):

```text
stats/    agreement_<cl>.csv        one row per pair per variant, controls included
          stats_<cl>.json           AJIVE joint rank + null, CCA, hygiene report
ml/       ml_<cl>.csv               every model, real and permuted
          ml_summary_<cl>.csv       with gap_vs_permuted -- the honest effect
          skipped_targets_<cl>.csv  what was guarded out, and why
xai/      stability_selection_*.png shap_*.png feature_grammar_*.png + CSVs
figures/  agreement_ embedding_ joint_structure_ ml_performance_ modality_overlap_
```

**How to read them**, briefly — the full reasoning is in
[the toolchain page](ws4a-toolchain.md):

- **`rv_adj`, not `rv_plain`.** Plain RV is 0.91 on independent blocks at this shape;
  adjusted RV is 0.013. The permutation p-value is trustworthy for both.
- **`gap_vs_permuted`, not `score_mean`.** A model scoring well on permuted labels is
  measuring selection bias, which is worst at this n.
- **Controls must collapse.** `ctl_scrambled` and `ctl_random` at `rv_adj` ≈ 0 with no
  significant p-values. If they do not, the pipeline is wrong, not the biology.

## 10. Iterating

```bash
# fast smoke test on a login/interactive node -- NOT for reporting
bash scripts/ws4a.sh stats --cell-line hepg2 --quick --no-cca
bash scripts/ws4a.sh ml    --cell-line hepg2 --target toxicity --quick

# one explanation, for one model
bash scripts/ws4a.sh xai --cell-line hepg2 \
    --target tox_cardiotoxicity --block morphology --model xgboost
```

`--quick` shrinks permutations, control replicates, the AJIVE null and the CV grid,
and says so in the log. Its numbers must never be reported.

Changed a config value and want it on the cluster? It is in git:

```bash
# workstation
vim configs/ws4a.yaml && git commit -am "config: ..." && git push
# cluster
cd $P && git pull origin main
```

## 11. The tuned run (Optuna)

!!! tip "There is a faster way to run the ML stage"
    The commands in this section run the ML stage as **one job** — ~5 h untuned,
    ~22 h tuned at 20 trials. [§12](#12-fast-path-the-ml-stage-as-an-array-across-nodes)
    runs the same work as a 30-task array across nodes in **~5 / ~20 minutes** and is
    the way to go whenever more than one node is available. Use this section for
    `stats`, `xai`, `plots` and `compare`, which are cheap.

The baseline above uses a **fixed, pre-declared grid**. The tuned run replaces it with
an Optuna TPE search that lives **entirely inside the inner CV fold**. Full reasoning,
and the reason the budget is deliberately small, is on
[the tuning page](ws4a-tuning.md).

Run it as a *second* run. It writes to `LINCS/data/ws4a_tuned/`, so it cannot
overwrite the baseline it exists to be compared against:

```bash
# 1. baseline, if you have not already run it
sbatch slurm/ws4a_pipeline.sbatch --cell-line hepg2

# 2. tuned -- one flag, and it picks up configs/ws4a_tuned.yaml
sbatch slurm/ws4a_pipeline.sbatch --tuned --cell-line hepg2

# 3. compare, once both are done
sbatch --export=ALL,WS4A_BASELINE_DIR=$P/Results/ws4a/ml,WS4A_TUNED_DIR=$P/Results/ws4a_tuned/ml \
       slurm/ws4a_pipeline.sbatch --stage compare --cell-line hepg2
```

`configs/ws4a_tuned.yaml` is an **overlay** — `extends: ws4a.yaml` — so targets,
models, guards and data paths are inherited, never copied. Edit them in one place.

### Budget the wall time from the untuned run's own timings

The search multiplies fits per inner fold by the trial budget, so the cost is set by
whichever cells were already slow. **Projected from job 1349676's measured seconds**
(HepG2, 5 blocks, 4 models, 6 usable targets, 128 cores):

| | per target | whole ML stage |
|---|---|---|
| untuned (fixed grid) | ~50 min | **~5 h** |
| tuned, 10 trials | ~1.8 h | **~11 h** |
| tuned, 20 trials | ~3.7 h | **~22 h** |
| tuned, 40 trials | ~7.3 h | **~44 h** |

**95 % of that is one cell type**: `elastic_net` on the two expression-containing
blocks, at 2854 s per target against 84 s for all three narrow blocks combined. saga
on 41,780 features is the whole bill. Everything else — every model on ecfp,
morphology and morphology+ecfp — is seconds.

Two consequences:

- **40 trials on the full block list is not affordable.** Use `--n-trials 20`
  (~22 h) or `--n-trials 10` (~11 h) and say which in the write-up; each row records
  its own `n_trials`, and `compare` reads it.
- **If your queue caps below that**, restrict the tuned run to the narrow blocks with
  `--models` / a `feature_blocks` overlay rather than lowering the budget further.
  Tuning all four models on ecfp + morphology + morphology+ecfp at 40 trials is
  ~1 h, and that is where the "does imaging beat chemistry" question lives.

`--n-trials`, `--models`, `--bias-sweep`, `--quick` and `--no-permuted-control` are
routed to the **ml stage only** by the sbatch; passing them to `stats` would abort the
job on an unrecognised argument before it produced anything.

### What the budget costs, measured rather than assumed

The tuned and untuned scores are not directly comparable — a bigger search raises the
score on data with **no signal at all**. To see the size of that effect on *your*
data, sweep the budget against permuted labels only:

```bash
bash scripts/ws4a.sh ml --config /work/configs/ws4a_tuned.yaml \
     --cell-line hepg2 --target toxicity --models xgboost \
     --bias-sweep 0,10,40,150
```

Every point above `chance` in `bias_sweep_hepg2.csv` was manufactured by the search.
If the curve at 40 trials is already well above chance, lower `ml.tuning.n_trials`.

### Reading the comparison

`compare_tuned_vs_untuned_<cl>.csv` carries a `verdict` per target × block × model:

| verdict | meaning |
|---|---|
| `improved` | the honest gap grew |
| `bias` | the real score grew but the **permuted control grew as much or more** — the search bought apparent performance, not signal |
| `degraded` | the gap shrank |
| `unchanged` | \|Δ gap\| < `--tol` (0.02, about one fold of noise here) |

Read `d_gap`. `d_score` rises on pure noise; only `d_gap` survives subtracting a
control that was given the **identical** search budget.

### Expect DEGENERATE warnings, and do not ignore them

```text
tox_liver_toxicity morphology xgboost  DEGENERATE: exactly chance with zero variance
  -- the model predicted one class every fold. This is a broken configuration, not a
  null result.
```

A row scoring exactly chance with zero variance across folds predicted one class
every time. It is flagged in the log and carries `degenerate=True` in
`ml_<cl>.csv`. **Never quote such a row as a null result** — it measured nothing.

On the full HepG2 run the untuned baseline produced **10** of these — all XGBoost, on
the two targets with the smallest minority class — and the tuned run produced
**none**, because it searches `min_child_weight` from 1 instead of fixing it at 5.
See [the results page](ws4a-results.md#tuning-did-the-search-buy-signal-or-bias).

## 12. Fast path: the ML stage as an array across nodes

The ML stage is embarrassingly parallel at two levels, and until 2026-09-03 it used
neither: the outer CV folds ran one after another, and one job ran every target. Two
changes, both measured:

- **Outer folds run on a worker pool.** Each fold is seeded by its index, so the
  numbers are bit-identical to the serial loop (pre-flight check 23 asserts it on the
  grid path and the Optuna path). Inner fits are serial on purpose; a nested pool
  would oversubscribe.
- **`cv.repeats` is now applied.** It was in the config from the start and never
  used — `RepeatedStratifiedKFold` was imported and never called, so every earlier
  run (job 1349676 included) was a single 5-fold. Now 25 outer splits, labelled
  `stratified_5fold_x5` in the tables. 5× less fold noise, and 5× more folds to
  run in parallel, so wall time per unit does not move.
- **One array task per (target column, feature block).** 30 units on HepG2, 32 cores
  each, 12 to a zen5_dense node. Two standalone sbatch files, same shape as the
  pipeline one: own `#SBATCH` header, plain `sbatch`, account and partition edited in
  the file or passed as `-A` / `-p`.

```bash
cd $P && git pull origin main
bash scripts/build_ws4a_container.sh        # optuna is new; or rsync the rebuilt .sif
bash scripts/ws4a.sh selftest                # 23/23

# stats, once (4 min, unaffected by tuning) -- the normal sbatch, ml skipped
sbatch --time=00:30:00 slurm/ws4a_pipeline.sbatch --stage stats --cell-line hepg2

# ML baseline: 30 tasks x 32 cores.  Note the job id it prints.
sbatch slurm/ws4a_ml_array.sbatch --cell-line hepg2
sbatch --dependency=afterany:<ID> slurm/ws4a_ml_merge.sbatch --cell-line hepg2

# ML tuned: same 30 tasks, Optuna at 20 trials, separate outputs
sbatch slurm/ws4a_ml_array.sbatch --tuned --n-trials 20 --cell-line hepg2
sbatch --dependency=afterany:<ID> slurm/ws4a_ml_merge.sbatch --tuned --cell-line hepg2

# when both merges have run
sbatch --export=ALL,WS4A_BASELINE_DIR=$P/Results/ws4a/ml,WS4A_TUNED_DIR=$P/Results/ws4a_tuned/ml \
       slurm/ws4a_pipeline.sbatch --stage compare --cell-line hepg2
sbatch --time=00:30:00 slurm/ws4a_pipeline.sbatch --stage plots --cell-line hepg2
```

Each task builds the same work list itself (`ml --list-units`, ~2 s, deterministic,
guards applied) and takes its own line of it — no wrapper, no shared state. The
`--array=0-29` in the header is HepG2's 30 usable units; another cell line needs
`--array=0-$((N-1))` on the command line, with N from a `--list-units` call.

**The merge refuses to write a table with units missing.** A failed task cannot
produce an `ml_summary` that looks complete; `logs/ws4a-merge-*.out` names the
missing units instead. Re-run one index and resubmit the merge:

```bash
sbatch --array=17 slurm/ws4a_ml_array.sbatch --tuned --n-trials 20 --cell-line hepg2
sbatch --dependency=afterany:<ID> slurm/ws4a_ml_merge.sbatch --tuned --cell-line hepg2
```

`--allow-missing` on the merge exists for a look at partial results and is not for
reporting. `afterany`, not `afterok`, on purpose: a failed task must not stop the
merge from telling you which units are missing.

### What to expect, per task

The expensive cell is unchanged: `elastic_net` on the 41,780-feature blocks, where
one saga fit is ~9 s and the 16 inner fits per fold are serial.

| unit type | untuned | tuned, 20 trials |
|---|---|---|
| ecfp / morphology / morphology+ecfp | ~1 min | ~5 min |
| expression / morphology+expression | ~5 min | ~20 min |

All 30 run at once if the queue gives ~3 nodes; the wall time is the slowest task,
about **20 minutes for the tuned run and 5 for the baseline** — against ~22 h and
~5 h in a single job. To test one task without a scheduler, set the index by hand:
`SLURM_ARRAY_TASK_ID=0 bash slurm/ws4a_ml_array.sbatch --tuned --n-trials 20 --quick`.
