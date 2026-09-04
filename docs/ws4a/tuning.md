# Hyper-parameter tuning, and what it costs

*Code: `scripts/ws4a/tuning.py`, `scripts/ws4a_compare.py`, `configs/ws4a_tuned.yaml`.
Commands: [runbook §11](ws4a-runbook.md#11-the-tuned-run-optuna).*

The WS4A baseline evaluates every model on a **fixed, pre-declared grid**. This page
covers the optional second run that replaces the grid with an Optuna search, why it is
a *second* run rather than a replacement, and the three rules that keep it honest.

## The problem tuning creates

At WS4A's sample size — 63–70 labelled compounds per toxicity target, of which the
minority class is 16–33 — searching harder does not only find better models. It finds
better *fits to the validation noise*, and the two are indistinguishable from the
score alone.

The size of that effect is not a caveat, it is a measurement. Boulesteix & Strobl
selected the best of 124 classifier variants by cross-validation on **permuted**
high-dimensional data — data containing no signal by construction — and obtained
median error rates of **31–41 %** against a 50 % chance baseline. The bias grows as
*n* shrinks. WS4A sits well inside that regime.

So an untuned score and a tuned score are **not comparable**. The tuned one is higher
partly because the model is better and partly because the search is better at
exploiting noise, and nothing in the number says which.

## Three rules

### 1. The search never sees the data it is scored on

Optuna runs strictly inside the inner CV fold:

```text
for each OUTER fold:
    study = optuna(...)          <- sees the TRAINING part only
    best  = refit on that training part with the study's best parameters
    score = best.score(HELD-OUT part)     <- never influenced the search
```

This is what keeps the *outer* estimate unbiased at any budget. It is asserted, not
assumed: pre-flight check 19 tags every row with an id, replaces `tune_fit` with a
spy, runs `evaluate()` for real, and requires that no search ever saw all rows and
that every row was withheld from at least one search. A refactor passing `X` instead
of `X[tr]` fails it immediately.

### 2. The permuted control gets the identical budget

This one is enforced in code, not offered as a config option:

```python
r  = evaluate(X, y, ..., n_trials=n_trials)                  # real labels
rp = evaluate(X, y, ..., permuted=True, n_trials=n_trials)   # SAME budget
```

Tune the real labels with 40 trials and the control with a 5-point grid and the
reported `gap_vs_permuted` is inflated by exactly the bias being measured. Making
this configurable would let a later edit manufacture a gap quietly, so it is not.

### 3. The budget is small, and its price is measurable

`ml.tuning.n_trials: 40`. Enough for TPE to move off the defaults on a 1–2 parameter
space and to explore XGBoost's 8-parameter space coarsely. Deliberately not 500.

Before raising it, measure what it costs on this data:

```bash
bash scripts/ws4a.sh ml --config /work/configs/ws4a_tuned.yaml \
     --cell-line hepg2 --target toxicity --models xgboost --bias-sweep 0,10,40,150
```

`--bias-sweep` runs **permuted labels only** at each budget, averaged over
`bias_sweep_repeats` independent permutations. Those labels contain no signal, so
every point above chance is bias and nothing else — the curve is the price list.

The measured curves, and the two opposite failure modes they reveal, are in
[the bias sweep section](#the-measured-bias-sweep) below.

## What the tuned run fixed, and what it did not

### The degenerate baseline rows

The baseline's XGBoost reported **balanced accuracy 0.500 ± 0.000** on several
target × block combinations. That is not a null result; it is a broken configuration.
`min_child_weight: 5` against 14–16 positives satisfied no split at all — measured:
**2 of 200 trees grew**, the model returned the majority class in every fold, and
0.500 followed by construction. At `min_child_weight: 1`, all 200 trees split.

The pipeline now flags this rather than printing it as a number: `evaluate()` marks a
row `degenerate` when the score is exactly chance with zero variance across folds, and
logs a warning naming it a broken configuration.

The sweep confirms the mechanism from the other side. On `tox_liver_toxicity` at zero
trials the **permuted** score is also exactly 0.500 ± 0.000 — the splits are blocked
regardless of what the labels say, which is what a label-independent failure looks
like.

### Tuning removed every degenerate row — 10 to 0

*This section previously reported the opposite, from a `--quick` comparison. The full
run supersedes it; the correction is kept visible rather than edited away.*

| run | degenerate rows | which |
|---|---|---|
| baseline (fixed grid) | **10** | every one XGBoost: `moa-fine` × all 5 blocks, `tox_liver_toxicity` × all 5 blocks |
| tuned (20 trials) | **0** | — |

All ten sit on the two targets with the smallest minority class — `moa-fine` with 14
positives and `tox_liver_toxicity` with 16. At `min_child_weight = 5` no split
satisfied the constraint on those, so the model returned the majority class in every
fold, on every block. The tuned run searches `min_child_weight` from 1 and produced
none.

**What the `--quick` run got wrong, and why it is worth knowing.** The preliminary
comparison used 2 blocks, 2 models and a single CV repeat, and reported 4 degenerate
rows tuned against 2 untuned — the opposite conclusion. With one repeat instead of
five, a fold-level accident is enough to make a model look constant. A smoke test is
for finding crashes, not for measuring effects; `--quick` says so in its own log
line, and this is what ignoring that costs.

The mechanism described there — that a search free to choose the penalty can find
that zeroing every coefficient optimises the inner folds — remains a real failure
mode at 636 features and 64 rows. It simply did not happen in the full run.

### An absence of signal is still an absence of signal

Tuned XGBoost on `ecfp` × `tox_cardiotoxicity` scores 0.429 real against 0.470
permuted — an honest gap of **−0.041**. The degenerate 0.500 was hiding an absence of
signal, not a result. Removing the degeneracy makes the absence legible; it does not
turn it into a finding.

## The measured bias sweep

`--bias-sweep 0,10,40`, xgboost, HepG2, **permuted labels only**, 3 repeats per point
(the default is now 10 — see below). Every number is `permuted − chance`, so every
non-zero entry is bias:

| target | block | 0 trials | 10 | 40 | shape |
|---|---|---|---|---|---|
| tox_cardiotoxicity | ecfp | −0.009 | +0.053 | +0.011 | noise |
| tox_cardiotoxicity | morphology | +0.002 | +0.038 | +0.035 | search bias |
| tox_pulmonary_toxicity | ecfp | **+0.069** | +0.061 | +0.055 | **grid bias, flat** |
| tox_pulmonary_toxicity | morphology | **+0.085** | −0.012 | +0.008 | **grid bias, removed** |
| tox_renal_toxicity | ecfp | −0.008 | +0.022 | **+0.048** | **monotone search bias** |
| tox_renal_toxicity | morphology | −0.042 | −0.035 | −0.047 | below chance throughout |
| tox_infertility | ecfp | +0.003 | +0.009 | +0.000 | none |
| tox_infertility | morphology | +0.007 | −0.013 | −0.015 | none |
| tox_liver_toxicity | ecfp | +0.000 | +0.008 | — | degenerate at 0 |

**There is no single bias number for this pipeline.** The curve is target-dependent,
and the two failure modes point in opposite directions:

- **Grid bias.** `tox_pulmonary_toxicity` × morphology scores **0.585 on labels with
  no signal in them** using the *fixed grid*, and falls to chance once tuned. The
  baseline's −0.085 gap on that cell was a bad grid point, not anti-predictive
  morphology. Tuning *removed* bias here.
- **Search bias.** `tox_renal_toxicity` × ecfp is the textbook curve: −0.008 → +0.022
  → +0.048, monotone in the budget. Tuning *added* bias here.

An untuned baseline is therefore not automatically the conservative choice. A fixed
grid is just an unexamined search with a budget of one, and its bias is unmeasured
unless you sweep it — which is why the sweep runs the 0-trial point at all.

### The sweep validates itself against the main run

At `n_trials=0` the sweep re-derives the permuted score the main ML stage computed
through a different code path. They agree:

| cell | main run permuted | sweep at 0 trials |
|---|---|---|
| tox_pulmonary_toxicity × ecfp | 0.562 | 0.569 |
| tox_pulmonary_toxicity × morphology | 0.584 | 0.585 |

### Use enough repeats to read the curve

The first sweep used three, and produced `tox_cardiotoxicity` × ecfp at −0.009 →
+0.053 → +0.011: points that look like a curve and are not one. A single permuted run
of xgboost there has a fold-to-fold spread of ±0.075, so three repeats carry a
standard error of 0.043 against an effect of the same size. The default is now **10**
(standard error 0.024). Do not read a shape into a 3-repeat sweep.

## What it bought — the full run

6 targets × 5 blocks × 4 models, 25 folds, shuffled control at the identical budget.
Full detail on [the results page](ws4a-results.md#tuning-did-the-search-buy-signal-or-bias).

| | |
|---|---|
| median change in score on **shuffled** labels | **−0.0003** |
| median change in the honest gap | +0.0055 |
| verdicts | 31 improved, 62 unchanged, 27 degraded, **0 `bias`** |
| gaps clearing zero | 56 → **56** |
| degenerate rows | **10 → 0** |

**The budget of 20 is honest, and that is measured rather than argued.** The
shuffled-label scores did not move and not one row of 120 earned the `bias` verdict.
Quote this when asked why the budget is not 500.

**Tuning produced no new findings.** The number of gaps clearing zero is identical
before and after. It moved numbers and it repaired ten broken models; it did not turn
a single null into a result. That is the honest summary, and it is a better outcome
than a tuning step that appears to create findings.

## Reading the comparison

`ws4a_compare.py` reads two finished runs and refits nothing, so it works on a laptop
against results copied off the cluster.

| verdict | meaning |
|---|---|
| `improved` | the honest gap grew |
| `bias` | real score grew, but the permuted control grew as much or more |
| `degraded` | the gap shrank |
| `unchanged` | \|Δ gap\| < `--tol` (default 0.02, about one fold of noise here) |

Three figures, each chosen to make a specific failure visible:

- **`compare_gap_slopes_<cl>.png`** — one line per target × block × model, baseline gap
  to tuned gap. Left of the dashed line the model is worse than its own control.
- **`compare_real_vs_permuted_<cl>.png`** — the same runs plotted twice, real labels
  and permuted. Points above the diagonal on the **right** panel are the warning: those
  labels carry no signal, so any rise there was manufactured.
- **`compare_per_model_<cl>.png`** — Δ gap per model, median marked. Answers "did
  tuning help *this* model" rather than "did tuning help".

## Why a separate config, and why it is an overlay

`configs/ws4a_tuned.yaml` sets three things: a different `paths.outputs`,
`ml.tuning.n_trials`, and `bias_sweep_repeats`. Everything else — targets, guards,
models, feature blocks, data paths — is inherited through `extends: ws4a.yaml`.

A copy would have been simpler to write and wrong within a week: the first edit to a
target list that lands in only one of the two files makes the "before/after" a
comparison of two different experiments. The loader merges dicts key by key and
replaces lists whole (`models: [xgboost]` means only xgboost, not five models), and
pre-flight check 22 asserts that the overlay still inherits the base target list and
that the two runs' output directories differ.

## Repeats were not applied until 2026-09-03

`ml.cv.repeats: 5` has been in the config, and in the "nested, repeated" description
on the toolchain page, from the first commit. It was never used: `RepeatedStratifiedKFold`
was imported and never called, so every table produced before 2026-09-03 — the
`--quick` comparison above, job 1349676 — is a **single 5-fold**, with the fold-to-fold
spread that implies (the ±0.075 on a permuted xgboost run is that spread). Tables now
carry `stratified_5fold_x5` in `cv_scheme` when repeats are on; `--quick` runs one
repeat and says `stratified_5fold`. Do not compare a `_x5` table against a plain one
on `score_std`.

## Cost

The search multiplies fits per inner fold by the trial budget — the grid runs 1–5
candidates, Optuna runs 40. **Measured** on HepG2, same 20 target × block × model
combinations, 12 workers: untuned under 2 minutes, tuned at 40 trials ~30 minutes.
That is roughly **15×**, not the 8–10× a naive read of the trial counts suggests,
because the grid's candidates are cheap ones. Stats and xai are untouched.

Ask for the same cores and more wall time. `--quick` caps the budget at 5 trials for
exactly this reason: at the full budget a "quick" smoke test runs slower than the
complete untuned pipeline.
