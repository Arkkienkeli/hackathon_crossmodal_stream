# The WS4A toolchain

*Config-driven scripts for cross-modal integration on the `DATA_MAIN` release, in
their own container. Written and verified against the real data on **2026-09-03**.
Every measurement quoted here was produced by the code it describes.*

!!! abstract "What it is"
    Three tiers, matching [the integration plan](integration-plan.md): matrix-agreement
    statistics with permutation nulls, regularised supervised models with nested CV, and
    an explainability layer that carries each method's documented caveat onto the figure.

    Nothing hardcodes a path — every location and parameter lives in
    `configs/ws4a.yaml`.

## Layout

| Path | What |
|---|---|
| `configs/ws4a.yaml` | every path, parameter, cap and guard |
| `scripts/ws4a/` | five **vendored** algorithms (below) plus `common.py` |
| `scripts/ws4a_stats.py` | Tier 1 — Mantel, RV, PROTEST, AJIVE, permutation CCA |
| `scripts/ws4a_ml.py` | Tier 2 — elastic net, sparse PLS-DA, linear SVM, XGBoost |
| `scripts/ws4a_xai.py` | stability selection, SHAP, coefficients, feature grammar |
| `scripts/ws4a_plots.py` | integration figures |
| `scripts/ws4a.sh` | container entry point (the WS4A sibling of `ws4.sh`) |
| `container/ws4a.def` | **a separate image**, 60 packages, no deep-learning stack |
| `slurm/ws4a_pipeline.sbatch` | standalone job, own `#SBATCH` header, plain `sbatch` |

```bash
bash scripts/build_ws4a_container.sh          # once
bash scripts/ws4a.sh test                     # verify the image
bash scripts/ws4a.sh stats --cell-line hepg2
bash scripts/ws4a.sh ml    --cell-line hepg2 --target toxicity
bash scripts/ws4a.sh xai   --cell-line hepg2 --target tox_cardiotoxicity
bash scripts/ws4a.sh plots --cell-line hepg2
```

## Why five algorithms are vendored rather than installed

Each was chosen only after checking what actually installs on Python 3.12 with
numpy 2 and pandas 3 — **measured, not assumed**:

| Module | Why it is vendored |
|---|---|
| `ajive.py` | No pip-installable AJIVE works. `mvlearn` pins `matplotlib<=3.3.4`, which has no 3.12 wheel, so pip source-builds it and dies on `SafeConfigParser`; it runs only via `--no-deps`, which is a landmine in a container definition. `py_jive` installs but fails at import on modern scipy. The vendored numpy version was verified **numerically identical to mvlearn** — squared singular values agreeing to 1.6e-13 — and passes both destructive controls (joint rank 0 on noise, 0 on destroyed row correspondence). |
| `matrix_agreement.py` | No package offers a bias-adjusted RV with a permutation null. `scikit-bio` has Mantel but no RV or Procrustes; `hoggorm` has RV but no adjustment and no null. |
| `stabsel.py` | `stability-selection` is **not on PyPI at all**, was last committed in 2019, and imports `sklearn.externals.joblib`, removed in sklearn 0.23. |
| `splsda.py` | There is no maintained Python port of mixOmics. This implements its `keepX` semantics — sparsity as a **count**, which is tunable at n≈100 where a penalty has no interpretable scale. |
| `permcca.py` | Winkler's CCA permutation scheme has no packaged implementation. |

Vendoring keeps the container's dependency list to 60 packages and means no upstream
change can silently alter a published number.

## The three tiers

### Tier 1 — matrix agreement

The headline effect is the **adjusted RV coefficient** (Mayer et al. 2011), and that
choice is the single most important thing in the toolchain. Measured on independent
random blocks at our shape, where the true value is **0**:

| statistic | value when the truth is 0 |
|---|---:|
| plain RV | **0.913** |
| modified RV2 (Smilde et al. 2009) | 0.132 |
| **adjusted RV (Mayer et al. 2011)** | **+0.013** |
| raw Procrustes *r* | **0.976** |

Plain RV and raw Procrustes would each be reported as near-perfect agreement on data
containing none. Both are still written to the CSV — hiding them would make this claim
unverifiable — but neither is ever the headline.

**The permutation p-values are trustworthy even where the point estimate is not.** On
the same null blocks: plain RV = 0.913 with *p* = 0.173, correctly non-significant.

### Tier 2 — supervised models

At 63–70 labelled compounds the danger is not underfitting. The design follows from
that:

- the model list is **pre-declared** in the config and every model is always reported;
- CV is genuinely nested — scaling and every hyper-parameter fitted in the inner loop;
- a **permuted-label control** runs for every combination, and the reported effect is
  the *gap*, not the raw score;
- **ECFP-only is the QSAR control**: if chemical structure alone predicts the label as
  well as morphology does, morphology added nothing;
- where sites exist the outer loop is **leave-one-site-out** — OpenScreen has three
  independent sites measuring the same 118 compounds, which is far stronger than a CV
  fold at this n.

### Tier 3 — explainability

Every artefact carries its own caveat, printed on the figure:

| Method | Caveat carried |
|---|---|
| Stability selection *(primary)* | controls false positives, does **not** promise completeness — the bound assumes exchangeability and a β-min condition means small effects are missed |
| SHAP | **interventional** estimator against a background sample; falls back to path-dependent on models with categorical splits and **labels the figure when it does** |
| Coefficients | correlated features split coefficients arbitrarily; read alongside stability selection |
| Impurity / permutation importance | **not produced at all** — under 0.9 block correlation, zero-effect variables are selected as often as real ones |

`xai.forbid_impurity_importance: true` is enforced in code, not just documented.

!!! danger "A feature name is not a mechanism"
    Printed on every figure that names features: DNA-damaging drugs show **actin**
    features as most affected, because the cells detach and round up
    ([Carpenter-Singh Lab](https://carpenter-singh-lab.broadinstitute.org/blog/help-interpreting-image-based-profiles)).
    The toolchain therefore also aggregates attribution to **compartment / family /
    channel**, which is the level at which a morphology statement is defensible.

## Guards that refuse rather than produce a number

**Contamination.** The A549 morphology in the delivered MuData carries 44 of 615
features above |500|, max 1.5e19 — [the `drop_outliers` defect](finding.md), in the
organisers' release. The default policy is `abort`, and it fires:

```text
hepg2/morphology  clean (636 features, range -44.42 .. 145.4)
a549/morphology   REFUSED -- 44 of 615 features exceed |500| (max 1.516e+19);
                  worst: Nuclei_Neighbors_PercentTouching_2=1.52e+19
```

`hygiene.on_contamination: clean` drops them instead and says so. There is no silent path.

**Degenerate targets.** Measured label balance, per endpoint:

| endpoint | HepG2 pos/neg | verdict |
|---|---|---|
| cardiotoxicity | 37/33 | usable |
| pulmonary / renal / infertility | ~30–42 / ~23–41 | usable |
| liver | 54/16 | marginal |
| haematological | 62/8 | skipped |
| **dermatological** | **68/2** | **skipped** |

A classifier on a 68/2 target scores 0.97 by predicting one class. `target_guards`
skips it with a stated reason rather than reporting the number. `depmap_auc` is
n=53 on A549 but **n=22** on HepG2, also auto-skipped there.

**MoA is not multiclass-viable.** After dropping "unclear" (55/119 on HepG2), the
largest class has 14 members and the rest have ≤5. The config uses
`mode: one_vs_rest` on the largest annotated class rather than pretending 21 classes
are learnable.

## First measurements on real data

Tier 1, HepG2, n=119 (199 permutations — indicative, not final):

| pair | adjusted RV | Mantel *r* | Mantel *p* |
|---|---:|---:|---:|
| morphology ~ expression | +0.016 | **+0.202** | **0.005** |
| expression ~ ecfp | +0.035 | +0.094 | 0.050 |
| morphology ~ ecfp | +0.021 | +0.086 | 0.170 (n.s.) |

All destructive controls collapsed: `rv_adj` ≈ 0 and **0 %** of permutation p-values
below 0.05, for both `ctl_scrambled` and `ctl_random`.

**Morphology and expression do agree, and the effect is small** — Mantel *r* = 0.20
against a measured detection floor of ~0.12 at this n. And independently, the
responder-overlap figure puts **17 %** of compounds responding in both modalities,
inside the 11–34 % published for A549. Both say the same thing: a shared subspace,
not mutual predictability.

## Running it on HPC

```bash
mkdir -p logs
sbatch slurm/ws4a_pipeline.sbatch                       # stats -> ml -> plots
sbatch slurm/ws4a_pipeline.sbatch --stage stats
sbatch --gres=NONE -p <cpu> slurm/ws4a_pipeline.sbatch --stage stats
sbatch --export=ALL,WS4A_DATA=/scratch/$USER/stream slurm/ws4a_pipeline.sbatch
```

Standalone, like `ws4_batch_correction.sbatch`: its own `#SBATCH` header, plain
`sbatch`, reads no `configs/slurm.env`. Edit `--account` and `--partition` before the
first submission.

!!! note "Only XGBoost uses the GPU, and the CUDA situation differs from ws4.sif"
    XGBoost's PyPI wheel is built against **CUDA 13.3** and — unlike torch's
    equivalent, which fails outright — **it trains on a 12.8 driver**. Verified on the
    workstation's RTX A4000. `common.resolve_device` probes by training a tiny model
    rather than trusting an availability flag, and degrades to CPU with the reason
    logged.

    This is why the image needs no CUDA base and no torch: 60 packages, versus 215 for
    `ws4.sif`.

## Tuning is an optional second run, never a replacement

The models above are fitted on a **fixed, pre-declared grid**. `configs/ws4a_tuned.yaml`
swaps in an Optuna search that runs entirely inside the inner CV fold, writes to its
own output directory, and gives the permuted control the identical budget. It is a
separate run precisely because a tuned score and an untuned score are not comparable:
a bigger search raises the score on data with no signal at all.

`scripts/ws4a_compare.py` puts the two side by side on `gap_vs_permuted`, and
`ml --bias-sweep` measures what a given budget costs by running permuted labels only.
Full reasoning: [tuning](ws4a-tuning.md).

Two measured results from that run change how the baseline itself should be read:

- **A fixed grid is not the conservative choice.** It is an unexamined search with a
  budget of one. `tox_pulmonary_toxicity` × morphology scores **0.585 on permuted
  labels** under the fixed grid and falls to chance once tuned — so the baseline's
  −0.085 gap there was a bad grid point, not anti-predictive morphology.
- **Tuning removed every degenerate row: 10 to 0** on the full run — all ten XGBoost,
  on the two targets with the smallest minority class. The `degenerate` flag is what
  kept those ten from being reported as 0.500 null results in the baseline.
  (A preliminary `--quick` comparison suggested the opposite and was an artefact of
  its single CV repeat.) Measured outcome: [results](ws4a-results.md).

## Known limits

- **`--quick` is a smoke test, not a fast mode.** It shrinks folds and grids and says
  so; its numbers must not be reported.
- **`saga` is the bottleneck.** It is the only sklearn solver supporting an elastic-net
  penalty for classification, and it is slow on wide standardised data — ~93 s per
  (target, block, model) at 1024 features. `max_iter`/`tol` are in the config; this is
  work for a compute node.
- **SHAP falls back on XGBoost models with categorical splits.** The interventional
  estimator refuses them in shap 0.52. The fallback assumes feature independence and
  the figure is labelled accordingly — but stability selection is the more trustworthy
  read in that case.
- **`sparse_plsda` raises instead of truncating at high sparsity.** After the first
  component is deflated out, the `keep_x` selected columns can have no variance left,
  so the second component's `tᵀt` underflows and the vendored code raises
  `X-score collapsed to zero`. GridSearchCV absorbs it (that grid point scores `nan`,
  the rest are used), so no reported number is wrong — but the sparsest grid points
  are silently unavailable, and a tuned run prunes a larger share of its trials for
  the same reason. The right behaviour is to fit one component and report
  `n_components_ = 1`; that is not implemented.
  [Runbook §8.1](ws4a-runbook.md#81-sparse_plsda-x-score-collapsed-to-zero) has the
  full mechanism and how to check whether a tuned `sparse_plsda` row is affected.
- **The single-cell tier is not implemented.** [The plan](integration-plan.md) argues
  DL belongs at cell level only, and that is not built yet.
