# WS4A — results pack

HepG2. 119 compounds, three modalities: morphology (636 OpenScreen Cell Painting
features), expression (41,780 TAHOE pseudobulk genes), chemistry (1,024-bit ECFP).
Every number below is an **honest gap** = score on real labels − score on the *same
labels shuffled*, both with the identical model, folds and search budget. A raw score
is never quoted on its own.

Two complete runs: **untuned** (fixed grid) and **tuned** (Optuna, 20 trials per
inner fold, shuffled control given the same budget). 6 targets × 5 feature blocks ×
4 models × 2 label conditions × 25 CV folds = 240 model fits per run.

---

## The numbers you can quote

**1. Morphology adds information about mechanism, over and above chemical structure.**

| block | honest gap (best of 4 models) |
|---|---|
| chemistry (ECFP) alone | +0.175 |
| **morphology alone** | **+0.267** |
| chemistry + morphology | **+0.318** |
| expression | +0.339 |
| expression + morphology | +0.344 |

Morphology beats chemistry (+0.267 vs +0.175) and **adds +0.143** on top of it, with
**4 of 4 models** independently clearing zero. → figures 01, 02

**2. It adds nothing for toxicity.** Across the five endpoints (cardiac, pulmonary,
renal, hepatic, fertility) adding morphology to chemistry changes the gap by −0.084
to +0.055, median **−0.064**. Only pulmonary is positive (+0.055, exactly 0.0545). Toxicity is
predicted by chemistry alone. → figure 02

**3. Fusion does not help.** expression + morphology (+0.344) vs expression alone
(+0.339) = **+0.005**. This *agrees* with the main WS4 deck's Task 1 conclusion,
reached by a different pipeline and a different task formulation.

**4. The modalities agree weakly but really.** morphology ~ expression: adjusted RV
0.016, Mantel p = 0.0001, PROTEST p = 0.011. Both destructive controls collapse
(adjusted RV −0.001 to +0.002). → figure 04

**5. The shared morphology–expression axis is interpretable in morphology and in
nothing else.** With a max-statistic permutation null on canonical loadings:
**14 of 636** morphology features survive — coherent nuclear DNA texture, Zernike
shape, DNA/ER co-localisation — and **0 of 41,780 genes**, **0 of 1,024 ECFP bits**.
The transcriptional side is diffuse, not driven by a few genes. → figures 05, 06

**6. Tuning bought no measurable bias.** 20 trials, median change on shuffled labels
**−0.0003**, zero `bias` verdicts in 120 comparisons. It fixed 10 broken models
(degenerate → 0) and produced **no new findings** (gaps clearing zero: 56 → 56).
→ figure 07

**7. Selecting 2,000 HVGs costs ~16× what the leakage it is criticised for is worth.**
Across the three models that produced a model at all (XGBoost was degenerate in every
arm, so it is not a measurement): reduction **−0.273** median, leakage **+0.017**
median. With all 41,780 genes the expression arm clears the best morphology arm
(+0.309 and +0.248 vs +0.233); with 2,000 HVGs it does not come close. → figure 10

  | model | all genes | 2k in-fold | 2k all-rows | reduction | leakage |
  |---|---|---|---|---|---|
  | linear_svm | +0.309 | +0.027 | +0.053 | −0.281 | +0.026 |
  | elastic_net | +0.248 | −0.025 | −0.007 | −0.273 | +0.017 |
  | sparse_plsda | +0.075 | −0.062 | −0.063 | −0.137 | −0.001 |

---

## Do NOT claim

- **Do not quote r₁ = 0.903** for the CCA. The 95th percentile of its own permutation
  null is **0.902**. The pair is significant only via the pooled statistic (p = 0.017).
  Quote the p-value.
- **Do not report "4 joint components"** from AJIVE. Joint rank is 4, but the
  destroyed-correspondence null also reached 4 (p = 0.095). Not significant.
- **Do not quote plain RV.** On random noise of this shape it scores as high as on the
  real data — higher on two of three pairs. Use `rv_adj`.
- **Do not quote a `degenerate` row as a null result.** Exactly 0.500 ± 0.000 means the
  model predicted one class every fold — a broken configuration, not a measurement.
  The untuned run has 10; the tuned run has 0.
- **Do not present a gene list** from this data without a max-statistic null. Zero of
  41,780 genes survive one here.
- **Nothing about A549** — its delivered morphology is contaminated (44 of 615 features
  exceed |500|, max 1.5e19) and the pipeline refuses to analyse it.
- The gap intervals in figure 09 are **anti-conservative** — CV folds share training
  data (Bengio & Grandvalet 2004). Lean on the 4-of-4 model consensus instead.

---

## Figures

### 01_headline
| file | shows |
|---|---|
| `01_signal_map.png` | best honest gap for every target × block, with how many of 4 models clear zero. Red = signal, blue = worse than its own shuffled control. |
| `02_does_morphology_add_to_chemistry.png` | the project's question. Left: chemistry alone → chemistry + morphology, per target. Right: incremental value of the microscope. |

### 02_controls
| file | shows |
|---|---|
| `03_why_every_number_needs_a_control.png` | the shuffled-label score is **not** 0.5 — it spans 0.377–0.598. Right panel: height is not the finding, distance above the diagonal is. |
| `04_tier1_controls_vs_nulls.png` | AJIVE's joint rank inside its null (p = 0.095); the CCA statistic outside its own (p = 0.017); plain RV scoring random noise as high as real data. |

### 03_crossmodal
| file | shows |
|---|---|
| `05_shared_axis_embedding.png` | compounds on the shared morphology–expression axis, the same with pairing destroyed, and r₁ against its null. |
| `06_what_defines_the_shared_axis.png` | 14/636 morphology features survive the null; 0/41,780 genes; 0/1,024 ECFP bits. Grey bars are large **and not evidence**. |

### 04_methods
| file | shows |
|---|---|
| `07_tuning_audit.png` | did tuning buy signal or bias — shuffled-score shift centred at zero, per-model gains, and 10 → 0 degenerate rows. |
| `08_model_concordance.png` | the three linear-family models agree (ρ 0.73–0.87); XGBoost much less (0.47–0.53). So "4 models agree" is a real statement. |
| `07b_tuning_biggest_movers.png` | the 25 combinations tuning moved most. Hollow = untuned, filled = tuned. The green rows starting exactly at zero are degenerate XGBoost models being repaired. |
| `09_gap_uncertainty.png` | every gap with an approximate interval. Read as "is this near zero", never as a p-value. |

### 05_hvg_experiment
| file | shows |
|---|---|
| `10_hvg_selection_cost.png` | three arms: all genes / 2,000 HVGs chosen in-fold / 2,000 chosen on all rows. Reduction costs −0.273; leakage worth +0.017. ✗ marks XGBoost, degenerate in every arm. |

### 06_supporting
Tuned-vs-untuned comparison and the standard pipeline set (agreement, embedding,
joint structure, ML performance, modality overlap). `compare_gap_slopes_full_audit.png`
is all 120 comparisons — an audit trail for questions, not a slide.

---

## Tables

| file | contents |
|---|---|
| `signal_map.csv` | best gap per target × block, model consensus counts |
| `incremental_value.csv` | the morphology-adds-to-chemistry numbers |
| `gap_intervals.csv` | every gap with se and approximate interval |
| `tuned_vs_untuned.csv` | per-row verdicts: improved / unchanged / degraded / bias |
| `crossmodal_loadings.csv` | only the 14 features that beat their permutation null |
| `hvg_deltas.csv` | reduction cost and leakage worth, per model |
| `ml_summary_untuned.csv`, `ml_summary_tuned.csv` | all 120 rows of each run |
| `tier1_agreement.csv` | RV / Mantel / PROTEST, observed and both controls |

---

## Reproducing any of it

```bash
bash scripts/ws4a.sh report     --baseline <out>/ws4a/ml --tuned <out>/ws4a_tuned/ml \
                                --stats <out>/ws4a/stats --cell-line hepg2
bash scripts/ws4a.sh crossmodal --cell-line hepg2 --null-reps 200
bash scripts/ws4a.sh hvg        --cell-line hepg2 --target moa --morph-ref 0.233
```

Full method: `docs/ws4/ws4a-results.md`, `ws4a-for-dummies.md`, `ws4a-tuning.md`.
