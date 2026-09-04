# WS4A for dummies

*The cross-modal integration work, explained for a biologist who has never trained a
model. Nothing here is simplified to the point of being wrong; where a number is
quoted it is the real number from the real run, and where something is uncertain it
says so. The technical pages are [the plan](integration-plan.md), [the toolchain](ws4a-toolchain.md),
[the runbook](ws4a-runbook.md) and [tuning](ws4a-tuning.md).*

## The question, in one sentence

**When you treat cells with a drug and photograph them, does the photograph tell you
anything about the drug that you couldn't already have guessed from its chemical
structure?**

That is the whole project. Everything below is machinery for answering it honestly.

## The data: three witnesses describing the same 119 suspects

We have **119 compounds**. For each one, three independent descriptions:

| what we call it | what it actually is | how many numbers per compound |
|---|---|---|
| **chemistry** (`ecfp`) | the molecule's structure, encoded as a barcode of 1024 yes/no bits | 1,024 |
| **morphology** | what HepG2 liver cells looked like under the microscope after treatment — size, shape, texture, how organelles are arranged | 636 |
| **expression** | which genes the cells switched on or off after treatment | 41,780 |

Think of three witnesses who each saw the same person. The chemist saw the face. The
microscope saw the way they walk. The gene panel saw what they said. Two questions
follow: **do the witnesses agree with each other?** (that is Tier 1) and **can any of
them tell us what the person did?** (Tier 2).

Also for each compound we know some things about it already — its mechanism of
action for 64 of them, and whether it is known to be toxic to the heart, lungs,
kidneys and so on for about 70. Those are the *answers* we try to predict.

## What the ML is for, and what it can and cannot answer

The whole exercise is an **information-content** study, not a mechanism-discovery
study. It asks what each way of looking at a compound *tells you*. It does not ask
how any compound works, and it cannot answer that.

Why the distinction matters: imaging is cheap and scales to hundreds of thousands of
wells. Predicting a compound's properties from its chemical structure alone (QSAR)
has existed for decades and costs nothing — no cells, no microscope. So the case for
Cell Painting as a screening readout rests on the image telling you something the
structure does not. That is a decision-relevant question with a useful answer in
either direction, and it is the question this pipeline is built to answer.

### The five questions it actually resolves

**1. Which of the three descriptions carries the most information about a compound's
mechanism of action?**
Answer form: an honest gap per description. Full run — chemistry +0.175, morphology
+0.267, expression +0.339, and chemistry + morphology +0.318. Reading: **morphology
beats chemical structure**, gene expression beats both, and combining morphology with
chemistry gains +0.143 over chemistry alone.

**2. Is morphology redundant with chemistry, or complementary?**
Two descriptions can each carry signal and carry *the same* signal. Two independent
tests: does the combined `morphology + chemistry` block beat either alone, and do the
two tables agree structurally (Tier 1's adjusted RV)? Answer: adjusted RV 0.021,
**not significant** — the two are not redundant — and for mechanism the combination
gains +0.143 over chemistry alone. **Complementary, and measurably so.** For the
toxicity endpoints the same combination *loses* between 0.03 and 0.08, so the
complementarity is specific to mechanism, not general.

**3. Do the morphological and transcriptional responses describe the same thing?**
This is the cross-modal question proper, and the reason the workstream exists. Answer:
they agree **weakly** — adjusted RV 0.016, two of three tests significant. Read
biologically: at compound level, what a drug does to a cell's shape and what it does
to the cell's gene expression are only loosely coupled. That is interesting in itself,
and it constrains anyone hoping to use one modality as a cheap proxy for the other.

**4. Can toxicity be predicted from any of these at all?**
First pass: essentially no, for cardiac, pulmonary, renal, hepatic and fertility
endpoints. A negative answer at ~70 labelled compounds is weak evidence — but it is
honest evidence, and it says the bottleneck is the labels or the sample size, not the
choice of model. That is worth knowing before anyone builds a bigger model.

**5. Which cellular features carry whatever signal exists?**
The explainability stage. This is the closest the project comes to a biological
readout: stability selection plus the feature-name grammar can answer questions like
*are DNA-synthesis inhibitors separated by nuclear texture and area — consistent with
replication stress — or by mitochondrial intensity?* If the selected features cluster
in one compartment or one imaging channel, that is an interpretable phenotype rather
than just a score.

### What it cannot answer, stated plainly

- **Which gene or protein causes anything.** No causal claim is available from 119
  compounds and no perturbation of the mechanism itself.
- **The mechanism of any individual compound.** Separability across a population is
  not per-compound inference.
- **Anything about compounds unlike these 119.**
- **Whether imaging beats transcriptomics in general.** One cell line (HepG2), one
  assay, one dataset.
- **Anything at all about A549.** Its delivered morphology matrix is contaminated —
  44 of 615 features exceed |500| with a maximum of 1.5e19 — and the pipeline refuses
  to analyse it rather than quietly producing numbers from it.

### The uncomfortable part, and why it may be the real contribution

At 64 labelled compounds with 14 positives in the minority class, **most of what
looks like a result is selection bias**. That is why every number here is paired with
a shuffled-label control, and why a whole second run exists purely to measure what
hyper-parameter tuning costs in honesty.

Which means the most defensible deliverable from this work may be **methodological**
rather than biological: a demonstration, with measured numbers, that the standard
claims made at this sample size are inflated, and by how much. This project produced
three such measurements without looking for them:

| what was measured | the number |
|---|---|
| Plain RV between two **independent** tables of this shape (true value 0) | **0.93** — the adjusted version reads −0.0002 |
| Percent replicating for a control containing **no compound information at all** (every well replaced by its plate mean) | **89.3 %** — beating every real batch-correction method |
| A hand-set XGBoost configuration scoring on **shuffled labels** for one toxicity endpoint | **0.585** against a 0.5 baseline |

For an audience of people who build these pipelines, that is arguably worth more than
one more marginal AUC.

## Tier 2 — can a description predict what the drug does?

### What "training a model" means here

Take the 64 compounds whose mechanism is known. Show a program the morphology
numbers for 51 of them **together with** the right answer. Then show it the remaining
13 **without** the answer and ask it to guess. Count how often it is right. Repeat
with a different 13 held back, until every compound has been guessed once. That
average is the **score**.

A score of 0.50 means it guessed no better than a coin. A score of 1.00 means it was
always right. The models here score between about 0.34 and 0.77.

### The trap, and why you must not believe a score on its own

Here is the problem. We have 64 compounds and 41,780 gene numbers for each. **With
that many numbers and that few examples, some of the numbers will line up with the
answer purely by chance.** A model will find those and "predict" from them. It will
score well. And it will have learned nothing — it found a coincidence.

This is not a subtle effect. A published study (Boulesteix & Strobl) took data with
**no signal in it at all**, tried 124 different models, kept the best one, and got a
score of 0.60–0.70 — against a true value of 0.50. That is the size of the lie a
score can tell at this sample size.

So every model here is trained **twice**:

1. once on the real answers,
2. once on the **same answers shuffled** — so compound A gets compound B's mechanism,
   and so on. After shuffling there is no relationship left to learn. Whatever the
   model scores on shuffled answers is what coincidence alone buys.

The number to read is the **difference**:

```
honest gap  =  score on real answers  −  score on shuffled answers
```

A model that scores 0.67 on real answers and 0.45 on shuffled ones has a gap of
+0.22 — it found something. A model that scores 0.66 real and 0.56 shuffled has a
gap of +0.10 — most of its apparent skill was coincidence. **A high score with a
small gap is not a result.**

The tables call the shuffled run `PERMUTED` and the difference `gap_vs_permuted`.
When you see a score quoted from this project without the shuffled number next to
it, ask for the shuffled number.

### Why chemistry is always run first

A chemist can predict a fair amount about a drug from its structure alone — that is
what QSAR has done for decades. So chemistry (`ecfp`) is the **control**. If the
microscope cannot beat the chemical structure, the microscope added nothing, however
good its score looks in isolation.

### What the models found (HepG2, the full run)

Predicting "is this a DNA-synthesis inhibitor?" for 64 compounds — the honest gap for
the best of four models:

| description used | honest gap |
|---|---|
| chemistry alone | +0.175 |
| **morphology alone** | **+0.267** |
| chemistry + morphology | +0.318 |
| expression | +0.339 |
| expression + morphology | +0.344 |

**Morphology beats chemistry** (+0.267 vs +0.175), and adding morphology *to*
chemistry gains +0.143 — with all four models agreeing. That is the project's
result.

Gene expression is better still. And adding morphology on top of expression gains
almost nothing (+0.344 vs +0.339) — if you already have RNA-seq, the image is not
adding to it here. The practical argument for the image is that it is far cheaper per
sample, not that it carries more.

**Now the part that is easy to skip.** For the five toxicity endpoints — heart, lung,
kidney, liver, fertility — morphology adds **nothing**, and usually makes things
worse:

| target | chemistry alone | + morphology | change |
|---|---|---|---|
| mechanism of action | +0.175 | +0.318 | **+0.143** |
| pulmonary toxicity | +0.092 | +0.146 | +0.055 |
| renal toxicity | +0.147 | +0.118 | −0.030 |
| infertility | +0.193 | +0.129 | −0.064 |
| liver toxicity | +0.244 | +0.170 | −0.074 |
| cardiotoxicity | +0.141 | +0.057 | −0.084 |

Toxicity is predicted by chemistry, and only by chemistry. That is what a
structure-based baseline is *supposed* to do — and piling 636 imaging features onto
~70 compounds makes it worse, which is the small-sample dilution problem in one table.

So the one-sentence version is narrower and more defensible than "imaging works":

> **Cell Painting tells you about a compound's mechanism, over and above its chemical
> structure. It does not tell you about its toxicity.**

Full tables and figures: [WS4A results](ws4a-results.md).

### "0.500 ± 0.000" is not a result either

Some rows in the early tables show a score of exactly 0.500 with zero variation.
That is not "the model found nothing". That is a model that **refused to guess** —
it answered "not toxic" for every single compound, every time. Like a student who
writes "B" for every question and scores exactly what the answer distribution gives
them.

It happened because one of the model's safety settings was too strict for how few
positive examples there are (14 of 64). The pipeline now flags such rows as
`degenerate` and prints a warning. **Never quote one as a null result.** It measured
nothing.

In the full run there were **ten** such rows in the untuned version — every one the
same model, on the two questions with the fewest positive examples — and **zero**
after tuning, which searched that setting instead of fixing it by hand.

## Tuning — trying many settings, and why that is dangerous here

Every model has knobs — how strongly to penalise complexity, how deep to grow a
decision tree, and so on. The first run set them by hand, once. "Tuning" means
letting a program try many combinations and keep the best.

Sounds strictly better. It is not, at this sample size, for the same reason as
before: **the more combinations you try, the more likely one of them fits the
coincidences.** Trying harder raises the score on shuffled answers too.

Three rules keep it honest, and they are enforced in code rather than left to
discipline:

1. The knob-search only ever sees the compounds it is training on. The held-back
   compounds it will be scored on are hidden from it. Always.
2. **The shuffled-answers control gets exactly the same number of tries.** If you
   search 40 combinations on the real answers and only 5 on the shuffled ones, the
   "honest gap" is inflated by exactly the amount you are trying to measure.
3. The number of tries is kept small (20), and its cost is *measured*: the pipeline
   can run the search on shuffled answers only, at several budgets, and report how
   much score each budget manufactures out of nothing.

That last measurement produced a surprise: the hand-set knobs were **not** the safe
choice. On one toxicity question the hand-set model scored 0.585 on *shuffled*
answers — 8.5 points of pure coincidence — and the tuned one fell back to 0.50. On
another question tuning went the other way and added coincidence. There is no single
"tuning is good/bad" answer; there is a table, per question.

And tuning produced *more* refuse-to-guess models, not fewer: with 636 morphology
numbers and 64 examples, a search that is free to choose the penalty discovers that
"ignore everything and always say no" scores as well on the training data as
anything else. The flag catches those too.

## Tier 1 — do the witnesses agree with each other?

Before asking whether a description predicts anything, ask whether the descriptions
even agree: **do compounds that look alike under the microscope also have similar
gene-expression changes?**

The statistic for that is called RV, a kind of correlation between two whole tables
rather than two columns. It has a trap of its own: **plain RV reads 0.91 on two
tables of random numbers** of this shape. That is not a bug; it is what happens when
you have 41,780 columns and 119 rows. The pipeline uses the *bias-adjusted* RV,
which reads 0.01 on the same random tables, and every number quoted is that one.

| pair | adjusted RV | is it significant? |
|---|---|---|
| morphology vs expression | 0.016 | borderline — two of three tests say yes (p = 0.0001, 0.011), one says not quite (p = 0.097) |
| morphology vs chemistry | 0.021 | no (p = 0.19) |
| expression vs chemistry | 0.035 | yes (p = 0.0001) — but it is 3.5 % |

The honest summary: **the descriptions agree weakly.** Microscope and gene panel see
*something* in common, and it is small. A method (AJIVE) that tries to extract the
shared part found 4 shared components — but a control with the compound
correspondence destroyed also reached 4, so that is not distinguishable from chance
(p = 0.095). Do not report "four joint components".

### Controls that must fail

Every Tier 1 statistic is also run on two deliberately broken versions of the data:
the pairing between compounds scrambled, and pure random noise of the same shape.
On both, every statistic must collapse to zero. They do (adjusted RV between
−0.001 and +0.002 on every pair). **If they had not, the pipeline would be wrong,
not the biology.** This is the same logic as the shuffled answers in Tier 2: a
measurement nobody has tried to break is a measurement nobody understands.

## Why it is built this way — every choice, and the measurement behind it

Almost nothing in this pipeline is a default. Each choice below was made because a
measurement showed the obvious alternative was wrong. They are grouped by what they
protect against.

### Group 1 — choices that stop a number from lying

**Chemistry is always the control, and always runs first.**
*Why:* a chemist can predict a fair amount from structure alone. If morphology cannot
beat structure, morphology added nothing, however good its score looks by itself.
*Consequence:* `ecfp` is first in the block list and appears in every results table,
even when nobody asked about chemistry.

**Every model is trained a second time on shuffled labels.**
*Why:* with 64 examples and up to 41,780 features, some features line up with the
answer by luck. A model finds them and scores well having learned nothing.
*The measurement (literature):* Boulesteix & Strobl took data with no signal,
selected the best of 124 model variants by cross-validation, and got 31–41 % error
against a 50 % chance baseline. That is the size of the lie available at this n.
*Consequence:* the honest gap (real − shuffled) is the reported effect. Never the
raw score.

**Balanced accuracy, not accuracy.**
*Why:* one toxicity endpoint splits 68 to 2. Plain accuracy of 0.97 is available by
answering "yes" every time. Balanced accuracy averages the per-class hit rate, so
that strategy scores 0.5.
*Consequence:* every score in every table is on a scale where 0.5 = coin flip,
regardless of how lopsided the labels are.

**Guards that refuse to produce a number.**
*Why:* balanced accuracy hides less of the lopsided-label problem than accuracy does,
but a 68/2 target still cannot support a conclusion.
*The rule:* at least 30 labelled compounds, at least 10 in the smaller class, and at
least 15 % of the labelled rows in it.
*Consequence:* `tox_dermatological_toxicity` (68/2), `tox_hematological` (62/8) and
HepG2's `depmap_auc` (only 22 labelled) are skipped, with the reason written to
`skipped_targets_*.csv`. **A refusal is a result** — it says the data cannot answer
that question — and it is far better than a plausible-looking 0.97.

**Four different models, and all four are always reported.**
*Why:* running many models and reporting the best one *is* the selection-bias
mechanism above. The four are deliberately different in kind — a penalised linear
model, a sparse latent-component method, a margin classifier and a boosted tree —
so agreement between them means something.
*Consequence:* the model list is declared in the config in advance and
`report_all_models: true`. You do not get to pick a winner after seeing the scores.

**Nested cross-validation.**
*Why:* if any choice — which features, which settings, which scaling — is made using
the data you then score on, the score is optimistic.
*Consequence:* an outer loop estimates performance; every selection happens strictly
inside the inner loop, on the training part only. This is asserted mechanically, not
trusted: one pre-flight check tags every compound, replaces the search with a spy,
runs the real evaluation, and requires that no search ever saw all compounds and that
every compound was withheld from at least one search.

**"Exactly 0.5 with zero variation" is flagged, not reported.**
*Why:* it looks like a clean null result. It is a model that answered the same thing
for every compound.
*The measurement:* XGBoost's `min_child_weight = 5`, against 14 positives, allowed
**2 of 200 trees to grow a single split**. The rest returned the majority class, and
0.500 followed by construction.
*Consequence:* such rows are marked `degenerate` and a warning is logged naming them
a broken configuration. Two rows in the baseline; **four** in the tuned run (see
Group 3).

### Group 2 — choices that stop a *statistic* from lying

**The adjusted RV, never the plain one.**
*Why:* the plain RV coefficient is severely biased upward when features outnumber
samples.
*The measurement:* on two **independent** blocks at this exact shape (n = 94,
p = 615, q = 41,780), where the true value is 0 — plain RV **0.930**, Smilde's RV2
0.234, Mayer's adjusted RV **−0.0002**.
*Consequence:* `report_effect: rv_adj` in the config. The raw statistics stay in the
CSV, clearly marked, and are never the headline.

**Two destructive controls on every Tier 1 statistic.**
*Why:* a measurement nobody has tried to break is a measurement nobody understands.
*The controls:* (a) the compound-to-compound pairing between the two tables is
scrambled, so any agreement is impossible by construction; (b) both tables are
replaced by random noise of the same shape.
*The result:* every statistic collapses — adjusted RV between −0.001 and +0.002 on
every pair. **If they had not collapsed, the pipeline would be wrong, not the
biology.**
*Where this habit came from:* the same discipline applied to a standard Cell Painting
quality metric found that replacing every well by its plate mean — destroying all
compound information — scored **89.3 % percent replicating**, beating every real
batch-correction method. The metric was measuring plate structure, not biology.

**A permutation null on the joint rank itself.**
*Why:* a method that extracts shared structure will extract *something* from any two
tables. The question is whether it is more than it would find in nothing.
*The measurement:* AJIVE reported joint rank 4 — and the destroyed-correspondence
control also reached 4 (mean 1.55, p = 0.095).
*Consequence:* **"four joint components" is not reported as a finding.** This is the
clearest example on the page of a control changing a conclusion.

**A deliberately small number of components before CCA.**
*Why:* canonical correlation reaches r ≈ 0.9 on **permuted null data** if too many
principal components are retained.
*Consequence:* `pc_budget: 10` at n = 119, and the permutation p-value is always
reported alongside r. The observed r = 0.90 means nothing on its own — that is
exactly the value the null produces when the budget is too generous.

### Group 3 — choices about tuning, where the surprises were

**The tuned run is a separate run, writing to a separate directory.**
*Why:* it is compared against the baseline. Overwriting the baseline would destroy
the comparison.
*Consequence:* `configs/ws4a_tuned.yaml` changes the output path and the trial budget
and inherits everything else, so the two runs cannot drift apart in targets, guards
or data.

**The shuffled-label control gets the *identical* search budget.**
*Why:* searching harder raises the score on shuffled labels too. Search 40 settings
on the real labels and 5 on the shuffled ones, and the honest gap is inflated by
exactly the quantity being measured.
*Consequence:* this equality is **enforced in code and is deliberately not a config
option**, because making it configurable would let a later edit manufacture a gap
quietly.

**20 trials, not 500.**
*Why:* at 64 rows a large search fits coincidences.
*And the cost is measured, not assumed:* the pipeline can run the search on shuffled
labels only, at several budgets, and report how much score each budget manufactures
from nothing.

**The surprise: a fixed grid is not the safe choice.**
*The measurement:* on `tox_pulmonary_toxicity` × morphology, the **hand-set**
settings scored **0.585 on shuffled labels** — 8.5 points of pure coincidence — and
the tuned version fell back to 0.50. So the baseline's apparently anti-predictive
−0.085 gap there was a bad hand-chosen setting, not anti-predictive morphology.
On `tox_renal_toxicity` × chemistry it went the other way: −0.008 → +0.022 → +0.048
as the budget grew, the textbook shape.
*Conclusion:* there is **no single "tuning is good or bad" answer** — there is a
table, per question. A fixed grid is just an unexamined search with a budget of one,
and its bias is unmeasured unless you sweep the zero-trial point.

**The second surprise: tuning removed every refuse-to-guess model — 10 to 0.**
All ten in the untuned run were the same model type, on the two questions with the
fewest positive examples (14 and 16). Searching the setting instead of fixing it by
hand eliminated all of them.

*A correction worth seeing.* A preliminary smoke-test comparison reported the
opposite — 4 tuned against 2 untuned — and that appeared on this site. It was an
artefact of the smoke test using a single cross-validation repeat instead of five,
where one unlucky fold is enough to make a model look constant. **A smoke test finds
crashes; it does not measure effects.** The underlying worry is still real — with 636
features and 64 examples a free search *can* discover that "ignore everything and
always answer no" optimises the training folds — it simply did not happen here.

**And the search bought no measurable dishonesty.** Across all 120 comparisons the
median change in the shuffled-label score was **−0.0003**, and not one row was flagged
as having gained score on the control. At 20 trials the search is small enough to be
honest — measured, not assumed. It also produced no new findings: the number of
results clearing zero was **56 before and 56 after**.

### Group 4 — which data, and why not the other

**HepG2, not A549.** A549's delivered morphology carries 44 of 615 features above
|500| with a maximum of 1.5e19 — the fingerprint of a known normalisation defect.
The default is to **abort** rather than analyse it.
*Where that defect is understood:* the same failure mode was traced in this project's
earlier work — a normalisation epsilon of 1e-18 inflating zero-variance features to
2.4e19, which moved a quality metric from 35.9 % to 83.9 % once fixed.

**Compound-level, not single-cell.** Single-cell files for this data are roughly
3,000× larger than the aggregated profiles. Aggregation is a genuine trade-off —
it loses within-well heterogeneity — but a single-cell tier is not justified until the
compound-level question is answered.

**Chemical identity is the join key.** The three tables are matched by compound, not
by gene space. That is what makes a morphology row, an expression row and a structure
row describe the same experiment.

### Group 5 — methods considered and deliberately not used

Colleagues at the hackathon are running **MOFA** and **DIABLO**. Both were considered.

**MOFA** answers the same question AJIVE already answers here: which latent factors
are shared across modalities and which are specific to one. AJIVE is the
geometric counterpart of MOFA's Bayesian factor model. Adding MOFA would produce a
prettier figure carrying *less* information than what is already computed, because
the existing answer comes with a permutation null (joint rank 4, null also 4,
p = 0.095) that MOFA analyses usually lack.

**DIABLO** is a real methodological gap — it fits block-wise components with a design
matrix linking blocks, which is genuinely different from feeding a concatenated
matrix. It was still not adopted, for two reasons. There is no maintained Python port
of mixOmics (which is why sparse PLS-DA is hand-implemented in this repo), so DIABLO
means writing multi-block sparse PLS-DA and testing it. And at 64 rows with 14
positives, a method with *more* free parameters — per-block sparsity plus the design
matrix — is the wrong direction: it adds knobs to a problem defined by not having
enough data for the knobs already present.

**Neither addresses the actual bottleneck**, which is sample size and honest
evaluation. Both are high-dimensional latent-variable methods, so both are *subject
to* the selection-bias problem rather than remedies for it — DIABLO especially, since
its sparsity is chosen by cross-validation, the exact mechanism the tuned-versus-
untuned comparison exists to measure. Either will produce a convincing factor plot
on data containing no signal at all.

*The more useful position:* the controls in this pipeline — scrambled pairing, random
blocks, shuffled labels — apply to **any** method's output, including a colleague's.
Being able to say whether someone's MOFA factors survive a destroyed-correspondence
control is a larger contribution than being the third person with a factor plot.

### Group 6 — engineering choices that exist to protect the science

These sound like plumbing. Each one exists because getting it wrong would have
produced wrong numbers silently.

| choice | what it prevents |
|---|---|
| The parallel loop is seeded **per fold**, not by execution order, and a pre-flight check asserts the parallel result is identical to the serial one | Running on 25 workers instead of 1 changing the answer. The first version of that check **failed** — four of the models had no random seed set and were drawing from a per-process random state |
| The merge step **refuses** to write a results table if any unit is missing | A crashed task producing a file that looks like a complete run. For a deadline, a silently partial result is worse than an obvious failure |
| Two containers with shared packages pinned to identical versions | A number computed in one environment not being comparable with a number from the other |
| The pre-flight check runs before every job (23 assertions, ~25 s) | A mismatch between a call site and a hand-written algorithm surfacing hours into a run, after the expensive part finished — which is precisely how one earlier job died |
| Every path lives in one config file; no script hard-codes one | The workstation and the cluster silently reading different data |

### The one-sentence version

**Every number in this project is reported next to a version of itself computed on
data where the answer is known to be nothing — and several of those controls changed
a conclusion.**

## What to say, and what not to say

If you present this:

- Say **"morphology adds information about mechanism over and above chemical
  structure — +0.143, with all four models agreeing"**. You may say morphology beats
  chemistry *for mechanism* (+0.267 vs +0.175). Do not generalise it: say in the same
  breath that it adds **nothing for the toxicity endpoints**, and nothing on top of
  transcriptomics.
- Always show the shuffled-answer number next to the real one.
- Say **"the modalities agree weakly"**. Do not quote the plain RV of 0.9 — it is
  what random data gives.
- Do not present any toxicity result as a finding yet.
- Do not present a `degenerate` row as "no signal".
- **Do not claim batch correction is solved**, and never say CellProfiler produces
  wrong numbers — the project's [framing rules](../plan.md) explain why these two
  matter politically as well as scientifically.

## Where the numbers come from, and how to get new ones

Everything runs on the cluster from one config file; the [runbook](ws4a-runbook.md)
has every command. The fast version splits the work into 30 pieces (one per question
× description) that run on different nodes at once, then merges them — and the merge
refuses to produce a table if any piece is missing, so a half-finished run cannot be
mistaken for a finished one.

## Glossary

| term | meaning here |
|---|---|
| **block / modality / description** | one of the three tables: chemistry, morphology, expression |
| **feature** | one column of a table — one measured number per compound |
| **target / label** | the thing being predicted: mechanism class, or a yes/no toxicity |
| **model** | a program that learns to guess the target from the features. Four are used: elastic net, sparse PLS-DA, linear SVM, XGBoost — different in mechanism, so agreement between them means something |
| **score** | balanced accuracy: how often the guess is right, adjusted so a lopsided yes/no split can't inflate it. 0.5 = coin flip |
| **fold / cross-validation** | hold some compounds back, train on the rest, score on the held-back ones, rotate. 5 folds × 5 repeats = 25 scores, averaged |
| **permuted / shuffled** | the same run with the answers randomly reassigned to compounds. Measures what coincidence alone buys |
| **honest gap** | real score minus permuted score. The only number worth reading |
| **degenerate** | a model that gave the same answer for every compound. A broken configuration, not a result |
| **tuning** | searching over a model's settings. Raises real *and* permuted scores; only the gap says whether it helped |
| **adjusted RV** | agreement between two whole tables, corrected so random tables read ~0 |
| **control** | a deliberately broken version of the data on which the statistic must fail. If it doesn't, the pipeline is wrong |
