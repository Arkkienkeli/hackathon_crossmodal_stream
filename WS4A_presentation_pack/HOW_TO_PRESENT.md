# WS4A presentation guide

*Every figure in the [results pack](ws4a-results.md), explained for someone who has
never trained a model: what you are looking at, how to read it line by line, what it
proves, the sentence to say out loud, the questions you will be asked, and what you
must not claim. Then how to order them into a talk.*

The figures are in `WS4A_presentation_pack/`, numbered 01–10 in the order recommended
below.

---

## The story in one line

> **A photograph of drug-treated cells tells you something about what the drug does
> that its chemical structure alone does not — but only about its mechanism, not its
> toxicity.**

Everything else is the machinery for making that sentence believable at 119 compounds.

## The three ideas the audience must absorb, in order

Do not reorder these. Each one is unusable without the one before it.

1. **A score means nothing on its own.** With 64 examples and 41,780 measurements, a
   model can score 0.65 on data containing no information whatsoever. So every model
   is run twice — once on real answers, once on the same answers shuffled — and only
   the *difference* is reported. (figure 03)
2. **Where the difference is big.** (figures 01, 02)
3. **Why you should believe it.** (figures 08, 04, 07)

If you have five minutes, that is the whole talk: figure 03, figure 02, figure 08.

---

## If you only get two or three slides — read this section only

The rest of this page is reference. **This section is the talk.**

Three figures, chosen because each one is (a) **not already in the main WS4 deck**,
(b) **readable on its own** without any of my other figures, and (c) **changes what
somebody does next**. In priority order — if you get two slides, drop the third.

| # | figure | file | the one thing it establishes |
|---|---|---|---|
| **1** | *choose one* — signal map **or** incremental value | `01_headline/01_signal_map.png` **or** `01_headline/02_does_morphology_add_to_chemistry.png` | Morphology adds real information about **mechanism** over and above chemical structure — and nothing for toxicity |
| **2** | HVG experiment | `05_hvg_experiment/10_hvg_selection_cost.png` | The workstream's "gene expression doesn't beat morphology" is a property of the **gene selection**, not the modality |
| **3** | model concordance | `00_THE_THREE_SLIDES/SLIDE_3_models_agree.png` | Four model families that work differently **agree about where the signal is** — this is not one lucky model |

### Slide 1 is a genuine choice — pick by whether you will be narrating

Both figures carry the same headline. They differ in what else they carry and in how
much work they leave to the audience.

The **signal map** shows *levels*: every one of the 30 combinations, with the model
consensus counts printed in each cell. Every number the incremental figure computes is
on it — chemistry 0.175, morphology 0.267, both together 0.318 — plus two things the
incremental figure does not have: the **whole landscape**, including the near-white
morphology column that shows the toxicity endpoints are empty, and the **`n/4 models`
counts**, which put the credibility argument inside the same image.

The **incremental figure** shows *differences*. Your headline claim is a difference —
"morphology adds +0.143 **over** chemistry" — and on the heatmap the audience has to
do that subtraction in their heads. This figure does the arithmetic for them and draws
a zero line where the answer is.

| | signal map | incremental value |
|---|---|---|
| speaking live over the slide | ✅ walk them across the top row, then down the morphology column | |
| slide must stand alone (poster, emailed deck, no narration) | | ✅ the comparison is pre-computed |
| want landscape + toxicity nulls + model consensus in one image | ✅ | |
| want the "adds to chemistry" claim unambiguous | | ✅ |
| audience is methods-sceptical and will ask "was it one lucky model?" | ✅ the `n/4` counts answer it in place | |

**If you are presenting live, use the signal map.** If you have two slots for the
finding, show the signal map first for the landscape, then the incremental figure to
make the chemistry comparison explicit — in that case drop slide 3.

The reading guide for the signal map is [in the reference section](#figure-01--the-signal-map);
the deep background for the incremental figure is immediately below.

**Slide 3 changed after reading v1 of the main deck.** It used to be the shared-axis
gene analysis (0 of 41,780 genes survive a permutation null). Version 1 of the main
deck now carries its own gene-level mapping slide reporting **0 of 1.27 M feature–gene
pairs at FDR < 0.05** — the same conclusion by a different method, already covered. So
that figure moves to backup and model concordance takes the slot, because v1 also adds
CCA, PLS, sliced-Wasserstein distances and per-drug rankings: a **methods-heavier deck
invites methods-heavier scrutiny**, and "did you just get lucky with one model?" is now
more likely to be asked, not less.

**What is still left out, and why:** the controls and tuning-audit figures are
credibility rather than findings — hold them for questions. The gap-uncertainty figure
needs a caveat you do not want to spend a slide on. And the 120-row tuning-slopes
figure stays out even in a large auditorium — see [below](#a-note-on-the-120-row-slopes-figure).

---

### SLIDE 1 — Does the microscope beat the thing that is free?

*Shown here with the incremental figure. If you chose the signal map instead, the
background below applies unchanged — only the "What is plotted" subsection differs,
and the signal map's reading guide is [in the reference section](#figure-01--the-signal-map).*

![Incremental value](../assets/figures/ws4a-incremental-value.png)

#### Background: why this is the question nobody in the deck asked

Cell Painting takes one photograph of drug-treated cells stained in five channels and
turns it into hundreds of numbers describing nuclear shape, texture, organelle
arrangement and so on. The promise is that it is a **cheap, general, unbiased readout
of what a compound does to a cell** — you do not need to know what to look for in
advance, and it scales to hundreds of thousands of wells.

That promise is worth testing against the right baseline, and the right baseline is
**not random guessing.**

For thirty years, chemists have predicted compound properties from molecular structure
alone — QSAR. An ECFP fingerprint costs nothing: no cells, no plates, no microscope,
no imaging time, no analyst. You compute it from the molecule on a laptop in
milliseconds.

So "Cell Painting predicts mechanism better than chance" is not a finding anyone
should act on. **The decision-relevant question is whether the photograph tells you
something the free option does not.** Answering it requires all three descriptions —
imaging, transcriptomics, structure — measured on the *same* compounds. That is
exactly what this MuData has, and it is why the chemistry block is in every
comparison we ran. The main WS4 deck compares morphology against gene expression and
never brings chemistry in, so this comparison does not exist anywhere else in the
workstream.

#### What is plotted

Both panels use the **honest gap**: how well a model did on the real answers, *minus*
how well the identical model did when the answers were scrambled between compounds.
Scrambled means nothing is left to learn, so that second number is what luck alone
buys. Explain this in one sentence when you show the slide — the y-axis is
meaningless without it.

**Left panel.** One row per question. Two dots joined by a line:

- **grey dot** — honest gap using **chemistry alone**
- **coloured dot** — honest gap using **chemistry plus morphology**
- **green** if adding the photograph helped, **red** if it hurt

The direction of the line is the entire message. Rightward = the microscope
contributed something chemistry did not already have.

**Right panel.** The same thing as bars, three per question:

- **blue** — morphology alone *minus* chemistry alone → is the image better *instead of*?
- **green** — chemistry+morphology *minus* chemistry → is it better *as well as*?
- **orange** — expression+morphology *minus* expression → does it add on top of RNA-seq?

#### Where each number comes from — the worked example to use when someone is lost

The single most common confusion is that the numbers look like accuracies and are not.
Walk through one box:

**Morphology predicting mechanism, the box reading +0.267.**

| | |
|---|---|
| best model, **unshuffled** (real answers) | got **74.5 %** right |
| the same model, **shuffled** (answers dealt out at random) | still got **47.7 %** |
| printed in the box | 0.745 − 0.477 = **+0.267** |

Two things to draw out of that:

1. **It is not accuracy.** The model was right 74.5 % of the time. The reported figure
   is how far it beat *its own* shuffled twin.
2. **The shuffled run was 47.7 %, not 50 %.** Across the four models in that one box the
   shuffled runs ranged **47.7 – 51.7 %**. Each box therefore gets compared against its
   own control, never against a nominal 0.5 — that spread is exactly why.

And what "shuffled" means concretely: every box is run **twice**. Unshuffled, each drug
keeps its true answer. Shuffled, drug A is given drug B's answer, so the link between
measurements and answer is destroyed on purpose and there is nothing left to learn. The
model still scores something, because with 119 drugs and up to 41,780 measurements some
of them line up with random answers by chance. That residue is the luck level.

The results table holds both halves: **120 rows unshuffled, 120 shuffled, 240 in
total.** `Results/ws4a_tuned/ml/ml_hepg2.csv`, column `permuted`.

#### The numbers

| question | chemistry alone | + morphology | change |
|---|---|---|---|
| **mechanism of action** | +0.175 | **+0.318** | **+0.143** |
| pulmonary toxicity | +0.092 | +0.146 | +0.055 |
| renal toxicity | +0.147 | +0.118 | −0.030 |
| infertility | +0.193 | +0.129 | −0.064 |
| liver toxicity | +0.244 | +0.170 | −0.074 |
| cardiotoxicity | +0.141 | +0.057 | −0.084 |

Morphology **alone** also beats chemistry alone for mechanism: +0.267 vs +0.175, the
blue bar at +0.093.

Every orange bar is near zero, mechanism included (+0.005). Once you have
transcriptomics, the image adds nothing measurable on top.

#### Why toxicity goes the other way — have this answer ready

Adding 636 columns to about 70 compounds gives a model 636 fresh opportunities to fit
a coincidence. If those columns carry no signal for that particular question, the
added noise costs more than the added information is worth. That is textbook
small-sample dilution, and it is exactly what the four red rows are.

There is also a label-quality point worth making: "known to be hepatotoxic" is a
coarse, literature-derived, often dose- and context-dependent annotation. It is a much
noisier target than a curated mechanism class. Do not present the toxicity nulls as
"Cell Painting cannot see toxicity" — present them as "not from 70 compounds with
these labels".

#### Say this

> "Chemistry is our control, because chemistry is free — you compute it from the
> molecule, no experiment at all. So the question is not whether morphology beats
> chance, it is whether morphology beats the free option. For mechanism of action:
> chemistry alone gives 0.175, and adding the photograph takes it to 0.318 — a gain of
> 0.143, with all four of our models agreeing independently. For all five toxicity
> endpoints: no. Adding morphology makes it worse. And the orange bars say that once
> you already have RNA-seq, the image adds nothing on top of it. So the claim is
> narrow and it is defensible: **Cell Painting tells you about mechanism, over and
> above structure. It does not tell you about toxicity.**"

#### Questions you will get

**"Isn't 0.143 small?"** It is a 82 % increase over the chemistry-only gap. And the
absolute scale is compressed by the small sample — what matters is that four
independent model families all found it and the shuffled control did not.

**"Why not just use RNA-seq then?"** Per sample, imaging is far cheaper and faster.
That is the practical argument. Do not dress it up as an information argument — on
this dataset expression carries more.

**"Is this specific to HepG2 / to these 119 drugs?"** Yes. State it before someone
else does.

---

### SLIDE 2 — Why our gene-expression result differs from Task 1

![HVG experiment](../assets/figures/ws4a-hvg-experiment.png)

#### Background: the most routine step in single-cell analysis

Almost every single-cell RNA workflow begins by keeping only the **highly variable
genes** — typically the top 2,000 of 20,000–40,000. It is so standard it is rarely
discussed. Two reasons are given: it makes computation tractable, and it removes genes
whose variation is mostly technical noise.

The main WS4 deck does this, and does the honest thing by flagging it: *"HVGs were
selected globally before supervised CV, so this is an exploratory workstream benchmark
rather than a final leakage-free classifier."*

**That caveat bundles two completely different problems**, and they point in opposite
directions:

1. **Reduction.** Going from 41,780 genes to 2,000 throws information away. Whether
   that hurts depends entirely on whether the discarded genes carried signal.
2. **Leakage.** Choosing *which* 2,000 using all compounds — including the ones later
   held out for testing — lets the test set influence the feature list it will be
   scored on. This is a known and serious effect in exactly this setting: Ambroise &
   McLachlan showed in 2002 that gene selection performed before cross-validation on
   microarray data produces near-zero error rates on data with no signal at all.

Because both were bundled, nobody knew which one was actually costing anything. This
experiment separates them.

#### The design — three arms, everything else identical

| arm | genes | who chose them |
|---|---|---|
| **all genes** | 41,780 | nobody — no selection at all |
| **2,000 HVGs, in-fold** | 2,000 | chosen **separately inside each training fold**, so held-out compounds never influence the choice — leakage-free |
| **2,000 HVGs, all rows** | 2,000 | chosen **once using all 119 compounds** — the deck's approach |

Same nested cross-validation, same models, same folds, and each arm gets its own
shuffled-label control. The only thing that changes between arms 2 and 3 is *which
rows were allowed to influence the gene list* — which is precisely the definition of
the leakage.

So:

- **arm 2 − arm 1** isolates what **reduction** costs
- **arm 3 − arm 2** isolates what the **leakage** was worth

#### What is plotted

**Left panel.** Three groups of bars, one per arm, coloured by model. The **brown
dotted line is the best morphology arm (+0.233)** — the comparison the entire
experiment exists to make. **✗ marks XGBoost**, which was degenerate in every arm
(predicted one class in every fold) and therefore is not a measurement at all.

**Right panel.** The two effects separated. Blue = cost of reduction. Red = worth of
the leakage.

#### The numbers

| model | all genes | 2k in-fold | 2k all-rows | **reduction** | **leakage** |
|---|---|---|---|---|---|
| linear_svm | **+0.309** | +0.027 | +0.053 | **−0.281** | +0.026 |
| elastic_net | **+0.248** | −0.025 | −0.007 | **−0.273** | +0.017 |
| sparse_plsda | +0.075 | −0.062 | −0.063 | −0.137 | −0.001 |

Median reduction **−0.273**; median leakage **+0.017**. **Throwing genes away cost
about sixteen times what the leakage it was criticised for was ever worth.**

And the load-bearing observation: with **all** genes, two models sit **above the brown
line** — expression beats the best morphology arm. With **either** 2,000-gene version,
nothing comes close.

#### How to present this without stepping on colleagues

This **reconciles two analyses**; it does not overturn one. Lead with the agreement:

> "First, where we agree. The deck finds that fusing morphology and gene expression
> does not improve mechanism prediction. We got the same thing independently — our
> fusion gain is +0.005, theirs is −0.096. Different pipeline, different task
> formulation, same conclusion. That is a replication and it is worth saying.
>
> Second, one thing we can add. The deck's Task 1 selects 2,000 highly variable genes
> and flags the leakage in its own caveat. We separated the two things that caveat
> bundles. Blue is what throwing genes away costs. Red is what the leakage was
> actually worth. Reduction costs about sixteen times more. And look at the brown
> line — that is the best morphology arm. With all 41,780 genes, expression clears it.
> With 2,000 HVGs it does not come close. So the finding that gene expression does not
> beat morphology looks like a property of the gene selection rather than of the
> modality. Our two pipelines agree once the same genes are used."

#### Caveat to state on the slide

This ran on **mechanism of action only, one target, untuned** — which is why the
morphology reference is the untuned +0.233 rather than the tuned +0.267. Everything in
the figure is like-for-like.

---

### SLIDE 3 — Four different models, one answer

![Model concordance](../assets/figures/ws4a-model-concordance-slide.png)

#### Background: why consensus, and not a p-value

At 119 compounds the honest thing to admit is that our uncertainty intervals are
**anti-conservative**. Cross-validation folds share training data, so they are not
independent, and there is no unbiased estimator of the variance of k-fold
cross-validation (Bengio & Grandvalet 2004). Any confidence interval we draw is
narrower than the truth.

Rather than lean on a shaky interval, we lean on something sturdier: **four model
families that fail in different ways, agreeing about where the signal is.** For a
small-sample study that is the more honest evidence, and it is the direct answer to
the objection every small-n result attracts — *"you got lucky with one model."*

#### What is plotted

**Left panel.** Each number is a **Spearman ρ** between two models, computed across
all 30 target × block combinations. Take model A's 30 honest gaps and sort the
combinations best-to-worst; do the same for model B; ρ asks how similar those two
orderings are. 1.00 = identical, 0 = unrelated. The dark diagonal is each model
against itself — ignore it.

**Right panel.** All 30 combinations, one row each, with all four models plotted, and
**sorted by their weakest model** — a combination is only as good as the worst of the
four. The red dashed line is zero. The shaded band at the top is the set where *every*
model clears zero (14 of 30).

**Every row is labelled**, and the hierarchy comes from colour rather than omission:
the 14 combinations where all four models clear zero are **green and bold**, the other
16 recede to grey. So the rows worth naming stand out without the rest being deleted —
and the sweep from near-zero to strongly positive stays visible, which is the honest
landscape in one picture.

The `tox_` prefix is dropped from the labels (every toxicity target carries it) to buy
the width that keeps the longest labels clear of the matrix panel.

#### The numbers

|  | elastic_net | linear_svm | sparse_plsda | xgboost |
|---|---|---|---|---|
| elastic_net | — | **0.87** | 0.75 | 0.47 |
| linear_svm | 0.87 | — | 0.73 | 0.50 |
| sparse_plsda | 0.75 | 0.73 | — | 0.53 |
| xgboost | 0.47 | 0.50 | 0.53 | — |

Two blocks. The three **linear-family** models agree strongly (0.73–0.87) — they are
variations on drawing a dividing line, so that is expected. **XGBoost, a boosted
tree, agrees much less** (0.47–0.53).

**That low number is the interesting one.** If all four sat at 0.95 you would have
four views of essentially one model, and "they all agree" would mean nothing. At 0.5,
when XGBoost does land on the same combinations as the others, that is independent
confirmation rather than an echo.

On the right, **14 of the 30 combinations have all four models right of zero** — the
shaded band — including `moa-fine · morphology`, the headline cell.

#### Be honest about the alternative reading

A ρ of 0.5 can mean two things: XGBoost is finding different *real* structure, or
XGBoost is simply noisier and worse on this data. Both are consistent with 0.5 — it
had 10 degenerate rows in the untuned run and rarely wins a combination. **The safe
claim is the narrow one:** the three linear models agree strongly, and on the top
combinations XGBoost independently clears zero as well. Do not oversell the "different
inductive bias" story.

#### Sorting by the weakest model is deliberate

Sorting by the *mean* would scatter the "all four clear zero" rows through the panel,
and the shading would read as zebra stripes rather than a set. Sorting by the
*minimum* makes that set exactly the top block — and it is the conservative ranking
anyway: a combination is only as good as its worst model.

Sixteen of the thirty do not qualify. They are plotted, unshaded, and their spread
below the band is the point of the figure as much as the band itself.

#### Say this

> "The obvious objection to anything from 119 compounds is that we got lucky with one
> model. So: four model families that look for structure in genuinely different ways.
> On the left, how similarly each pair ranks all thirty combinations. The three linear
> ones agree at 0.73 to 0.87. XGBoost, which builds decision trees, agrees much less —
> around 0.5. And that low number is the point: if all four agreed perfectly you would
> have four views of one model. On the right, the ten strongest combinations with every
> model shown, sorted by their weakest model. Most sit at zero — the signal is
> concentrated at the top. In the shaded band, fourteen of thirty, every single model
> clears zero, and that includes mechanism from morphology. Those are the ones we
> quote."

#### A note on the 120-row slopes figure

The full tuned-vs-untuned slopes figure — all 120 comparisons — stays out of the deck
**even though the auditorium is large**, for two reasons that a bigger screen does not
fix:

1. **It is dense, not merely small.** 120 rows at ~35 px each, projected 3 m tall, is
   about 2.5 cm per row — roughly 0.07° from 20 m back, well under the ~0.3° needed to
   read comfortably. A wider screen enlarges the rows *and* the gaps proportionally,
   so the ratio never improves.
2. **Its message is a null about tuning** — 56 results cleared zero before, 56 after.
   That is your methodology being careful, not your finding, and in a limited slot it
   competes with the chemistry result and loses.

A 25-row version exists (`04_methods/07b_tuning_biggest_movers.png`) if the topic comes
up in questions.

## What the ML run actually was, step by step

*If someone asks "what did you actually do?", this is the answer in twelve steps, in
plain language. Worth reading before you present, and worth having as a backup slide.*

### Why we ran it at all

Tier 1 of this workstream asks **do morphology and gene expression agree with each
other?** The ML run asks a different question: **is either of them any use?** — and
specifically, **does either beat chemical structure, which is free?**

Agreement is not usefulness. Two thermometers can agree perfectly and both be broken.
Two witnesses can contradict each other and both be telling you something valuable.
Tier 1 asks whether the witnesses tell the same story; the ML run asks whether any
witness can actually identify the suspect.

### Step 0 — what we start with

119 drugs. For each, three completely different descriptions: a **photograph** of the
treated liver cells, a readout of **which genes** switched on or off, and the drug's
**chemical structure** as a barcode. The same 119 drugs in all three — that is what
makes them comparable, and the compound is the bridge, not the cell.

### Step 1 — squash everything to one row per drug

The raw gene data is **384,533 individual cells**, roughly 2,700 per drug (range
925–8,171). But cells given the same drug are replicates, not independent
observations. So they are averaged into **one row per drug**. Same on the imaging
side: many photographed wells become one profile.

**Result: a spreadsheet with 119 rows.**

This is the most important fact in the project. 384,533 cells *sounds* enormous. To
the model it is **119 examples** — about a small classroom. Everything else in the
design follows from that number.

### Step 2 — decide what we are trying to guess

Two things are already known for these drugs from the literature: the **mechanism**
(how it works) and whether it is known to be **toxic** to heart, lung, kidney, liver
or fertility. These are the answers we hide and ask the model to recover.

### Step 3 — throw out the questions the data cannot answer

**Mechanism:** 55 of 119 drugs are labelled "unclear" — nobody knows. Dropped. Of what
remains the biggest group has 14 and every other has ≤5; you cannot teach a model to
separate groups of three. So we ask the one answerable version: *is this a
DNA-synthesis inhibitor?* — **14 yes, 50 no, n = 64**.

**Toxicity:** two endpoints are **refused outright**. Dermatological toxicity is
68 yes / 2 no — a model that always says "yes" is right 97 % of the time and has
learned nothing. Rather than report that as a success, the pipeline stops and records
why. Hematological (62/8) likewise.

**Six questions survive.**

### Step 4 — decide what the model may look at

Five options: chemistry alone, photograph alone, genes alone, photograph **+**
chemistry, photograph **+** genes.

**Chemistry is in there on purpose, and it is the point.** You get it from the
molecule on a laptop — no cells, no microscope, no sequencing. So "the photograph
beats random guessing" is not interesting. The only question worth asking is whether
the photograph beats, or adds to, the thing that is already free.

6 questions × 5 options = **30 combinations** — the 30 squares of the signal map.

### Step 5 — the actual test: hide some drugs

For one combination, say *mechanism from photographs*: show the model **51 drugs with
their answers**, then show it the **other 13 with the answers hidden** and count how
often it guesses right. It never saw those 13 answers, so it cannot cheat.

### Step 6 — do that 25 times with different drugs hidden

One split could be lucky — maybe those 13 were easy. So rotate: hide a different 13,
retrain from scratch, score again, 25 times, so every drug is hidden repeatedly. The
score is the average.

### Step 7 — do the whole thing again with the answers scrambled

**This is the step that makes everything else trustworthy, and it is the one usually
skipped.**

Same drugs, same photographs, but drug A gets drug B's mechanism, B gets C's, and so
on. Now there is genuinely **nothing to learn**. Run the identical procedure.
Whatever it scores now is pure luck.

Why bother: with 119 rows and up to 41,780 numbers each, some of those numbers line up
with the answer by coincidence, and a model will find them. On our grid **luck alone
scored between 0.38 and 0.60**, where 0.50 is a coin flip. So a model scoring 0.60 may
have found something real or nothing at all — the score cannot tell you which.

### Step 8 — subtract

```text
   score on real answers
 − score on scrambled answers
 = the honest gap
```

**That difference is the only number we report.** Every value in the signal map is one
of these.

### Step 9 — use four different kinds of model

Four families that look for structure in genuinely different ways: three variations on
drawing a straight dividing line, plus one that builds decision trees. If only one
finds something, that is a quirk of that model. If all four find it independently, it
is in the data. That is what `4/4 models` under each square means.

### Step 10 — repeat for all 30 combinations

30 combinations × 4 models × 2 label conditions = **240 runs**, each with 25
train-and-test cycles. Then **the whole thing again** with the models' settings
automatically tuned, to check tuning was not quietly manufacturing the result.
**480 runs, roughly 12,000 model fits.**

### Step 11 — split it across the cluster

About 22 hours as one job. Cut into **30 independent pieces**, one per combination,
run simultaneously: **~20 minutes**. A final step glued them back together — and
**refused to write a results table while any piece was missing**. That is how a failed
task was caught, instead of a table that looked complete and was not.

### Step 12 — read the map

Each square holds the best honest gap of four models. **Red = real signal. Blue = the
model did worse than its own scrambled version.** And the answer that fell out:

> The photograph tells you something real about a drug's **mechanism** — more than its
> chemical structure does, and it *adds to* chemistry rather than repeating it.
> For **toxicity**, the photograph adds nothing; chemistry alone does better.

### One caveat to know before anyone asks

`morphology + expression` is **98.5 % expression** — 41,780 of its 42,416 columns.
The blocks are concatenated and each column standardised, so the wider block dominates
by construction. That is why that column tracks `expression` almost exactly (+0.344 vs
+0.339). **`morphology + ecfp` (636 + 1,024) is the balanced fusion**, and it is the
one to interpret.

---

## Reference: fuller running orders

*Only if you get more than three slides.*

=== "5 minutes (3 slides)"

    | # | figure | the one thing it does |
    |---|---|---|
    | 1 | **03** why every number needs a control | teaches the honest gap — without it nothing later parses |
    | 2 | **02** does morphology add to chemistry | the result, and the question it answers |
    | 3 | **08** model concordance | why it is not one lucky model |

=== "15 minutes (8 slides)"

    | # | figure | the argument |
    |---|---|---|
    | 1 | — | the question and the three modalities (no figure; use the table below). If the audience is methods-minded, [the twelve steps](#what-the-ml-run-actually-was-step-by-step) is the backup slide for "what did you actually do?" |
    | 2 | **03** why a control | the trap, and the fix |
    | 3 | **01** signal map | where signal exists across the whole grid |
    | 4 | **02** incremental value | **the headline** — morphology adds to chemistry for mechanism only |
    | 5 | **08** model concordance | four different model families agree |
    | 6 | **04** Tier 1 controls | our controls work, and one of them killed a finding of ours |
    | 7 | **06** what defines the shared axis | 14 morphology features, zero genes |
    | 8 | **10** HVG cost | the methodological finding, and how it reconciles us with the main deck |

=== "Poster / backup"

    Figures 05, 07, 09 and everything in `06_supporting/`. Keep them out of the main
    flow; bring them out when someone asks.

**Slide 1 has no figure.** Put this table on it — the audience cannot follow anything
without knowing what the three blocks are:

| what we call it | what it actually is | numbers per compound |
|---|---|---|
| chemistry (`ecfp`) | the molecule's structure as a 1,024-bit barcode | 1,024 |
| morphology | how HepG2 cells looked under the microscope after treatment | 636 |
| expression | which genes switched on or off after treatment | 41,780 |

119 compounds. Same compounds in all three. That last point is the entire study
design: **the compound is the bridge**, not the cell.

---

## Figure 03 — Why every number needs a control

**Show this first. Nothing else on the page can be read without it.**

![Why subtract](../assets/figures/ws4a-why-subtract.png)

### What you are looking at

Two panels. Both are about the *shuffled-label* run — the same models trained on the
same data, but with each compound given a **random other compound's answer**. After
shuffling there is nothing left to learn, so whatever a model still scores is pure
luck.

**Left panel.** A histogram. Each of the 120 grey blocks is one target × block ×
model combination. The horizontal axis is the score that combination got **on
shuffled labels**. Red dashed line at 0.500 is the coin-flip level. Blue line is the
average actually observed, 0.483.

**Right panel.** A scatter plot. Every dot is again one combination. Horizontal axis
is its score on shuffled labels; vertical axis is its score on real labels. The red
dashed diagonal is where the two are equal. Green dots are the ones whose gap
excludes zero; grey dots are the rest.

### How to read it

- **Left:** if the shuffled score were always exactly 0.500, this histogram would be a
  single spike on the red line, and you could simply compare any score to 0.5. It is
  not a spike. It runs from **0.377 to 0.598** — a quarter of the entire scale.
- Why it is not 0.5: how the folds happen to split, how imbalanced that target's
  labels are, how many features the model has to overfit. It is *structure*, specific
  to each combination, not noise you can average away.
- **Right:** a dot's **height** is how well the model did. Its **distance above the
  diagonal** is how much of that was real. Look at the dots near real-score 0.62 that
  are grey — those models scored well and learned nothing, because their shuffled
  twin scored just as well.

### Say this

> "Before any result, one thing. With 64 compounds and up to 41,780 measurements, a
> model can score well on pure noise. So we ran everything twice — once with the real
> answers, once with the answers shuffled between compounds. If you shuffle, there is
> nothing to learn. Whatever the model still scores is luck. Here is what luck was
> worth across our whole grid: not 0.5, but anywhere from 0.38 to 0.60. So we never
> report a score. We report the difference between the real run and its own shuffled
> twin. We call it the honest gap, and it is the only number on any slide after this."

### Questions you will get

**"Isn't shuffling just a permutation test?"** Yes — the difference is that we run it
for *every single* model × target × block combination and subtract it individually,
rather than computing one global p-value. Each combination has its own luck level, as
the left panel shows.

**"Why is the mean below 0.5?"** Balanced accuracy over 25 folds of ~14 held-out
compounds is noisy, and a model that overfits scores *below* chance on held-out data.
Expected at this sample size. What matters is that the real and shuffled runs of the
same combination are affected identically, so the subtraction removes it.

---

## Figure 01 — The signal map

![Signal map](../assets/figures/ws4a-signal-map.png)

### What you are looking at

A grid. **Rows are the questions** we tried to predict — mechanism of action at the
top, then five toxicity endpoints. **Columns are the descriptions** we tried to
predict them from. Each cell holds the best honest gap of four models.

**Colour: red means signal, blue means the model did worse than its own shuffled
control.** (The colour scale is centred at zero, so near-white is near-nothing.)

Under each number is `n/4 models` — how many of the four independent models
*separately* got a gap whose uncertainty interval excludes zero. **Bold** numbers are
where at least three of four agree.

### How to read it

Read the **top row first**. Mechanism of action is the only row that is dark across
the board — every description beats its own shuffled control, and two reach +0.34.

Then read **down the `morphology` column**. It is dark at the top (+0.267, 4/4
models) and almost white everywhere below (+0.012, +0.011, +0.008, +0.064, −0.005,
and 0/4 models each time). Morphology carries information about mechanism and
essentially nothing about toxicity.

Then read **down the `ecfp` column**, which is chemistry. It is moderately red all
the way down — chemistry predicts the toxicity endpoints, and morphology does not.

### Say this

> "Six questions down the side, five kinds of information across the top, and in each
> cell the honest gap — real minus shuffled. Red is signal. The top row is mechanism
> of action and it is the only row where everything works. Look down the morphology
> column: strong at the top, nothing below. Now look at the chemistry column: it is
> the one predicting the toxicity endpoints. So already the story is narrow —
> imaging is telling us about mechanism, not toxicity."

### What NOT to say

Do not compare two cells and say one is significantly better than the other. The
intervals are wide and overlapping. The map shows **where** signal is, not a ranking.

---

## Figure 02 — Does morphology add to chemistry? (**the headline**)

![Incremental value](../assets/figures/ws4a-incremental-value.png)

### Why this figure is the point of the project

Chemistry is free. No cells, no microscope, no plates, no sequencing — you get an
ECFP fingerprint from the molecule's structure on a laptop. Predicting drug properties
from structure has been standard for thirty years.

So "morphology beats chance" is not interesting. **"Morphology beats, or adds to,
chemistry"** is the only version of the claim worth making, and it is what this figure
tests.

### What you are looking at

**Left panel.** One row per question. Two dots joined by a grey line:

- **grey dot** = honest gap using chemistry alone
- **coloured dot** = honest gap using chemistry **plus** morphology
- **green** if adding morphology helped, **red** if it hurt

The line's direction is the whole message: rightward = the microscope added
something.

**Right panel.** The same information as bars, three per question:

- **blue** — morphology alone minus chemistry alone (is the image better *instead of*?)
- **green** — chemistry + morphology minus chemistry (is it better *as well as*?)
- **orange** — expression + morphology minus expression (does it add on top of RNA-seq?)

Right of the zero line = imaging added information.

### How to read it

| question | chemistry alone | + morphology | change |
|---|---|---|---|
| **mechanism of action** | +0.175 | **+0.318** | **+0.143** |
| pulmonary toxicity | +0.092 | +0.146 | +0.055 |
| renal toxicity | +0.147 | +0.118 | −0.030 |
| infertility | +0.193 | +0.129 | −0.064 |
| liver toxicity | +0.244 | +0.170 | −0.074 |
| cardiotoxicity | +0.141 | +0.057 | −0.084 |

One large green bar and one small one; four red. Morphology alone also beats
chemistry alone for mechanism (+0.267 vs +0.175, the blue bar at +0.093).

The **orange bars are all near zero** — including for mechanism (+0.005). Once you
have transcriptomics, the image adds nothing measurable.

### Say this

> "Chemistry is the control, because it is free — you get it from the molecule, no
> experiment needed. So the question is not whether morphology beats chance, it is
> whether it beats chemistry. For mechanism of action, yes: chemistry alone gives
> 0.175, adding morphology takes it to 0.318 — a gain of 0.143, and all four of our
> models agree independently. For all five toxicity endpoints, no: adding morphology
> makes it worse. And the orange bars say that once you already have RNA-seq, the
> image adds nothing on top. So the honest claim is narrow: **Cell Painting tells you
> about mechanism, over and above structure. It does not tell you about toxicity.**"

### Questions you will get

**"Why does adding data make it worse?"** Small-sample dilution. Adding 636 columns
to ~70 compounds gives the model 636 more chances to fit a coincidence. If those
columns carry no signal for that question, the extra noise costs more than nothing.
This is exactly why the toxicity rows go red.

**"Isn't +0.055 for pulmonary a second win?"** Weakly. It is 4/4 models on the signal
map, so it is not nothing — but it is a fifth the size of the mechanism effect, on one
of five endpoints tested. Mention it, don't lead with it.

**"So imaging is worse than RNA-seq?"** For information about mechanism, yes on this
dataset. But per sample, imaging is far cheaper and faster. That is the practical
argument and it is a legitimate one — just don't dress it up as an information
argument.

---

## Figure 08 — Do four different models agree?

![Model concordance](../assets/figures/ws4a-model-concordance.png)

### Why it comes right after the result

The obvious objection to any small-sample result is *"you got lucky with one model."*
This is the answer, and it is stronger here than a p-value would be.

### What you are looking at

**Left panel.** A 4×4 grid of the four models against each other. Each number is
**Spearman's ρ** — how similarly two models *rank* the 30 target × block cells. 1.00
means identical ranking, 0 means unrelated. Red is high agreement.

**Right panel.** Every cell of the grid on its own row, sorted, with one coloured dot
per model. The dashed vertical line is zero.

### How to read it

|  | elastic_net | linear_svm | sparse_plsda | xgboost |
|---|---|---|---|---|
| elastic_net | 1.00 | **0.87** | 0.75 | 0.47 |
| linear_svm | 0.87 | 1.00 | 0.73 | 0.50 |
| sparse_plsda | 0.75 | 0.73 | 1.00 | 0.53 |
| xgboost | 0.47 | 0.50 | 0.53 | 1.00 |

Two things at once:

1. The three **linear-family** models agree strongly (0.73–0.87). They find the same
   structure.
2. **XGBoost, a boosted decision tree, agrees much less** (0.47–0.53). It looks for a
   fundamentally different kind of pattern.

That second point is what makes the agreement meaningful. If all four were at 0.95
you would have four views of one fit. They are not.

On the right panel, the **rows at the top — mechanism with morphology, with
morphology+chemistry, with expression — have all four dots right of zero.** Those are
the results to quote.

### Say this

> "The obvious worry is that we got lucky with one model. So: four model families,
> and here is how similarly they rank all thirty combinations. The three linear ones
> agree at 0.73 to 0.87. The boosted tree agrees much less, around 0.5 — which is the
> point. It is looking for a different kind of pattern. When methods that disagree
> about *how* to find structure agree about *where* it is, that is worth more than a
> p-value from any one of them. And on the right, the mechanism rows are the ones
> where all four dots sit right of zero."

---

## Figure 04 — The controls, drawn against their nulls

![Tier 1 controls](../assets/figures/ws4a-controls.png)

### What this slide is for

It shows the audience you tried to break your own results — and that one attempt
succeeded and you kept the answer.

### What you are looking at

Three panels, three separate arguments.

**Left — AJIVE joint rank.** AJIVE is a method that finds structure shared between
morphology and expression, and reports how many shared "components" exist. The grey
histogram is what the same method finds after we **destroy the pairing** between
compounds — feeding it morphology from compound A and expression from compound B. The
red line is what it found on the real data.

**Middle — permutation CCA.** CCA finds the single direction along which morphology
and expression agree most. Grey histogram is the test statistic under destroyed
pairing; green line is the real value.

**Right — plain vs adjusted RV.** RV is a correlation between two whole tables. Four
bars per modality pair: dark red = plain RV on real data, pale red = plain RV **on
random noise of the same shape**, dark blue = adjusted RV on real data, pale blue =
adjusted RV on noise.

### How to read it

**Left: our own finding died here.** The real joint rank is 4 — and the null reaches
4 too. p = 0.095. The shared structure AJIVE reports is not distinguishable from what
it finds in data with no correspondence at all. **We do not report "four joint
components."**

**Middle: this one survived.** Observed 3.59, well beyond the bulk of the null.
p = 0.017.

**Right: this is the most damning panel on the slide.**

| pair | plain RV, real | plain RV, **random noise** |
|---|---|---|
| morphology ~ chemistry | 0.2300 | **0.2313** |
| expression ~ chemistry | 0.2779 | **0.2744** |
| morphology ~ expression | 0.0855 | **0.2449** |

Plain RV scores random noise **as high as the real data — higher on two of three
pairs.** The pale blue bars (adjusted RV on noise) are invisible because they are
~0.000. Plain RV is measuring the *shape* of the tables, not their agreement.

### Say this

> "We ran every statistic on two deliberately broken versions of the data: compound
> pairing scrambled, and pure random noise. Left panel: a method called AJIVE told us
> there were four shared components between morphology and expression. Then we
> destroyed the pairing and it still found four. p = 0.095. So we do not report that
> result — the control killed it, and we kept the control's answer. Middle: the
> canonical correlation did survive, p = 0.017. Right: this is the one to take home.
> The plain RV coefficient scores random noise as high as our real data — higher on
> two of the three pairs. The blue bars are the bias-adjusted version, which reads
> essentially zero on noise. If you see a plain RV in a talk, ask what it scores on
> noise."

### What NOT to say

Never quote **r₁ = 0.903** for the CCA. The 95th percentile of its own null is
**0.902**. The pair is significant only through the pooled statistic. Quote p = 0.017
and never the correlation.

---

## Figure 06 — What defines the shared axis

![Cross-modal features](../assets/figures/ws4a-crossmodal-features.png)

*(Show figure 05, the embedding, first if you have time — it places each compound on
the shared axis and shows the same plot with the pairing destroyed.)*

### What you are looking at

We built the single direction along which morphology and expression agree most, then
asked: **which individual measurements and genes define it?**

Three panels — morphology, expression, chemistry. Each bar is one feature; its length
is that feature's correlation with the shared axis. **Red dashed lines are the
threshold**, and they are the whole point.

### Where the threshold comes from — explain this, it is the interesting part

With 41,780 genes and 119 compounds, the largest correlation you can get **from pure
noise** is enormous, because the axis was *fitted* to maximise agreement and can align
with almost any single feature by chance.

So the threshold is measured, not chosen: destroy the compound correspondence, rebuild
the entire axis from scratch, record the single largest correlation anywhere in that
block, and repeat 200 times. The line is the 95th percentile. Anything below it is
**large and not evidence** — those are the grey bars.

### How to read it

| block | features above the threshold | threshold |
|---|---|---|
| **morphology** | **14 of 636** | ±0.921 |
| expression | **0 of 41,780** | ±0.908 |
| chemistry (ECFP) | **0 of 1,024** | ±0.981 |

The 14 morphology survivors are **coherent, not a grab bag**: nuclear DNA texture
(three angular-second-moment measurements at the same scale differing only in angle —
a chromatin-homogeneity readout), cell and cytoplasm shape (Zernike moments), DNA–ER
co-localisation, AGP granularity. A chromatin-organisation and nuclear-morphology
phenotype.

The middle panel is a wall of grey bars with names like `ARHGEF7-AS1` and `MARK2P14`.
Those look like a top-gene list. **Not one of them clears the threshold.**

### Say this

> "We built the single direction where morphology and gene expression agree most, then
> asked what defines it. The red lines are a threshold we measured rather than chose:
> scramble the compounds, rebuild the whole axis, note the biggest correlation
> anywhere, do it 200 times, take the 95th percentile. On the morphology side, 14 of
> 636 features clear it — and they are coherent: nuclear DNA texture, cell shape,
> DNA–ER co-localisation. A chromatin and nuclear-shape phenotype. On the gene side:
> zero of 41,780. Look at that middle panel — those grey bars are a perfectly
> plausible top-gene list, and every one of them is inside what noise produces. The
> transcriptional half of this axis is real but diffuse — many genes each contributing
> a little, no drivers. If anyone shows you a top-20 gene list from data this shape,
> ask what their null was."

### Why this slide matters politically

It **supports** the main WS4 deck's own pathway-analysis slide, which found 0 gene
pairs at BH q<0.05 and a pathway null at p = 0.857. Two independent methods, same
conclusion. Present it as reinforcement, not correction.

---

## Figure 10 — What HVG selection costs

![HVG experiment](../assets/figures/ws4a-hvg-experiment.png)

### What this is for

The main WS4 deck selects 2,000 highly variable genes once, on all compounds, then
cross-validates — and flags the leakage in its own caveat. **Two different things are
bundled in that sentence**, and this figure separates them.

- **Reduction** — going from 41,780 genes to 2,000 throws information away.
- **Leakage** — choosing *which* 2,000 using all compounds, including the held-out
  ones, lets the test set influence its own feature list.

### What you are looking at

**Left panel.** Three groups of bars, one group per way of handling genes:

1. **all genes (41,780)** — no selection
2. **2,000 HVGs chosen in-fold** — selected separately inside each training fold, so
   held-out compounds never influence the choice (leakage-free)
3. **2,000 HVGs chosen on all rows** — the deck's approach

Colours are the four models. The **brown dotted line is the best morphology arm
(+0.233)** — the comparison the whole experiment exists to make. **✗ marks XGBoost**,
which was degenerate in every arm (predicted one class every fold) and is therefore
not a measurement.

**Right panel.** The two effects separated, per model. Blue = cost of reduction. Red =
worth of leakage.

### How to read it

| model | all genes | 2k in-fold | 2k all-rows | reduction | leakage |
|---|---|---|---|---|---|
| linear_svm | **+0.309** | +0.027 | +0.053 | **−0.281** | +0.026 |
| elastic_net | **+0.248** | −0.025 | −0.007 | **−0.273** | +0.017 |
| sparse_plsda | +0.075 | −0.062 | −0.063 | −0.137 | −0.001 |

Left panel: with **all genes**, two models sit **above the brown line** — expression
beats the best morphology arm. With **either** 2,000-gene version, nothing comes close.

Right panel: the blue bars are enormous, the red bars are tiny. **Reduction costs
about sixteen times what the leakage was worth** (−0.273 vs +0.017 median).

### Say this

> "The main deck selects 2,000 highly variable genes before cross-validating, and
> flags the leakage in its own caveat. We separated the two things that caveat
> bundles. Blue is what throwing genes away costs. Red is what the leakage was
> actually worth. Reduction costs about sixteen times more. And look at the brown
> line — that is the best morphology arm. With all 41,780 genes, expression clears it.
> With 2,000 HVGs it does not come close. So the workstream's finding that gene
> expression doesn't beat morphology is a property of the gene selection, not of the
> modality. Our two pipelines agree once the same genes are used."

### How to present this without stepping on colleagues

This **reconciles** two analyses rather than overturning one. Say it that way:

- The deck's **fusion** conclusion is unaffected — we independently agree (our fusion
  gain is +0.005, theirs −0.096; both are "no help").
- The deck's **"morphology is the strongest arm"** turns out to depend on the HVG
  step, which its own caveat slide already flagged as exploratory.

Lead with the agreement, then the reconciliation.

### Caveat to state

This ran on mechanism of action only, one target, untuned — which is why the
morphology reference is the untuned +0.233 and not the tuned +0.267. Everything in
the figure is like-for-like.

---

## Figure 07 — The tuning audit (backup, unless asked)

![Tuning audit](../assets/figures/ws4a-tuning-audit.png)

### What you are looking at

Three panels answering "did hyper-parameter tuning buy real improvement, or just
better noise-fitting?"

**Left.** Histogram of the change in score **on shuffled labels** between the untuned
and tuned runs. Those labels contain nothing, so any rightward shift would be pure
manufactured performance. Red line is the median: **−0.0003**.

**Middle.** Change in honest gap per model, one dot per combination, black bar at the
median.

**Right.** Count of **degenerate** models — ones scoring exactly 0.500 with zero
variation, meaning they predicted the same class every single fold. That is a broken
configuration, not a null result. **10 in the untuned run, 0 in the tuned run.**

### The three conclusions, and only one is a win

1. **No bias was bought.** Shuffled scores did not move; zero of 120 rows earned the
   `bias` verdict. At 20 trials the search is small enough to stay honest.
2. **No new findings.** The number of results clearing zero was **56 before and 56
   after**. Tuning moved numbers; it did not create discoveries.
3. **Ten broken models were fixed.** All ten were XGBoost on the two questions with the
   fewest positive examples.

### Say this

> "We also asked whether hyper-parameter tuning helped. Left panel: the change in
> score on shuffled labels — median −0.0003. The search manufactured nothing, which is
> the honest answer to 'did you just tune until it looked good'. Middle: modest gains.
> Right: it fixed ten models that had been refusing to guess. But the number of
> results clearing zero was 56 before and 56 after — tuning did not create a single
> finding."

---

## Figure 07b — Which combinations tuning actually moved (backup)

![Tuning biggest movers](../assets/figures/ws4a-compare-gap-slopes-top25.png)

The 25 largest movers of the 120. Each row is one target × block × model; the
**hollow** circle is the untuned gap, the **filled** one is tuned, and the line runs
between them. Green = the honest gap grew, red = it shrank, grey = it barely moved.

The block of green rows with hollow circles sitting **exactly on zero** — five
XGBoost rows — are the degenerate models being repaired. They were not "no effect";
they were a model that refused to guess, and tuning gave them a working
configuration.

**There is a full 120-row version** (`06_supporting/compare_gap_slopes_full_audit.png`)
showing every comparison. It is an audit trail, not a slide: at 120 rows nothing is
legible on a projector. Use it if someone asks to see everything, or on a poster
where people can walk up to it.

**Why neither version is in the main flow:** figure 07 makes the same three points in
one glance, and the honest headline from all of this is "tuning changed nothing that
mattered" (56 → 56 results clearing zero). A null result does not deserve two slides
in a fifteen-minute talk.

## Figure 09 — How certain is each gap (backup)

![Gap uncertainty](../assets/figures/ws4a-gap-uncertainty.png)

Every gap with an approximate 95% interval; green where the interval excludes zero.

**State the caveat if you show it.** The intervals are **anti-conservative** — CV
folds share training data, so they are not independent, and there is no unbiased
estimator of the variance of k-fold cross-validation (Bengio & Grandvalet 2004). The
true intervals are wider than drawn. This is why the model-consensus counts in figure
01 are the better evidence. Read these as "is this near zero", never as a p-value, and
never to claim one result beats another.

---

## Three things to raise with the team privately, not on stage

Version 1 of the main deck added a representation-comparison slide, per-drug
concordance rankings and a gene-level mapping slide. Each contains a claim that would
benefit from the same control you already applied to your own work. **Raise these
before the talk, as questions rather than corrections.** A public challenge to a
colleague's slide costs you more than it gains, and all three are easy fixes.

**1. "Averaging PCs improves Mantel-r, suggesting that population level variability is
noise rather than biologically relevant."**
The inference runs from one metric moving to a claim about biology. Averaging also
reduces noise *mechanically* — averaging any set of noisy vectors raises the
correlation between them, signal or not. The question to ask: *what does the same
comparison give when the drug correspondence is scrambled?* If averaging raises the
scrambled Mantel-r too, the improvement is arithmetic rather than biological.

**2. The per-drug concordance rankings** (Ciclopirox r = .509, Clofarabine .457,
Bergenin .452, Olaparib .429 …).
These are the top of a ranked list of 119 per-drug correlations. The top of *any*
ranked list of 119 correlations looks impressive — that is what ranking does. The
question: *what does the top of the scrambled list look like?* If the scrambled top
drug reaches r ≈ 0.45, the observed table is a ranking artefact. If it reaches 0.2,
the finding stands and is much stronger for having been checked. The
"Olaparib is concordant in both cell lines" observation is the interesting one and
deserves that check most.

**3. "Top strength gene: ENSG00000234796, ρ = −.386, q = .029"** sitting on the same
slide as "0 / 1.27 M feature–gene pairs at FDR < .05".
A single BH survivor out of roughly 40,000 tests is fragile — at q = .029 you expect
some survivors by construction — and it reads oddly next to a categorical negative on
the same slide. A max-statistic null (scramble, recompute, record the largest statistic
anywhere, repeat) would settle whether it is real. Our shared-axis analysis ran exactly
that test on the same expression matrix and **0 of 41,780 genes survived**, which is
the backdrop worth mentioning when raising it.

None of these change the deck's conclusions. All three make them harder to attack.

## The claims table — print this and keep it next to you

| ✅ you may say | ❌ never say |
|---|---|
| "Morphology adds +0.143 over chemistry for mechanism, 4/4 models" | "Morphology predicts mechanism" (without the chemistry comparison) |
| "Morphology adds nothing for the five toxicity endpoints" | "Imaging predicts toxicity" |
| "The modalities agree weakly; adjusted RV 0.016, Mantel p=0.0001" | any **plain RV** number |
| "The CCA pair is significant, p = 0.017" | "**r = 0.903**" |
| "Joint rank is not distinguishable from chance, p = 0.095" | "**four joint components**" |
| "14 of 636 morphology features define the shared axis" | any **gene list** from this data |
| "Tuning manufactured no measurable bias" | "tuning improved the models" |
| "Fusion does not help — we agree with the main deck" | "fusion improves prediction" |
| HepG2 results | anything about **A549** (its morphology is contaminated) |
| "a degenerate row means the model refused to guess" | quoting **0.500 ± 0.000** as a null result |

## If you only remember one sentence

> **Every number we show is printed next to the same number computed on data where
> the answer is known to be nothing — and several of those controls changed our
> conclusions.**

That is the contribution, and at 119 compounds it is worth more than another decimal
place of accuracy.
