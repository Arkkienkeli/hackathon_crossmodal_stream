# The three slides

If you only get two or three, use these, in this order. Drop slide 3 first.

| slide | establishes | why it earns the space |
|---|---|---|
| **1** morphology vs chemistry — **two options, pick one** | Morphology adds +0.143 over chemical structure for **mechanism**, and nothing for toxicity | The chemistry baseline does not exist anywhere in the main WS4 deck. This is the decision-relevant comparison. |
| **2** HVG selection cost | The deck's "expression doesn't beat morphology" is a property of the **gene selection**, not the modality | Reconciles our result with the deck's Task 1. Reduction costs ~16× what the leakage it was criticised for was worth. |
| **3** four models agree | The three linear models rank the grid alike (ρ 0.73–0.87); XGBoost, a tree, much less (0.47–0.53). all 30 shown, sorted by weakest model; 14 of 30 have all four right of zero | Answers "you got lucky with one model" — and v1 of the main deck adds CCA, PLS and Wasserstein, so methods scrutiny is more likely, not less. |

*Slide 3 changed after reading v1 of the main deck: it now reports 0 of 1.27 M
feature–gene pairs at FDR < .05, which is the same conclusion our gene figure reached
by a different route. That figure is kept as `BACKUP_what_defines_shared_axis.png`.*

Full background, how to read every element, the sentences to say, and the questions
you will get: `../HOW_TO_PRESENT.md`, section **"If you only get two or three
slides"**.

## SLIDE_3b — the explainer, if anyone looks lost

`SLIDE_3b_where_a_row_comes_from.png` walks the whole chain in five numbered steps,
with real numbers rather than symbols: one combination (6 questions × 5 data types =
30) → four recipes → each run twice, true answers and randomised → subtract → those
four gaps ARE one row of the concordance plot.

The worked box is mechanism-from-photograph: **74.5 % right on the true answers,
47.7 % with the answers randomised, so +0.267**. It also shows the four randomised
runs came out at 47.7, 48.2, 48.5 and 51.7 % — never exactly 50 %, which is the
clearest answer to "why not just compare against a coin flip?"

Put it immediately before the concordance slide, or hold it in reserve.

## Slide 1: which of the two?

`SLIDE_1_morphology_vs_chemistry.png` shows **differences** — it does the subtraction
for the audience and draws a zero line. Use it when the slide must stand alone.

`SLIDE_1_ALT_signal_map.png` shows **levels** — all 30 combinations, with the
`n/4 models` consensus counts in every cell. It contains every number the other one
computes, plus the whole landscape and the built-in answer to "was it one lucky
model?". **Use this one if you are narrating live**: walk across the top row
(chemistry 0.175 → morphology 0.267 → both 0.318), then down the morphology column to
show the toxicity rows are empty.

With two slots for the finding: signal map first, then the incremental figure.

## The three numbers to memorise

- **+0.143** — what morphology adds to chemistry for mechanism of action (4/4 models)
- **−0.273 vs +0.017** — cost of dropping to 2,000 HVGs, vs what the leakage was worth
- **14 of 30** — combinations where all four models independently clear zero
  *(backup fact: 0 of 41,780 genes survive the shared-axis permutation null)*

## Say once, early, or nothing else parses

> "Every number here is an **honest gap**: how well the model did on the real answers,
> minus how well the same model did when we scrambled the answers between compounds.
> Scrambled means there is nothing left to learn, so that second number is what luck
> alone buys. Across our whole grid, luck scored between 0.38 and 0.60 — where 0.50 is
> a coin flip. That is why we never quote a raw score."
