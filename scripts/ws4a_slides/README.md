# WS4A slide deck

Regenerate after any figure changes:

```bash
PACK=<stream>/WS4A_presentation_pack
python scripts/ws4a_slides/make_method_figure.py "$PACK/00_THE_THREE_SLIDES/SLIDE_0_how_it_was_trained.png"
python scripts/ws4a_slides/build_deck.py "$PACK" "$PACK/WS4A_slides.pptx"
```

Needs `python-pptx` and `pillow`, which are NOT in container/ws4a.sif — the deck is a
deliverable, not part of the analysis, so the image stays lean. Use any environment
with those two installed.

Slide layout is fixed by `add_slide()`: kicker, say-this headline, figure scaled to fit
12.45 x 4.35 in preserving aspect ratio, plain-language paragraph, and the full
speaking script plus Q&A in the speaker notes.

## The slides carry their own explanation

The deck is presented by someone who did not run the analysis, so each slide holds:

- a **kicker** (which result this is),
- a **say-this headline** — one sentence, the claim,
- the figure at `fig_h` inches (deliberately modest; the presenter resizes it by hand,
  but they cannot invent the explanation), and
- **labelled blocks** — `HOW TO READ IT`, `WHAT IT SHOWS`, `THE POINT` — rendered by
  `blocks_box()` with the label in accent colour.

Speaker notes are still the full script plus Q&A, but nothing essential lives only
there.

**Verify rendering after any text change**, because a text box does not clip and
python-pptx cannot measure wrapped height:

```bash
soffice --headless --convert-to pdf --outdir /tmp "$PACK/WS4A_slides.pptx"
pdftoppm -png -r 70 /tmp/WS4A_slides.pdf /tmp/slide   # then look at each one
```
