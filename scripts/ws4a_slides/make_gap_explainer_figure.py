#!/usr/bin/env python3
"""Where one row of the concordance plot comes from — the whole chain, drawn.

Layout is on an explicit two-row grid with stated bounds, because the first version
collided in six places. Row 1 is y 38..64, row 2 is y 2..34; nothing may cross 36.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = sys.argv[1]
BG, INK, MUT = "#fcfcfb", "#1a1a1a", "#75756f"
BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#b02418"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # matches the plot's legend

fig, ax = plt.subplots(figsize=(13.33, 7.0))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")


def txt(x, y, s, fs=8.4, c=INK, w="normal", ha="left", va="top", it=False, ls=1.55):
    ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va,
            style="italic" if it else "normal", linespacing=ls)


def arrow(x1, y1, x2, y2, c=MUT, lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=c, lw=lw, shrinkA=4, shrinkB=4))


def step(n, x, y, title, fs=10.5):
    ax.add_patch(Circle((x, y), 1.35, fc=BLUE, ec="none"))
    ax.text(x, y, str(n), color="white", fontsize=9.5, weight="bold",
            ha="center", va="center")
    ax.text(x + 2.4, y, title, fontsize=fs, color=INK, weight="bold", va="center")


txt(50, 69.4, "Where ONE ROW of that plot comes from", 15.5, INK, "bold", ha="center")

# ══════════ ROW 1  (y 38 .. 64) ══════════════════════════════════════════════
# ---- step 1: the 30 combinations
step(1, 2.5, 63.0, "A combination  =  one question × one data type")
qs = ["mechanism", "liver tox", "kidney tox", "lung tox", "heart tox", "fertility"]
ds = ["chemistry", "photo", "genes", "photo\n+chem", "photo\n+genes"]
x0, cw, ch = 12.0, 5.6, 2.15
top = 57.0
for j, d in enumerate(ds):
    txt(x0 + j * cw + (cw - 0.5) / 2, top + 3.4, d, 7.0, MUT, ha="center", ls=1.25)
for i, q in enumerate(qs):
    yy = top - i * ch
    txt(x0 - 0.9, yy + (ch - 0.4) / 2, q, 7.4, MUT, ha="right", va="center")
    for j in range(len(ds)):
        hit = (i == 0 and j == 1)
        ax.add_patch(Rectangle((x0 + j * cw, yy), cw - 0.5, ch - 0.4,
                               fc=RED if hit else "#e6e6e3",
                               ec="#333" if hit else "none", lw=1.5 if hit else 0))
txt(2.5, 42.6, "6 questions × 5 data types  =  30 combinations",
    9, INK, "bold")
txt(2.5, 40.0, "= 30 rows in the plot", 9, INK, "bold")
txt(2.5, 37.4, "we follow the red one:  mechanism, from the photograph",
    8.4, RED, it=True)

# ---- step 2: the four recipes
step(2, 49.0, 63.0, "Four different recipes try that one combination")
names = ["elastic net", "linear SVM", "sparse PLS-DA", "XGBoost"]
what = ["draws a line and\nignores most\nmeasurements",
        "draws the line with\nthe widest possible\ngap",
        "draws a line from a\nchosen few\nmeasurements",
        "no line at all — a\nflowchart of yes/no\nquestions"]
for k, (nm, wt) in enumerate(zip(names, what)):
    bx = 49.5 + k * 12.6
    ax.add_patch(FancyBboxPatch((bx, 48.5), 11.4, 9.2, boxstyle="round,pad=0.3",
                                fc="white", ec=PAL[k], lw=1.8))
    txt(bx + 5.7, 56.4, nm, 8.4, PAL[k], "bold", ha="center")
    txt(bx + 5.7, 53.6, wt, 7.0, MUT, ha="center", ls=1.45)
txt(49.5, 45.8, "Three draw a dividing line in different ways.  The fourth does not "
                "draw a line at all.\nSo if all four agree, the result is not a quirk of "
                "one method.", 8.4, INK, ls=1.7)

ax.plot([1.5, 98.5], [34.8, 34.8], color="#dcdcd8", lw=1.1)

# ══════════ ROW 2  (y 2 .. 34) ═══════════════════════════════════════════════
# ---- step 3: run twice
step(3, 2.5, 33.0, "Each recipe is run TWICE", 10)
ax.add_patch(FancyBboxPatch((3.0, 22.6), 25.5, 7.4, boxstyle="round,pad=0.3",
                            fc="#eef7f1", ec=GREEN, lw=1.7))
txt(4.4, 29.1, "REAL  —  true answers", 8.8, GREEN, "bold")
txt(4.4, 26.9, "hide a fifth of the drugs, learn from the rest,\nguess the hidden ones — 25 times over",
    7.6, INK, ls=1.5)
ax.add_patch(FancyBboxPatch((3.0, 13.6), 25.5, 7.4, boxstyle="round,pad=0.3",
                            fc="#fdeeec", ec=RED, lw=1.7))
txt(4.4, 20.1, "SHUFFLED  —  answers randomised", 8.8, RED, "bold")
txt(4.4, 17.9, "drug A is given drug B's answer, so nothing is\nleft to learn — whatever it scores is LUCK",
    7.6, INK, ls=1.5)

# ---- step 4: subtract
step(4, 33.0, 33.0, "Subtract:  real − shuffled", 10)
rows = [("elastic net", 72.3, 48.5, 0.238), ("linear SVM", 74.5, 47.7, 0.267),
        ("sparse PLS-DA", 69.2, 48.2, 0.210), ("XGBoost", 63.9, 51.7, 0.123)]
cx = [34.0, 46.0, 53.5, 62.0]
txt(cx[0], 29.2, "recipe", 7.8, MUT, "bold")
txt(cx[1], 29.2, "real", 7.8, GREEN, "bold")
txt(cx[2], 29.2, "shuffled", 7.8, RED, "bold")
txt(cx[3], 29.2, "gap", 7.8, INK, "bold")
for k, (nm, r, sh, g) in enumerate(rows):
    yy = 26.6 - k * 2.3
    txt(cx[0], yy, nm, 7.8, PAL[k], "bold")
    txt(cx[1], yy, f"{r:.1f} %", 7.8)
    txt(cx[2], yy, f"{sh:.1f} %", 7.8)
    txt(cx[3], yy, f"+{g:.3f}", 7.8, INK, "bold")
txt(34.0, 16.0, "the shuffled runs are 47.7–51.7 %, NOT exactly 50 % —\n"
                "which is why each box is compared against its OWN\n"
                "control rather than against a nominal 0.5",
    7.4, MUT, it=True, ls=1.6)

# ---- step 5: the row
step(5, 70.0, 33.0, "Those four gaps ARE one row", 10)
ax.add_patch(Rectangle((72.0, 24.0), 26.0, 5.4, fc="#e8f4ee", ec="none"))
zero = 75.5
ax.plot([zero, zero], [23.6, 29.8], color=RED, lw=1.8, ls="--")
for k, (_, _, _, g) in enumerate(rows):
    ax.add_patch(Circle((zero + g * 68, 26.7), 0.9, fc=PAL[k], ec="#333", lw=0.6))
txt(zero, 22.9, "0", 7.6, RED, ha="center")
txt(98, 22.9, "honest gap →", 7.6, MUT, ha="right")
txt(72.0, 31.2, "mechanism · photograph", 8.4, "#12805a", "bold")
txt(72.0, 20.4, "All four dots sit RIGHT of the red line, so this row\n"
                "is shaded green and its label is bold — quotable.\n\n"
                "14 of the 30 rows pass that test.  In the other 16,\n"
                "at least one recipe failed, so those labels stay grey.",
    8.0, INK, ls=1.7)

arrow(29.2, 22.0, 32.5, 22.0)
arrow(66.5, 26.6, 71.0, 26.6)

ax.text(50, 1.2, "30 combinations × 4 recipes × 2 label conditions × 25 train-test rounds "
                 "  ≈  6,000 model fits",
        fontsize=8.0, ha="center", va="center", color=MUT)

fig.savefig(OUT, dpi=200, facecolor=BG, bbox_inches="tight")
print(OUT)
