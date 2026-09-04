#!/usr/bin/env python3
"""'What was trained and how' — drawn, not written."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, Rectangle, RegularPolygon,
                                FancyArrowPatch, FancyBboxPatch)

OUT = sys.argv[1]
BG, INK, MUT = "#fcfcfb", "#1a1a1a", "#75756f"
BLUE, GREEN, ORANGE, RED, PURPLE = "#2a78d6", "#1baf7a", "#d1571f", "#b02418", "#4a3aa7"

fig, ax = plt.subplots(figsize=(13.33, 7.0))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")


def txt(x, y, s, fs=9, c=INK, w="normal", ha="left", va="top", it=False, ls=1.5):
    ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va,
            style="italic" if it else "normal", linespacing=ls)


def arrow(x1, y1, x2, y2, c=MUT, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=c, lw=lw,
                                 shrinkA=3, shrinkB=3))


# ── icons ────────────────────────────────────────────────────────────────────
def icon_molecule(cx, cy, r=2.7):
    # bonds point up and sideways only -- a downward one collides with the cell icon
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=r, fc="none", ec=PURPLE, lw=2))
    for ang in (35, 145):
        x2 = cx + (r + 1.5) * np.cos(np.radians(ang))
        y2 = cy + (r + 1.5) * np.sin(np.radians(ang))
        ax.plot([cx + r * np.cos(np.radians(ang)), x2],
                [cy + r * np.sin(np.radians(ang)), y2], color=PURPLE, lw=1.8)
        ax.add_patch(Circle((x2, y2), 0.62, fc=PURPLE, ec="none"))


def icon_cell(cx, cy, r=3.9):
    ax.add_patch(Ellipse((cx, cy), r * 2, r * 1.75, fc="#eaf3ec", ec=GREEN, lw=2))
    ax.add_patch(Ellipse((cx - 0.5, cy + 0.2), r * 0.85, r * 0.78,
                         fc="#bcd9e8", ec="#2a78d6", lw=1.4))
    rng = np.random.default_rng(4)
    for col in ("#d94f2b", "#e8a33d", "#1baf7a"):
        for _ in range(4):
            a, rad = rng.uniform(0, 2 * np.pi), rng.uniform(1.9, r * 0.88)
            ax.add_patch(Circle((cx + rad * np.cos(a) * 1.05,
                                 cy + rad * np.sin(a) * 0.82), 0.42,
                                fc=col, ec="none", alpha=0.95))


def icon_genes(cx, cy, nr=6, nc=9, cw=0.78):
    rng = np.random.default_rng(1)
    x0, y0 = cx - nc * cw / 2, cy - nr * cw / 2
    for i in range(nr):
        for j in range(nc):
            v = rng.random()
            col = plt.get_cmap("RdBu_r")(0.5 + (v - 0.5) * 0.95)
            ax.add_patch(Rectangle((x0 + j * cw, y0 + i * cw), cw * 0.88, cw * 0.88,
                                   fc=col, ec="none"))


def icon_compound(cx, cy):
    """A capsule, tilted -- reads as 'a drug' at a glance."""
    from matplotlib.transforms import Affine2D
    tr = Affine2D().rotate_deg_around(cx, cy, -28) + ax.transData
    ax.add_patch(FancyBboxPatch((cx - 2.9, cy - 1.25), 5.8, 2.5,
                                boxstyle="round,pad=0,rounding_size=1.25",
                                fc="white", ec=INK, lw=1.8, transform=tr))
    ax.add_patch(FancyBboxPatch((cx - 2.9, cy - 1.25), 2.9, 2.5,
                                boxstyle="round,pad=0,rounding_size=1.25",
                                fc="#d94f2b", ec=INK, lw=1.8, transform=tr))


# ═══════════════ THE QUESTION ═══════════════
ax.add_patch(FancyBboxPatch((1.5, 59.5), 97, 9.5, boxstyle="round,pad=0.5",
                            fc="#eef3fa", ec=BLUE, lw=2))
txt(50, 66.6, "Can we tell what a drug does — from a photograph of the cells it was put on?",
    16.5, INK, "bold", ha="center", va="center")
txt(50, 62.2,
    "and does that photograph beat the drug's chemical structure, which is free?",
    12.5, BLUE, "bold", ha="center", va="center")

# ═══════════════ 119 COMPOUNDS → 3 DESCRIPTIONS ═══════════════
# vertical budget: banner ends 59.5, divider at 33.0.  Icons sit at 53 / 45 / 37.5,
# each with its own label to the right, and nothing may cross either boundary.
icon_compound(8, 45)
txt(8, 40.4, "119\ncompounds", 9.5, INK, "bold", ha="center", ls=1.4)
for ytgt in (53.0, 45.0, 37.5):
    arrow(12.4, 45, 20.4, ytgt, MUT, 1.4)

icon_molecule(25, 53.0)
txt(31, 54.6, "chemical structure", 11, PURPLE, "bold")
txt(31, 51.8, "1,024 numbers  ·  FREE — no experiment at all", 9, MUT)

icon_cell(25, 45.0)
txt(31, 46.6, "the photograph  (Cell Painting)", 11, GREEN, "bold")
txt(31, 43.8, "636 numbers  ·  shape, texture, where organelles sit", 9, MUT)

icon_genes(25, 37.5)
txt(31, 39.1, "gene expression", 11, RED, "bold")
txt(31, 36.3, "41,780 numbers  ·  what switched on or off", 9, MUT)

txt(72, 53.4, "…each on its own,", 9.5, MUT, it=True)
txt(72, 50.6, "and photo + chemistry\nand photo + expression", 9.5, INK, "bold", ls=1.5)
txt(72, 43.6, "5 ways of describing\na compound", 9.5, MUT, it=True, ls=1.5)

# ═══════════════ WHAT WE PREDICT ═══════════════
ax.plot([1.5, 98.5], [33.0, 33.0], color="#dcdcd8", lw=1.2)
txt(2, 31.3, "WE ASK 6 QUESTIONS OF EACH", 11, INK, "bold")
txt(2, 27.2,
    "MECHANISM   ·   is this a DNA-synthesis inhibitor?          14 yes  ·  50 no", 11)
txt(2, 23.2,
    "TOXICITY        ·   is it toxic to  heart · lung · kidney · liver · fertility?"
    "     ~70 drugs each", 11)
txt(2, 19.4, "two further endpoints refused — at 68 vs 2, no honest answer exists",
    8.4, RED, it=True)

# ═══════════════ HOW IT IS TRAINED ═══════════════
ax.plot([1.5, 98.5], [16.5, 16.5], color="#dcdcd8", lw=1.2)
txt(2, 14.8, "HOW EACH IS TRAINED", 11, INK, "bold")

rng = np.random.default_rng(0)
hidden = set(rng.choice(24, 5, replace=False))
for i in range(24):
    ax.add_patch(Circle((3.6 + i * 1.55, 9.8), 0.62,
                        fc=ORANGE if i in hidden else BLUE, ec="none"))
txt(3, 6.6, "hide a fifth of the drugs   →   train on the rest   →   guess the hidden ones",
    9, INK)
txt(3, 3.4, "rotate which are hidden, 25 times over", 8.4, MUT, it=True)

arrow(43, 9.0, 48, 9.0)
txt(50, 14.0, "every setting is chosen using the BLUE drugs only", 9.6, BLUE, "bold")
txt(50, 10.9,
    "how much to penalise  ·  how many features to keep  ·  how deep a tree may grow\n"
    "either a fixed grid, or Optuna searching 20 combinations per fold", 9)
txt(50, 5.4, "4 model families try each combination:", 9.6, INK, "bold")
txt(50, 2.4, "elastic net  ·  sparse PLS-DA  ·  linear SVM  ·  XGBoost (trees)", 9, MUT)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor=BG)
print(OUT)
