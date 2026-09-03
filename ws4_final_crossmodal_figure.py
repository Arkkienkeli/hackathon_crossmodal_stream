#!/usr/bin/env python3

"""
FINAL presentation figure:
OpenScreen HepG2 morphology ↔ plate-corrected TAHOE HepG2 gene expression

Analysis:
- 119 matched drugs
- Morphology: 636 features
- Gene expression: 2000 HVGs
- PCA separately within each modality: 30 PCs
- z-score PCs across drugs
- Euclidean drug-drug distances
- Spearman correlation between distance matrices
- 999 drug-label permutations

Expected final result:
r ≈ 0.370
permutation p = 0.001
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ============================================================
# FILES
# ============================================================

MORPH_FILE = Path(
    "OpenScreen/data/hepg2_morphology_final.parquet"
)

GEX_FILE = Path(
    "hepg2_platecorrected_drug_2000hvg.parquet"
)

OUT_DIR = Path(
    "OpenScreen/presentation_figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_PNG = OUT_DIR / (
    "06_FINAL_crossmodal_distance_platecorrected.png"
)

OUT_PDF = OUT_DIR / (
    "06_FINAL_crossmodal_distance_platecorrected.pdf"
)


# ============================================================
# SETTINGS
# ============================================================

N_PCS = 30
N_PERM = 999
RANDOM_STATE = 0


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FINAL CROSS-MODAL FIGURE")
print("=" * 80)

M = pd.read_parquet(
    MORPH_FILE
)

G = pd.read_parquet(
    GEX_FILE
)

M.index = M.index.astype(str)
G.index = G.index.astype(str)


shared = sorted(
    set(M.index)
    &
    set(G.index)
)

M = M.loc[
    shared
].copy()

G = G.loc[
    shared
].copy()


print(
    "Morphology:",
    M.shape
)

print(
    "Gene expression:",
    G.shape
)

print(
    "Shared drugs:",
    len(shared)
)


assert len(shared) == 119


# ============================================================
# PCA — SEPARATELY FOR EACH MODALITY
# ============================================================

pca_m = PCA(
    n_components=N_PCS,
    random_state=RANDOM_STATE
)

pca_g = PCA(
    n_components=N_PCS,
    random_state=RANDOM_STATE
)


Mpc = pca_m.fit_transform(
    M.to_numpy(
        dtype=float
    )
)

Gpc = pca_g.fit_transform(
    G.to_numpy(
        dtype=float
    )
)


# ============================================================
# Z-SCORE EACH PC ACROSS DRUGS
# ============================================================

def zscore_columns(X):

    X = np.asarray(
        X,
        dtype=float
    )

    mu = X.mean(
        axis=0
    )

    sd = X.std(
        axis=0,
        ddof=0
    )

    sd[
        sd == 0
    ] = 1.0

    return (
        X - mu
    ) / sd


Mz = zscore_columns(
    Mpc
)

Gz = zscore_columns(
    Gpc
)


# ============================================================
# DRUG × DRUG EUCLIDEAN DISTANCES
# ============================================================

Dm = squareform(
    pdist(
        Mz,
        metric="euclidean"
    )
)

Dg = squareform(
    pdist(
        Gz,
        metric="euclidean"
    )
)


# Upper triangle = unique drug pairs
iu = np.triu_indices(
    len(shared),
    k=1
)

dm = Dm[
    iu
]

dg = Dg[
    iu
]


n_pairs = len(
    dm
)


# ============================================================
# OBSERVED CROSS-MODAL CORRELATION
# ============================================================

r_obs = float(
    spearmanr(
        dm,
        dg
    ).statistic
)


print()
print(
    "Observed Spearman r:",
    f"{r_obs:.6f}"
)

print(
    "Drug pairs:",
    n_pairs
)


# ============================================================
# DRUG-LABEL PERMUTATION TEST
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

null_r = np.zeros(
    N_PERM,
    dtype=float
)


for b in range(
    N_PERM
):

    perm = rng.permutation(
        len(shared)
    )

    # Shuffle drug identity in expression modality
    Dg_perm = Dg[
        np.ix_(
            perm,
            perm
        )
    ]

    dg_perm = Dg_perm[
        iu
    ]

    null_r[
        b
    ] = spearmanr(
        dm,
        dg_perm
    ).statistic


# one-sided test:
# is observed positive correspondence stronger than shuffled labels?
p_perm = (
    1
    +
    np.sum(
        null_r
        >=
        r_obs
    )
) / (
    N_PERM
    +
    1
)


print(
    "Permutation p:",
    f"{p_perm:.4f}"
)

print(
    "Null mean:",
    f"{null_r.mean():.4f}"
)

print(
    "Null SD:",
    f"{null_r.std():.4f}"
)


# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 22,
    "axes.labelsize": 16,
})


fig, ax = plt.subplots(
    figsize=(10.5, 8.5)
)


hb = ax.hexbin(
    dm,
    dg,
    gridsize=42,
    mincnt=1,
    cmap="viridis"
)


cbar = fig.colorbar(
    hb,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "Number of drug pairs",
    fontsize=14
)


ax.set_xlabel(
    "OpenScreen morphology\n"
    "drug–drug Euclidean distance"
)

ax.set_ylabel(
    "TAHOE gene expression\n"
    "drug–drug Euclidean distance"
)


ax.set_title(
    "Morphology and gene expression show\n"
    "a reproducible drug-level relationship",
    pad=18
)


# Main result box
result_text = (
    f"119 matched HepG2 drugs\n"
    f"{n_pairs:,} unique drug pairs\n\n"
    f"Spearman r = {r_obs:.3f}\n"
    f"Permutation p = {p_perm:.3f}"
)


ax.text(
    0.045,
    0.95,
    result_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    bbox=dict(
        boxstyle="round,pad=0.55",
        facecolor="white",
        edgecolor="0.7",
        alpha=0.92
    )
)


# Interpretation underneath plot
fig.text(
    0.50,
    0.015,
    (
        "Drug pairs that differ more morphologically also tend to "
        "differ more transcriptionally."
    ),
    ha="center",
    fontsize=13
)


plt.tight_layout(
    rect=[
        0,
        0.045,
        1,
        1
    ]
)


plt.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close()


print()
print(
    "Saved PNG:",
    OUT_PNG
)

print(
    "Saved PDF:",
    OUT_PDF
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)

