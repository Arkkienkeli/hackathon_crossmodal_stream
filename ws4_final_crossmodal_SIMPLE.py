#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

MORPH_FILE = Path(
    "OpenScreen/data/hepg2_morphology_final.parquet"
)

GEX_FILE = Path(
    "hepg2_platecorrected_drug_2000hvg.parquet"
)

OUT = Path(
    "OpenScreen/presentation_figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PNG = OUT / "06_FINAL_crossmodal_SIMPLE.png"
PDF = OUT / "06_FINAL_crossmodal_SIMPLE.pdf"


N_PCS = 30
N_PERM = 999
RANDOM_STATE = 0
N_BINS = 10


# ------------------------------------------------------------
# LOAD / MATCH
# ------------------------------------------------------------

M = pd.read_parquet(MORPH_FILE)
G = pd.read_parquet(GEX_FILE)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

shared = sorted(
    set(M.index) & set(G.index)
)

M = M.loc[shared]
G = G.loc[shared]

assert len(shared) == 119

print("Morphology:", M.shape)
print("Expression:", G.shape)
print("Shared drugs:", len(shared))


# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

Mp = PCA(
    n_components=N_PCS,
    random_state=RANDOM_STATE
).fit_transform(
    M.to_numpy(dtype=float)
)

Gp = PCA(
    n_components=N_PCS,
    random_state=RANDOM_STATE
).fit_transform(
    G.to_numpy(dtype=float)
)


def zscore(X):

    X = np.asarray(
        X,
        dtype=float
    )

    sd = X.std(
        axis=0,
        ddof=0
    )

    sd[
        sd == 0
    ] = 1

    return (
        X - X.mean(axis=0)
    ) / sd


Mz = zscore(Mp)
Gz = zscore(Gp)


# ------------------------------------------------------------
# PAIRWISE DISTANCES
# ------------------------------------------------------------

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

iu = np.triu_indices(
    len(shared),
    k=1
)

dm = Dm[iu]
dg = Dg[iu]

n_pairs = len(dm)


# ------------------------------------------------------------
# FULL-DATA CORRELATION
# ------------------------------------------------------------

r_obs = float(
    spearmanr(
        dm,
        dg
    ).statistic
)


# ------------------------------------------------------------
# PERMUTATION
# ------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_STATE
)

null = []

for _ in range(N_PERM):

    perm = rng.permutation(
        len(shared)
    )

    Dgp = Dg[
        np.ix_(
            perm,
            perm
        )
    ]

    null.append(
        spearmanr(
            dm,
            Dgp[iu]
        ).statistic
    )

null = np.asarray(null)

p_perm = (
    1
    +
    np.sum(
        null >= r_obs
    )
) / (
    N_PERM + 1
)


print(
    f"Spearman r = {r_obs:.3f}"
)

print(
    f"Permutation p = {p_perm:.3f}"
)

print(
    "Pairs:",
    n_pairs
)


# ------------------------------------------------------------
# CREATE EQUAL-COUNT MORPHOLOGY-DISTANCE BINS
# ------------------------------------------------------------

df = pd.DataFrame({
    "morph_distance": dm,
    "ge_distance": dg
})


df["bin"] = pd.qcut(
    df["morph_distance"],
    q=N_BINS,
    labels=False,
    duplicates="drop"
)


summary = (
    df
    .groupby(
        "bin",
        observed=True
    )
    .agg(
        morph_median=(
            "morph_distance",
            "median"
        ),
        ge_median=(
            "ge_distance",
            "median"
        ),
        ge_q25=(
            "ge_distance",
            lambda x: np.quantile(
                x,
                0.25
            )
        ),
        ge_q75=(
            "ge_distance",
            lambda x: np.quantile(
                x,
                0.75
            )
        ),
        n=(
            "ge_distance",
            "size"
        )
    )
    .reset_index()
)


print()
print(summary.to_string(index=False))


summary.to_csv(
    OUT / "06_FINAL_crossmodal_SIMPLE_bins.csv",
    index=False
)


# ------------------------------------------------------------
# PRESENTATION FIGURE
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 22,
    "axes.labelsize": 17,
})


fig, ax = plt.subplots(
    figsize=(10.5, 7.5)
)


x = summary[
    "morph_median"
].to_numpy()

y = summary[
    "ge_median"
].to_numpy()

low = (
    y
    -
    summary[
        "ge_q25"
    ].to_numpy()
)

high = (
    summary[
        "ge_q75"
    ].to_numpy()
    -
    y
)


# Connecting line
ax.plot(
    x,
    y,
    linewidth=3,
    marker="o",
    markersize=10
)


# IQR bars
ax.errorbar(
    x,
    y,
    yerr=[
        low,
        high
    ],
    fmt="none",
    capsize=5,
    linewidth=2
)


ax.set_xlabel(
    "OpenScreen morphology difference\n"
    "between drug pairs"
)

ax.set_ylabel(
    "TAHOE gene-expression difference\n"
    "between the same drug pairs"
)


ax.set_title(
    "Drug pairs that look more different also\n"
    "tend to respond more differently transcriptionally",
    pad=18
)


# Main result box
ax.text(
    0.05,
    0.94,
    (
        "119 matched HepG2 drugs\n"
        "7,021 drug pairs\n\n"
        f"Spearman r = {r_obs:.3f}\n"
        f"Permutation p = {p_perm:.3f}"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="0.7",
        alpha=0.95
    )
)


# Explain what each displayed point means
ax.text(
    0.97,
    0.06,
    (
        "Each point = ~702 drug pairs\n"
        "Line shows median trend\n"
        "Bars show interquartile range"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="0.8",
        alpha=0.9
    )
)


# Helpful LOW / HIGH labels
ax.text(
    0.01,
    -0.18,
    "More similar morphology",
    transform=ax.transAxes,
    ha="left",
    fontsize=12
)

ax.text(
    0.99,
    -0.18,
    "More different morphology",
    transform=ax.transAxes,
    ha="right",
    fontsize=12
)


fig.text(
    0.5,
    0.015,
    (
        "Statistics use all 7,021 drug pairs; "
        "binning is only for visualization."
    ),
    ha="center",
    fontsize=11
)


plt.tight_layout(
    rect=[
        0,
        0.06,
        1,
        1
    ]
)

plt.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    PDF,
    bbox_inches="tight"
)

plt.close()


print()
print("Saved:", PNG)
print("Saved:", PDF)

