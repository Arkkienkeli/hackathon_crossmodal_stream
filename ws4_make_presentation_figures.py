#!/usr/bin/env python3

"""
Presentation figures for:

    OpenScreen HepG2 morphology
            ↕
    TAHOE HepG2 transcriptomics

Task 2: morphology ↔ gene-expression integration.

Outputs
-------
OPENSCREEN
  openscreen_01_cross_site_reproducibility.png
  openscreen_02_within_site_reproducibility.png
  openscreen_03_sphering_decision.png
  openscreen_04_morphology_pca.png

TAHOE
  01_tahoe_cells_per_sample.png
  02_tahoe_pseudobulk_umi.png
  03_tahoe_replicate_qc.png
  04_tahoe_drug_pca.png
  05_tahoe_clustered_expression_heatmap.png

TASK 2
  06_crossmodal_distance_relationship.png
  07_crossmodal_perturbation_magnitude.png
  08_direction_by_activity.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import anndata as ad


# ============================================================
# PATHS
# ============================================================

DATA = Path("OpenScreen/data")
SPHERE = Path("OpenScreen/sphering")

OUT = Path("OpenScreen/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

MORPH = DATA / "hepg2_morphology_final.parquet"
GEX = DATA / "hepg2_pseudobulk_2000hvg_shared119.parquet"

SAMPLE_QC = DATA / "hepg2_tahoe_sample_qc.csv"
SAMPLE_EXPR = DATA / "hepg2_sample_log1p_2000hvg.parquet"
RAW_PB = DATA / "hepg2_sample_pseudobulk_counts.h5ad"

DIRECTION_ACTIVITY = Path("direction_by_activity.csv")

CROSS_SITE_QC = DATA / "hepg2_cross_site_reproducibility.csv"
WITHIN_SITE_QC = DATA / "hepg2_within_site_reproducibility.csv"

GLOBAL_SPHERE = SPHERE / "global_sphere_final_summary.csv"

N_PCS = 30
RANDOM_STATE = 0


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "figure.titlesize": 18,
})


# ============================================================
# HELPERS
# ============================================================

def savefig(name):
    path = OUT / name

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("saved:", path)


def zscore_cols(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    mu = X.mean(axis=0)

    sd = X.std(
        axis=0,
        ddof=0
    )

    sd[sd == 0] = 1

    return (
        X - mu
    ) / sd


def first_existing_column(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    return None


def annotate_bars(ax, values, fmt=".3f"):

    for i, value in enumerate(values):

        ax.text(
            i,
            value,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=10
        )


# ============================================================
# LOAD FINAL MATRICES
# ============================================================

print("=" * 80)
print("LOADING FINAL MATRICES")
print("=" * 80)

M = pd.read_parquet(
    MORPH
)

G = pd.read_parquet(
    GEX
)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

shared = sorted(
    set(M.index)
    &
    set(G.index)
)

M = M.loc[shared]
G = G.loc[shared]

print(
    "Morphology:",
    M.shape
)

print(
    "Expression:",
    G.shape
)

print(
    "Matched drugs:",
    len(shared)
)


# ============================================================
# OPENSCREEN FIGURE 1
# CROSS-SITE REPRODUCIBILITY
# ============================================================

print("\n" + "=" * 80)
print("OPENSCREEN FIGURE 1")
print("CROSS-SITE REPRODUCIBILITY")
print("=" * 80)

cross = pd.read_csv(
    CROSS_SITE_QC
)

print(
    "Columns:",
    list(cross.columns)
)

print(
    cross.to_string(index=False)
)


pair_col = first_existing_column(
    cross,
    [
        "pair",
        "site_pair",
        "comparison",
    ]
)


# Actual file uses median_matched_r
r_col = first_existing_column(
    cross,
    [
        "median_matched_r",
        "median_r",
        "observed_median_r",
        "median_correlation",
        "matched_median_r",
        "r",
    ]
)


if pair_col is None:

    site_cols = [
        c
        for c in cross.columns
        if "site" in c.lower()
    ]

    if len(site_cols) >= 2:

        cross["site_pair_plot"] = (
            cross[
                site_cols[0]
            ].astype(str)
            +
            " vs "
            +
            cross[
                site_cols[1]
            ].astype(str)
        )

        pair_col = (
            "site_pair_plot"
        )


if pair_col is None:

    raise RuntimeError(
        "Could not identify site-pair column. "
        f"Available columns: "
        f"{list(cross.columns)}"
    )


if r_col is None:

    raise RuntimeError(
        "Could not identify matched-correlation column. "
        f"Available columns: "
        f"{list(cross.columns)}"
    )


fig, ax = plt.subplots(
    figsize=(9, 6)
)

x = np.arange(
    len(cross)
)


ax.bar(
    x,
    cross[r_col]
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    cross[pair_col],
    rotation=15,
    ha="right"
)


ax.set_ylabel(
    "Median same-drug correlation"
)

ax.set_xlabel(
    "OpenScreen site pair"
)

ax.set_title(
    "OpenScreen HepG2 cross-site reproducibility"
)


annotate_bars(
    ax,
    cross[r_col]
)


if "null_p95" in cross.columns:

    for i in range(
        len(cross)
    ):

        ax.plot(
            [
                i - 0.25,
                i + 0.25
            ],
            [
                cross.loc[
                    i,
                    "null_p95"
                ],
                cross.loc[
                    i,
                    "null_p95"
                ],
            ],
            linestyle="--",
            linewidth=2
        )


    ax.text(
        0.98,
        0.97,
        "Short dashed lines = null 95th percentile",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9
    )


savefig(
    "openscreen_01_cross_site_reproducibility.png"
)


# ============================================================
# OPENSCREEN FIGURE 2
# WITHIN-SITE REPRODUCIBILITY
# ============================================================

print("\n" + "=" * 80)
print("OPENSCREEN FIGURE 2")
print("WITHIN-SITE REPRODUCIBILITY")
print("=" * 80)

within = pd.read_csv(
    WITHIN_SITE_QC
)

print(
    "Columns:",
    list(within.columns)
)

print(
    within.to_string(index=False)
)


site_col = first_existing_column(
    within,
    [
        "site",
        "Metadata_Site",
        "Site",
        "Unnamed: 0",
    ]
)


r_col = first_existing_column(
    within,
    [
        "median_within_site_r",
        "median_matched_r",
        "median_replicate_r",
        "median_r",
        "observed_median_r",
        "median_correlation",
        "matched_median_r",
        "r",
    ]
)


# In case the site column was written unnamed/index-style
if site_col is None:

    possible = [
        c
        for c in within.columns
        if (
            "site" in c.lower()
            or "dataset" in c.lower()
        )
    ]

    if possible:
        site_col = possible[0]


if site_col is None:

    raise RuntimeError(
        "Could not identify site column in "
        f"{WITHIN_SITE_QC}. "
        f"Columns: {list(within.columns)}"
    )


if r_col is None:

    raise RuntimeError(
        "Could not identify replicate-correlation column in "
        f"{WITHIN_SITE_QC}. "
        f"Columns: {list(within.columns)}"
    )


fig, ax = plt.subplots(
    figsize=(8, 6)
)

x = np.arange(
    len(within)
)


ax.bar(
    x,
    within[r_col]
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    within[site_col]
)


ax.set_ylabel(
    "Median same-drug replicate correlation"
)

ax.set_xlabel(
    "OpenScreen site"
)

ax.set_title(
    "OpenScreen HepG2 within-site reproducibility"
)


annotate_bars(
    ax,
    within[r_col]
)


if "null_p95" in within.columns:

    for i in range(
        len(within)
    ):

        ax.plot(
            [
                i - 0.25,
                i + 0.25
            ],
            [
                within.loc[
                    i,
                    "null_p95"
                ],
                within.loc[
                    i,
                    "null_p95"
                ],
            ],
            linestyle="--",
            linewidth=2
        )


savefig(
    "openscreen_02_within_site_reproducibility.png"
)


# ============================================================
# OPENSCREEN FIGURE 3
# SPHERING DECISION
# ============================================================

print("\n" + "=" * 80)
print("OPENSCREEN FIGURE 3")
print("SPHERING DECISION")
print("=" * 80)

sphere = pd.read_csv(
    GLOBAL_SPHERE
)

print(
    "Columns:",
    list(sphere.columns)
)

print(
    sphere.to_string(index=False)
)


transform_col = first_existing_column(
    sphere,
    [
        "transform",
        "method",
        "normalization",
    ]
)


cross_r_col = first_existing_column(
    sphere,
    [
        "cross_r_mean",
        "cross_site_r_mean",
        "cross_r",
    ]
)


within_r_col = first_existing_column(
    sphere,
    [
        "within_r_mean",
        "within_site_r_mean",
        "within_r",
    ]
)


compound_nn_col = first_existing_column(
    sphere,
    [
        "same_compound_nn",
        "same_drug_nn",
        "compound_nn",
    ]
)


if transform_col is None:

    raise RuntimeError(
        "Could not identify transform column in "
        f"{GLOBAL_SPHERE}. "
        f"Columns: {list(sphere.columns)}"
    )


if (
    cross_r_col is None
    or within_r_col is None
    or compound_nn_col is None
):

    raise RuntimeError(
        "Could not identify sphering QC columns. "
        f"Columns: {list(sphere.columns)}"
    )


x = np.arange(
    len(sphere)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.bar(
    x - width,
    sphere[cross_r_col],
    width,
    label="Cross-site same-drug r"
)


ax.bar(
    x,
    sphere[within_r_col],
    width,
    label="Within-site replicate r"
)


ax.bar(
    x + width,
    sphere[compound_nn_col],
    width,
    label="Same-compound nearest-neighbour"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    sphere[transform_col],
    rotation=15,
    ha="right"
)


ax.set_ylabel(
    "Biological reproducibility / retrieval"
)

ax.set_title(
    "OpenScreen sphering reduced biological signal"
)

ax.legend(
    fontsize=9
)


savefig(
    "openscreen_03_sphering_decision.png"
)


# ============================================================
# OPENSCREEN FIGURE 4
# FINAL MORPHOLOGY PCA
# ============================================================

print("\n" + "=" * 80)
print("OPENSCREEN FIGURE 4")
print("FINAL MORPHOLOGY PCA")
print("=" * 80)

M_open = pd.read_parquet(
    MORPH
)

M_open.index = (
    M_open.index.astype(str)
)


pca_open = PCA(
    n_components=30,
    random_state=RANDOM_STATE
)


M_open_pc = (
    pca_open.fit_transform(
        M_open.to_numpy(
            dtype=np.float64
        )
    )
)


M_open_z = zscore_cols(
    M_open_pc
)


morph_activity = np.linalg.norm(
    M_open_z
    -
    M_open_z.mean(
        axis=0
    ),
    axis=1
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


points = ax.scatter(
    M_open_pc[:, 0],
    M_open_pc[:, 1],
    c=morph_activity,
    s=60,
    alpha=0.85
)


plt.colorbar(
    points,
    ax=ax,
    label="Morphology perturbation magnitude"
)


ax.set_xlabel(
    "PC1 "
    f"({100 * pca_open.explained_variance_ratio_[0]:.1f}% variance)"
)


ax.set_ylabel(
    "PC2 "
    f"({100 * pca_open.explained_variance_ratio_[1]:.1f}% variance)"
)


ax.set_title(
    "OpenScreen HepG2 final morphology landscape"
)


top = np.argsort(
    morph_activity
)[-10:]


names_open = np.asarray(
    M_open.index
)


for i in top:

    ax.annotate(
        names_open[i],
        (
            M_open_pc[i, 0],
            M_open_pc[i, 1]
        ),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
    )


savefig(
    "openscreen_04_morphology_pca.png"
)


# ============================================================
# TAHOE FIGURE 1
# CELLS PER SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("TAHOE FIGURE 1")
print("CELLS PER SAMPLE")
print("=" * 80)

qc = pd.read_csv(
    SAMPLE_QC,
    index_col=0
)


cells = qc[
    "cells"
].to_numpy()


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    cells,
    bins=40,
    alpha=0.8
)


ax.axvline(
    100,
    linestyle="--",
    linewidth=2,
    label="QC threshold = 100 cells"
)


ax.set_xlabel(
    "Cells per sample"
)

ax.set_ylabel(
    "Number of samples"
)

ax.set_title(
    "TAHOE HepG2 sample-level cell-count QC"
)


ax.legend()


ax.text(
    0.98,
    0.95,
    "416 total samples\n"
    "409 retained\n"
    "7 removed",
    transform=ax.transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.1
    )
)


savefig(
    "01_tahoe_cells_per_sample.png"
)


# ============================================================
# TAHOE FIGURE 2
# PSEUDOBULK UMI / LIBRARY SIZE
# ============================================================

print("\n" + "=" * 80)
print("TAHOE FIGURE 2")
print("PSEUDOBULK LIBRARY SIZE")
print("=" * 80)

a = ad.read_h5ad(
    RAW_PB,
    backed="r"
)


libsize = np.asarray(
    a.X[:, :].sum(
        axis=1
    )
).ravel()


a.file.close()


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    np.log10(
        libsize
    ),
    bins=35,
    alpha=0.8
)


median_lib = np.median(
    libsize
)


ax.axvline(
    np.log10(
        median_lib
    ),
    linestyle="--",
    linewidth=2,
    label=(
        f"Median = "
        f"{median_lib:,.0f} UMIs"
    )
)


ax.set_xlabel(
    "log10 total UMI counts per sample pseudobulk"
)

ax.set_ylabel(
    "Number of samples"
)

ax.set_title(
    "TAHOE HepG2 pseudobulk library-size distribution"
)

ax.legend()


savefig(
    "02_tahoe_pseudobulk_umi.png"
)


# ============================================================
# TAHOE FIGURE 3
# SAME-DRUG VS DIFFERENT-DRUG REPRODUCIBILITY
# ============================================================

print("\n" + "=" * 80)
print("TAHOE FIGURE 3")
print("DRUG-SPECIFIC REPLICATE QC")
print("=" * 80)

X = pd.read_parquet(
    SAMPLE_EXPR
)


meta = pd.read_csv(
    SAMPLE_QC,
    index_col=0
)


meta.index = (
    meta.index.astype(str)
)

X.index = (
    X.index.astype(str)
)


meta = meta.loc[
    X.index
]


labels = (
    meta["drug"]
    .astype(str)
    .to_numpy()
)


A = X.to_numpy(
    dtype=np.float64
)


mu = A.mean(
    axis=0
)

sd = A.std(
    axis=0,
    ddof=1
)

sd[sd == 0] = 1


Z = (
    A - mu
) / sd


C = np.corrcoef(
    Z
)


same = []
different = []


n_samples = len(
    labels
)


for i in range(
    n_samples
):

    for j in range(
        i + 1,
        n_samples
    ):

        r = C[
            i,
            j
        ]

        if not np.isfinite(
            r
        ):
            continue

        if (
            labels[i]
            ==
            labels[j]
        ):
            same.append(
                r
            )

        else:
            different.append(
                r
            )


same = np.asarray(
    same
)

different = np.asarray(
    different
)


null95 = np.quantile(
    different,
    0.95
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    different,
    bins=60,
    density=True,
    alpha=0.55,
    label="Different-drug pairs"
)


ax.hist(
    same,
    bins=40,
    density=True,
    alpha=0.65,
    label="Same-drug replicates"
)


ax.axvline(
    np.median(
        same
    ),
    linestyle="--",
    linewidth=2,
    label=(
        "Same-drug median "
        f"r={np.median(same):.3f}"
    )
)


ax.axvline(
    null95,
    linestyle=":",
    linewidth=2,
    label=(
        "Different-drug null "
        f"95%={null95:.3f}"
    )
)


ax.set_xlabel(
    "Pearson correlation of drug-specific expression patterns"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "TAHOE drug-specific replicate reproducibility"
)

ax.legend(
    fontsize=10
)


savefig(
    "03_tahoe_replicate_qc.png"
)


# ============================================================
# TAHOE FIGURE 4
# DRUG PCA
# ============================================================

print("\n" + "=" * 80)
print("TAHOE FIGURE 4")
print("DRUG PCA")
print("=" * 80)

pca_g_plot = PCA(
    n_components=30,
    random_state=RANDOM_STATE
)


Gpc_plot = (
    pca_g_plot.fit_transform(
        G.to_numpy(
            dtype=np.float64
        )
    )
)


Gz_plot = zscore_cols(
    Gpc_plot
)


activity_gex = np.linalg.norm(
    Gz_plot
    -
    Gz_plot.mean(
        axis=0
    ),
    axis=1
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


points = ax.scatter(
    Gpc_plot[:, 0],
    Gpc_plot[:, 1],
    c=activity_gex,
    s=55,
    alpha=0.85
)


plt.colorbar(
    points,
    ax=ax,
    label="Transcriptomic perturbation magnitude"
)


ax.set_xlabel(
    "PC1 "
    f"({100 * pca_g_plot.explained_variance_ratio_[0]:.1f}% variance)"
)


ax.set_ylabel(
    "PC2 "
    f"({100 * pca_g_plot.explained_variance_ratio_[1]:.1f}% variance)"
)


ax.set_title(
    "TAHOE HepG2 drug-level transcriptomic landscape"
)


top = np.argsort(
    activity_gex
)[-10:]


for i in top:

    ax.annotate(
        shared[i],
        (
            Gpc_plot[i, 0],
            Gpc_plot[i, 1]
        ),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
    )


savefig(
    "04_tahoe_drug_pca.png"
)


# ============================================================
# TAHOE FIGURE 5
# CLUSTERED EXPRESSION HEATMAP
# ============================================================

print("\n" + "=" * 80)
print("TAHOE FIGURE 5")
print("CLUSTERED EXPRESSION HEATMAP")
print("=" * 80)

gene_var = G.var(
    axis=0
)


top_genes = (
    gene_var
    .sort_values(
        ascending=False
    )
    .head(50)
    .index
)


H = G[
    top_genes
].copy()


H = (
    H - H.mean(
        axis=0
    )
) / (
    H.std(
        axis=0
    )
    .replace(
        0,
        1
    )
)


drug_link = linkage(
    H.to_numpy(),
    method="average",
    metric="correlation"
)


drug_order = leaves_list(
    drug_link
)


gene_link = linkage(
    H.to_numpy().T,
    method="average",
    metric="correlation"
)


gene_order = leaves_list(
    gene_link
)


Hc = H.iloc[
    drug_order,
    gene_order
]


fig, ax = plt.subplots(
    figsize=(14, 10)
)


im = ax.imshow(
    Hc.to_numpy(),
    aspect="auto",
    interpolation="nearest"
)


plt.colorbar(
    im,
    ax=ax,
    label="Gene expression z-score"
)


ax.set_xlabel(
    "50 most variable HVGs"
)

ax.set_ylabel(
    "119 drugs, hierarchically clustered"
)

ax.set_title(
    "TAHOE HepG2 drug-expression clusters"
)


ax.set_xticks(
    np.arange(
        len(
            top_genes
        )
    )
)


ax.set_xticklabels(
    Hc.columns,
    rotation=90,
    fontsize=6
)


yticks = np.arange(
    0,
    len(
        Hc
    ),
    5
)


ax.set_yticks(
    yticks
)


ax.set_yticklabels(
    Hc.index[
        yticks
    ],
    fontsize=6
)


savefig(
    "05_tahoe_clustered_expression_heatmap.png"
)


# ============================================================
# TASK 2 REPRESENTATION
# EXACT NOTEBOOK-STYLE:
# 30 PCs → z-score PCs → Euclidean distances
# ============================================================

print("\n" + "=" * 80)
print("TASK 2")
print("RECONSTRUCTING NOTEBOOK-STYLE DISTANCES")
print("=" * 80)

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
        dtype=np.float64
    )
)


Gpc = pca_g.fit_transform(
    G.to_numpy(
        dtype=np.float64
    )
)


Mz = zscore_cols(
    Mpc
)

Gz = zscore_cols(
    Gpc
)


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
    len(
        shared
    ),
    k=1
)


dm = Dm[
    iu
]

dg = Dg[
    iu
]


r_cross = spearmanr(
    dm,
    dg
).statistic


print(
    "Cross-modal r:",
    r_cross
)


# ============================================================
# TASK 2 FIGURE 6
# CROSS-MODAL DISTANCE RELATIONSHIP
# ============================================================

print("\n" + "=" * 80)
print("TASK 2 FIGURE 6")
print("=" * 80)

fig, ax = plt.subplots(
    figsize=(8, 7)
)


hb = ax.hexbin(
    dm,
    dg,
    gridsize=45,
    mincnt=1
)


plt.colorbar(
    hb,
    ax=ax,
    label="Number of drug pairs"
)


ax.set_xlabel(
    "Morphology drug-drug Euclidean distance"
)

ax.set_ylabel(
    "Transcriptomic drug-drug Euclidean distance"
)

ax.set_title(
    "Cross-modal drug-pair relationship"
)


ax.text(
    0.05,
    0.95,
    "119 drugs\n"
    "7,021 pairs\n"
    f"Spearman r = {r_cross:.3f}\n"
    "Permutation p = 0.001",
    transform=ax.transAxes,
    va="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.1
    )
)


savefig(
    "06_crossmodal_distance_relationship.png"
)


# ============================================================
# TASK 2 FIGURE 7
# PERTURBATION MAGNITUDE
# ============================================================

print("\n" + "=" * 80)
print("TASK 2 FIGURE 7")
print("=" * 80)

m_mag = np.linalg.norm(
    Mz
    -
    Mz.mean(
        axis=0
    ),
    axis=1
)


g_mag = np.linalg.norm(
    Gz
    -
    Gz.mean(
        axis=0
    ),
    axis=1
)


rho = spearmanr(
    m_mag,
    g_mag
).statistic


fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.scatter(
    m_mag,
    g_mag,
    s=60,
    alpha=0.8
)


ax.set_xlabel(
    "Morphology perturbation magnitude"
)

ax.set_ylabel(
    "Transcriptomic perturbation magnitude"
)

ax.set_title(
    "Perturbation strength agrees across modalities"
)


ax.text(
    0.05,
    0.95,
    f"Spearman rho = {rho:.3f}",
    transform=ax.transAxes,
    va="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.1
    )
)


mr = pd.Series(
    m_mag
).rank(
    pct=True
).to_numpy()


gr = pd.Series(
    g_mag
).rank(
    pct=True
).to_numpy()


joint = np.minimum(
    mr,
    gr
)


top_joint = np.argsort(
    joint
)[-10:]


for i in top_joint:

    ax.annotate(
        shared[i],
        (
            m_mag[i],
            g_mag[i]
        ),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
    )


savefig(
    "07_crossmodal_perturbation_magnitude.png"
)


# ============================================================
# TASK 2 FIGURE 8
# DIRECTION BY ACTIVITY
# ============================================================

print("\n" + "=" * 80)
print("TASK 2 FIGURE 8")
print("=" * 80)

if DIRECTION_ACTIVITY.exists():

    d = pd.read_csv(
        DIRECTION_ACTIVITY
    )


    # k=119 has no meaningful random-subset comparison
    d = d[
        d[
            "top_k_active"
        ] < 119
    ].copy()


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.plot(
        d[
            "top_k_active"
        ],
        d[
            "direction_r"
        ],
        marker="o",
        linewidth=2,
        label="Most-active compounds"
    )


    ax.plot(
        d[
            "top_k_active"
        ],
        d[
            "random_subset_mean_r"
        ],
        marker="o",
        linestyle="--",
        label="Random subsets: mean"
    )


    ax.plot(
        d[
            "top_k_active"
        ],
        d[
            "random_subset_p95"
        ],
        linestyle=":",
        linewidth=2,
        label="Random subsets: 95th percentile"
    )


    ax.axhline(
        0.0404,
        linestyle="--",
        linewidth=1,
        label="All 119 drugs: r=0.040"
    )


    ax.set_xlabel(
        "Number of most-active drugs retained"
    )


    ax.set_ylabel(
        "Direction-only cross-modal Mantel r"
    )


    ax.set_title(
        "Directional correspondence does not increase with activity"
    )


    ax.legend(
        fontsize=9
    )


    savefig(
        "08_direction_by_activity.png"
    )


else:

    print(
        "Skipping Figure 8: "
        "direction_by_activity.csv not found."
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("ALL PRESENTATION FIGURES CREATED")
print("=" * 80)

for f in sorted(
    OUT.glob(
        "*.png"
    )
):

    print(f)

