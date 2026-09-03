#!/usr/bin/env python3

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


EXPR = "LINCS/data/lincs_expression_a549.h5ad"
MORPH = "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"

N_HVG = 2000
N_PCS = 30
N_PERM = 999
SEED = 0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch-key",
    choices=["none", "plate", "sample"],
    default="plate",
)
args = parser.parse_args()

BATCH_KEY = args.batch_key

OUT = Path(
    f"LINCS/scanpy_sensitivity/batch_{BATCH_KEY}"
)
OUT.mkdir(parents=True, exist_ok=True)


def zscore_cols(X):
    X = np.asarray(X, dtype=np.float64)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)

    sd[sd == 0] = 1.0

    return (X - mu) / sd


def unit_rows(X):
    X = np.asarray(X, dtype=np.float64)

    norm = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    norm[norm == 0] = 1.0

    return X / norm


def distmat(X, metric="euclidean"):
    return squareform(
        pdist(
            np.asarray(X),
            metric=metric,
        )
    )


def permutation_geometry(D1, D2):
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)

    obs = float(
        spearmanr(
            D1[iu],
            D2[iu],
        ).statistic
    )

    rng = np.random.default_rng(SEED)

    null = np.zeros(N_PERM)

    idx = np.arange(n)

    for i in range(N_PERM):

        p = rng.permutation(idx)

        Dp = D2[
            np.ix_(p, p)
        ]

        null[i] = spearmanr(
            D1[iu],
            Dp[iu],
        ).statistic

    # positive-direction permutation test,
    # same convention as final A549 analysis
    pval = (
        1 + np.sum(null >= obs)
    ) / (
        N_PERM + 1
    )

    return obs, float(pval)


def strength_test(x, y):

    obs = float(
        spearmanr(x, y).statistic
    )

    rng = np.random.default_rng(SEED)

    null = np.zeros(N_PERM)

    for i in range(N_PERM):

        yp = y[
            rng.permutation(len(y))
        ]

        null[i] = spearmanr(
            x,
            yp,
        ).statistic

    pval = (
        1 + np.sum(null >= obs)
    ) / (
        N_PERM + 1
    )

    return obs, float(pval)


print("=" * 100)
print("LINCS A549 — SCANPY BATCH-AWARE HVG SENSITIVITY")
print("=" * 100)
print("batch key:", BATCH_KEY)


# =====================================================================
# EXPRESSION
# =====================================================================

print("\nLoading processed/log1p expression...")

a = ad.read_h5ad(EXPR)

print("cells:", a.n_obs)
print("genes:", a.n_vars)
print("samples:", a.obs["sample"].nunique())
print("plates:", a.obs["plate"].nunique())
print("drugs:", a.obs["Metadata_Drug"].nunique())
print("X type:", type(a.X))


# ---------------------------------------------------------------------
# Existing HVGs
# ---------------------------------------------------------------------

if "highly_variable" in a.var.columns:

    old_hvg = (
        a.var["highly_variable"]
        .fillna(False)
        .to_numpy(dtype=bool)
        .copy()
    )

else:

    old_hvg = np.zeros(
        a.n_vars,
        dtype=bool,
    )


print(
    "existing HVGs:",
    int(old_hvg.sum()),
)


# ---------------------------------------------------------------------
# IMPORTANT:
# X is already normalized + log1p.
#
# DO NOT normalize_total()
# DO NOT log1p()
#
# Re-run only Scanpy feature selection using technical batch information.
# Default Seurat flavor expects log-transformed expression.
# ---------------------------------------------------------------------

print(
    f"\nSelecting {N_HVG} HVGs "
    f"with batch_key='{BATCH_KEY}'..."
)

if BATCH_KEY == "none":
    sc.pp.highly_variable_genes(
        a,
        n_top_genes=N_HVG,
        flavor="seurat",
        inplace=True,
        subset=False,
    )
else:
    sc.pp.highly_variable_genes(
        a,
        n_top_genes=N_HVG,
        batch_key=BATCH_KEY,
        flavor="seurat",
        inplace=True,
        subset=False,
    )


new_hvg = (
    a.var["highly_variable"]
    .fillna(False)
    .to_numpy(dtype=bool)
)


print(
    "new HVGs:",
    int(new_hvg.sum()),
)

overlap = int(
    np.sum(
        old_hvg & new_hvg
    )
)

print(
    "overlap with previous 2000:",
    overlap,
)

print(
    "overlap fraction:",
    overlap / max(
        1,
        old_hvg.sum(),
    ),
)


# Save HVG table
hvg_table = a.var.loc[
    new_hvg
].copy()

if "gene_symbol" in hvg_table.columns:
    genes = (
        hvg_table["gene_symbol"]
        .astype(str)
        .tolist()
    )
else:
    genes = (
        hvg_table.index
        .astype(str)
        .tolist()
    )

hvg_table.to_csv(
    OUT / "scanpy_batchaware_2000HVG.csv"
)


# ---------------------------------------------------------------------
# Cells -> sample mean
# ---------------------------------------------------------------------

print("\nAggregating cells -> samples...")

X = a[:, new_hvg].X

if not sparse.issparse(X):
    X = sparse.csr_matrix(X)
else:
    X = X.tocsr()


sample_labels = (
    a.obs["sample"]
    .astype(str)
    .to_numpy()
)

samples, sample_code = np.unique(
    sample_labels,
    return_inverse=True,
)

n_samples = len(samples)
n_cells = a.n_obs


# Sparse sample × cell membership matrix
membership = sparse.csr_matrix(
    (
        np.ones(
            n_cells,
            dtype=np.float32,
        ),
        (
            sample_code,
            np.arange(n_cells),
        ),
    ),
    shape=(
        n_samples,
        n_cells,
    ),
)


sample_counts = np.asarray(
    membership.sum(axis=1)
).ravel()


sample_sum = membership @ X

sample_X = (
    sample_sum.toarray()
    /
    sample_counts[:, None]
)


# sample metadata
obs_tmp = a.obs.copy()
obs_tmp["_sample_string"] = (
    obs_tmp["sample"]
    .astype(str)
)

sample_meta = (
    obs_tmp
    .groupby(
        "_sample_string",
        observed=True,
    )
    .agg(
        drug=("Metadata_Drug", "first"),
        plate=("plate", "first"),
    )
    .reindex(samples)
)

sample_meta["n_cells"] = sample_counts


print(
    "sample matrix before QC:",
    sample_X.shape,
)

print(
    "minimum cells/sample:",
    int(sample_counts.min()),
)


# ---------------------------------------------------------------------
# Same QC100 rule as primary analysis
# ---------------------------------------------------------------------

keep_sample = (
    sample_counts >= 100
)

sample_X = sample_X[
    keep_sample
]

sample_meta = sample_meta.iloc[
    np.where(keep_sample)[0]
].copy()


print(
    "samples retained >=100 cells:",
    sample_X.shape[0],
)

print(
    "samples removed:",
    int(
        (~keep_sample).sum()
    ),
)

print(
    "minimum retained cells:",
    int(
        sample_meta["n_cells"].min()
    ),
)


# ---------------------------------------------------------------------
# Gene standardization across samples
# ---------------------------------------------------------------------

mu = sample_X.mean(
    axis=0,
    keepdims=True,
)

sd = sample_X.std(
    axis=0,
    ddof=0,
    keepdims=True,
)

sd[sd == 0] = 1.0

sample_Z = (
    sample_X - mu
) / sd


# ---------------------------------------------------------------------
# Within-plate centering
# ---------------------------------------------------------------------

plates = (
    sample_meta["plate"]
    .astype(str)
    .to_numpy()
)

sample_Z_corr = (
    sample_Z.copy()
)

for plate in np.unique(plates):

    mask = (
        plates == plate
    )

    sample_Z_corr[
        mask
    ] -= sample_Z_corr[
        mask
    ].mean(
        axis=0,
        keepdims=True,
    )


# ---------------------------------------------------------------------
# Samples -> drug consensus
# ---------------------------------------------------------------------

drug_labels = (
    sample_meta["drug"]
    .astype(str)
    .to_numpy()
)

expr_drugs = sorted(
    np.unique(
        drug_labels
    )
)

expr_drug_X = np.vstack([
    sample_Z_corr[
        drug_labels == d
    ].mean(axis=0)
    for d in expr_drugs
])


print(
    "\nExpression drug matrix:",
    expr_drug_X.shape,
)


expr_df = pd.DataFrame(
    expr_drug_X,
    index=expr_drugs,
    columns=genes,
)


# =====================================================================
# MORPHOLOGY
# =====================================================================

print("\nLoading morphology...")

m = ad.read_h5ad(MORPH)

MX = np.asarray(
    m.X,
    dtype=np.float64,
)

max_abs = np.nanmax(
    np.abs(MX),
    axis=0,
)

keep_morph = (
    np.all(
        np.isfinite(MX),
        axis=0,
    )
    &
    (max_abs <= 1e6)
)

print(
    "morphology dose profiles:",
    m.n_obs,
)

print(
    "original morphology features:",
    m.n_vars,
)

print(
    "excluded pathological:",
    int(
        (~keep_morph).sum()
    ),
)

print(
    "retained morphology features:",
    int(
        keep_morph.sum()
    ),
)


MX = MX[
    :,
    keep_morph
]


morph_drug_labels = (
    m.obs["Metadata_Drug"]
    .astype(str)
    .to_numpy()
)

morph_drugs = sorted(
    np.unique(
        morph_drug_labels
    )
)


# average six dose profiles -> drug consensus
morph_drug_X = np.vstack([
    MX[
        morph_drug_labels == d
    ].mean(axis=0)
    for d in morph_drugs
])


morph_df = pd.DataFrame(
    morph_drug_X,
    index=morph_drugs,
)


# =====================================================================
# MATCH
# =====================================================================

shared = sorted(
    set(expr_df.index)
    &
    set(morph_df.index)
)

print("\nShared drugs:", len(shared))
print(
    "drug pairs:",
    len(shared) * (
        len(shared) - 1
    ) // 2,
)


E = expr_df.loc[
    shared
].to_numpy()

M = morph_df.loc[
    shared
].to_numpy()


# =====================================================================
# PCA30
# =====================================================================

Epca = PCA(
    n_components=N_PCS,
    svd_solver="full",
    random_state=SEED,
).fit_transform(E)

Mpca = PCA(
    n_components=N_PCS,
    svd_solver="full",
    random_state=SEED,
).fit_transform(M)


Ez = zscore_cols(Epca)
Mz = zscore_cols(Mpca)


# =====================================================================
# 1. Overall geometry
# =====================================================================

DE = distmat(Ez)
DM = distmat(Mz)

overall_r, overall_p = (
    permutation_geometry(
        DM,
        DE,
    )
)


# =====================================================================
# 2. Perturbation strength
# =====================================================================

expr_strength = np.linalg.norm(
    Ez,
    axis=1,
)

morph_strength = np.linalg.norm(
    Mz,
    axis=1,
)

strength_rho, strength_p = (
    strength_test(
        morph_strength,
        expr_strength,
    )
)


# =====================================================================
# 3. Direction-only — unit Euclidean
# =====================================================================

Edir = unit_rows(Ez)
Mdir = unit_rows(Mz)

direction_r, direction_p = (
    permutation_geometry(
        distmat(Mdir),
        distmat(Edir),
    )
)


# =====================================================================
# 4. Explicit cosine
# =====================================================================

cosine_r, cosine_p = (
    permutation_geometry(
        distmat(
            Mz,
            metric="cosine",
        ),
        distmat(
            Ez,
            metric="cosine",
        ),
    )
)


# =====================================================================
# UMAP coordinates — visualization only
# =====================================================================

moa_map = (
    a.obs
    .groupby(
        "Metadata_Drug",
        observed=True,
    )["Metadata_moa_fine"]
    .first()
    .astype(str)
    .to_dict()
)


def make_umap(PC, label):

    d = ad.AnnData(
        X=np.zeros(
            (
                len(shared),
                1,
            )
        )
    )

    d.obs_names = shared

    d.obs["drug"] = shared

    d.obs["moa"] = [
        moa_map.get(
            x,
            "unknown",
        )
        for x in shared
    ]

    d.obsm["X_pca"] = PC

    sc.pp.neighbors(
        d,
        n_neighbors=10,
        use_rep="X_pca",
    )

    sc.tl.umap(
        d,
        random_state=SEED,
    )

    df = pd.DataFrame(
        {
            "drug": shared,
            "moa": d.obs["moa"].to_numpy(),
            "UMAP1": d.obsm["X_umap"][:, 0],
            "UMAP2": d.obsm["X_umap"][:, 1],
        }
    )

    df.to_csv(
        OUT / f"{label}_drug_UMAP.csv",
        index=False,
    )


make_umap(
    Mz,
    "morphology",
)

make_umap(
    Ez,
    "expression",
)


# =====================================================================
# SAVE SUMMARY
# =====================================================================

summary = pd.DataFrame([
    {
        "batch_key": BATCH_KEY,
        "n_drugs": len(shared),
        "n_pairs": len(shared) * (len(shared)-1) // 2,
        "old_hvg_count": int(old_hvg.sum()),
        "new_hvg_count": int(new_hvg.sum()),
        "hvg_overlap": overlap,
        "overall_r": overall_r,
        "overall_perm_p": overall_p,
        "strength_rho": strength_rho,
        "strength_perm_p": strength_p,
        "direction_r": direction_r,
        "direction_perm_p": direction_p,
        "cosine_r": cosine_r,
        "cosine_perm_p": cosine_p,
    }
])

summary.to_csv(
    OUT / "SCANPY_HVG_sensitivity_summary.csv",
    index=False,
)


print("\n" + "=" * 100)
print("SCANPY SENSITIVITY RESULT")
print("=" * 100)

print(
    f"Batch-aware HVGs : {BATCH_KEY}"
)

print(
    f"HVG overlap      : "
    f"{overlap}/{int(old_hvg.sum())}"
)

print(
    f"Overall geometry : "
    f"r={overall_r:.6f}, "
    f"p={overall_p:.4f}"
)

print(
    f"Strength         : "
    f"rho={strength_rho:.6f}, "
    f"p={strength_p:.4f}"
)

print(
    f"Direction only   : "
    f"r={direction_r:.6f}, "
    f"p={direction_p:.4f}"
)

print(
    f"Cosine           : "
    f"r={cosine_r:.6f}, "
    f"p={cosine_p:.4f}"
)

print("\nPRIMARY QC100 REFERENCE")
print("overall   r = 0.352234")
print("strength rho = 0.344026")
print("direction r = -0.001023")

print("\nSaved to:", OUT)

print("\nDONE")
