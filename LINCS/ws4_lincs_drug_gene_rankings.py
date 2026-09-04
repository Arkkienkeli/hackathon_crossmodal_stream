#!/usr/bin/env python3

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata, t
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

EXPR = "LINCS/data/lincs_expression_a549.h5ad"
MORPH = "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"

HVG_FILE = (
    "LINCS/scanpy_sensitivity/batch_plate/"
    "scanpy_batchaware_2000HVG.csv"
)

OUT = Path(
    "LINCS/scanpy_sensitivity/batch_plate/rankings"
)
OUT.mkdir(parents=True, exist_ok=True)

N_PCS = 30


# ============================================================
# HELPERS
# ============================================================

def zscore_cols(X):
    X = np.asarray(X, dtype=float)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)

    sd[sd == 0] = 1

    return (X - mu) / sd


def bh_fdr(p):
    """Benjamini-Hochberg FDR."""
    p = np.asarray(p, dtype=float)

    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / np.arange(1, n + 1)

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(q, 0, 1)

    out = np.empty(n)
    out[order] = q

    return out


print("=" * 100)
print("A549 PLATE-AWARE SCANPY — DRUG + GENE RANKINGS")
print("=" * 100)


# ============================================================
# LOAD EXPRESSION
# ============================================================

print("\nLoading expression...")

a = ad.read_h5ad(EXPR)

print("cells:", a.n_obs)
print("genes:", a.n_vars)


# ============================================================
# LOAD EXACT PLATE-AWARE HVGs
# ============================================================

hvg_table = pd.read_csv(
    HVG_FILE,
    index_col=0,
)

hvg_ids = (
    hvg_table.index
    .astype(str)
    .tolist()
)

hvg_mask = (
    a.var_names.astype(str)
    .isin(hvg_ids)
)

print(
    "plate-aware HVGs recovered:",
    int(hvg_mask.sum()),
)

if hvg_mask.sum() != 2000:
    raise RuntimeError(
        f"Expected 2000 HVGs, got {hvg_mask.sum()}"
    )


# gene labels
expr_var = a.var.loc[hvg_mask].copy()

gene_ids = (
    expr_var.index
    .astype(str)
    .to_numpy()
)

if "gene_symbol" in expr_var.columns:
    gene_symbols = (
        expr_var["gene_symbol"]
        .astype(str)
        .to_numpy()
    )
else:
    gene_symbols = gene_ids.copy()


# ============================================================
# CELLS -> SAMPLES
# ============================================================

print("\nAggregating cells -> samples...")

X = a[:, hvg_mask].X

if sparse.issparse(X):
    X = X.tocsr()
else:
    X = sparse.csr_matrix(X)


sample_labels = (
    a.obs["sample"]
    .astype(str)
    .to_numpy()
)

samples, sample_code = np.unique(
    sample_labels,
    return_inverse=True,
)

n_cells = a.n_obs
n_samples = len(samples)


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
    shape=(n_samples, n_cells),
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


obs_tmp = a.obs.copy()

obs_tmp["_sample"] = (
    obs_tmp["sample"]
    .astype(str)
)

sample_meta = (
    obs_tmp
    .groupby(
        "_sample",
        observed=True,
    )
    .agg(
        drug=("Metadata_Drug", "first"),
        plate=("plate", "first"),
    )
    .reindex(samples)
)

sample_meta["n_cells"] = sample_counts


# ============================================================
# QC >= 100 CELLS
# ============================================================

keep = sample_counts >= 100

sample_X = sample_X[keep]

sample_meta = sample_meta.iloc[
    np.where(keep)[0]
].copy()

print(
    "samples retained:",
    sample_X.shape[0],
)


# ============================================================
# STANDARDIZE GENES ACROSS SAMPLES
# ============================================================

mu = sample_X.mean(
    axis=0,
    keepdims=True,
)

sd = sample_X.std(
    axis=0,
    ddof=0,
    keepdims=True,
)

sd[sd == 0] = 1

sample_Z = (
    sample_X - mu
) / sd


# ============================================================
# WITHIN-PLATE CENTERING
# ============================================================

plates = (
    sample_meta["plate"]
    .astype(str)
    .to_numpy()
)

sample_Z_corr = sample_Z.copy()

for plate in np.unique(plates):

    mask = plates == plate

    sample_Z_corr[mask] -= (
        sample_Z_corr[mask]
        .mean(
            axis=0,
            keepdims=True,
        )
    )


# ============================================================
# SAMPLES -> DRUG EXPRESSION
# ============================================================

drug_labels = (
    sample_meta["drug"]
    .astype(str)
    .to_numpy()
)

expr_drugs = sorted(
    np.unique(drug_labels)
)

expr_drug_X = np.vstack([
    sample_Z_corr[
        drug_labels == d
    ].mean(axis=0)
    for d in expr_drugs
])


expr_df = pd.DataFrame(
    expr_drug_X,
    index=expr_drugs,
    columns=gene_ids,
)


# ============================================================
# MORPHOLOGY
# ============================================================

print("\nLoading morphology...")

m = ad.read_h5ad(MORPH)

MX = np.asarray(
    m.X,
    dtype=float,
)

max_abs = np.nanmax(
    np.abs(MX),
    axis=0,
)

keep_m = (
    np.all(
        np.isfinite(MX),
        axis=0,
    )
    &
    (max_abs <= 1e6)
)

morph_features = (
    m.var_names[
        keep_m
    ]
    .astype(str)
    .to_numpy()
)

MX = MX[:, keep_m]

print(
    "morphology features:",
    MX.shape[1],
)


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

morph_drug_X = np.vstack([
    MX[
        morph_drug_labels == d
    ].mean(axis=0)
    for d in morph_drugs
])


morph_df = pd.DataFrame(
    morph_drug_X,
    index=morph_drugs,
    columns=morph_features,
)


# ============================================================
# MATCH DRUGS
# ============================================================

shared = sorted(
    set(expr_df.index)
    &
    set(morph_df.index)
)

print(
    "\nShared drugs:",
    len(shared),
)


Egenes = (
    expr_df
    .loc[shared]
    .to_numpy()
)

Mfeatures = (
    morph_df
    .loc[shared]
    .to_numpy()
)


# ============================================================
# PCA30 FOR CROSS-MODAL GEOMETRY
# ============================================================

Epca = PCA(
    n_components=N_PCS,
    random_state=0,
).fit_transform(Egenes)

Mpca = PCA(
    n_components=N_PCS,
    random_state=0,
).fit_transform(Mfeatures)


Ez = zscore_cols(Epca)
Mz = zscore_cols(Mpca)


DE = squareform(
    pdist(Ez)
)

DM = squareform(
    pdist(Mz)
)


# ============================================================
# 1. PER-DRUG CROSS-MODAL NEIGHBORHOOD CONCORDANCE
# ============================================================

print(
    "\nCalculating per-drug cross-modal concordance..."
)

drug_rows = []

for i, drug in enumerate(shared):

    mask = np.arange(
        len(shared)
    ) != i

    r = spearmanr(
        DM[i, mask],
        DE[i, mask],
    ).statistic

    morph_strength = (
        np.linalg.norm(
            Mz[i]
        )
    )

    expr_strength = (
        np.linalg.norm(
            Ez[i]
        )
    )

    drug_rows.append(
        {
            "drug": drug,
            "crossmodal_neighborhood_r": r,
            "morphology_strength": morph_strength,
            "expression_strength": expr_strength,
        }
    )


drug_rank = pd.DataFrame(
    drug_rows
)

drug_rank[
    "abs_strength_difference"
] = np.abs(
    drug_rank[
        "morphology_strength"
    ]
    -
    drug_rank[
        "expression_strength"
    ]
)

drug_rank = drug_rank.sort_values(
    "crossmodal_neighborhood_r",
    ascending=False,
)

drug_rank.to_csv(
    OUT /
    "A549_drug_crossmodal_concordance.csv",
    index=False,
)


print(
    "\nTOP 15 CROSS-MODALLY CONCORDANT DRUGS"
)

print(
    drug_rank[
        [
            "drug",
            "crossmodal_neighborhood_r",
            "morphology_strength",
            "expression_strength",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 2. GENES ASSOCIATED WITH MORPHOLOGY STRENGTH
# ============================================================

print(
    "\nCalculating gene ↔ morphology-strength correlations..."
)

morph_strength = np.linalg.norm(
    Mz,
    axis=1,
)


gene_rows = []

for j in range(
    Egenes.shape[1]
):

    rho, p = spearmanr(
        Egenes[:, j],
        morph_strength,
    )

    gene_rows.append(
        (
            gene_ids[j],
            gene_symbols[j],
            rho,
            p,
        )
    )


gene_strength = pd.DataFrame(
    gene_rows,
    columns=[
        "gene_id",
        "gene_symbol",
        "rho_vs_morphology_strength",
        "p_value",
    ],
)

gene_strength[
    "q_value"
] = bh_fdr(
    gene_strength[
        "p_value"
    ].to_numpy()
)

gene_strength[
    "abs_rho"
] = np.abs(
    gene_strength[
        "rho_vs_morphology_strength"
    ]
)

gene_strength = (
    gene_strength
    .sort_values(
        "abs_rho",
        ascending=False,
    )
)


gene_strength.to_csv(
    OUT /
    "A549_genes_vs_morphology_strength.csv",
    index=False,
)


print(
    "\nTOP 20 GENES ASSOCIATED WITH MORPHOLOGY STRENGTH"
)

print(
    gene_strength[
        [
            "gene_symbol",
            "rho_vs_morphology_strength",
            "p_value",
            "q_value",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 3. ALL MORPHOLOGY FEATURE ↔ GENE CORRELATIONS
# ============================================================

print(
    "\nCalculating 574 × 2000 feature-gene associations..."
)


# Rank transform each variable across drugs.
MR = np.apply_along_axis(
    rankdata,
    0,
    Mfeatures,
)

ER = np.apply_along_axis(
    rankdata,
    0,
    Egenes,
)


# Standardize ranks
MR -= MR.mean(
    axis=0,
    keepdims=True,
)

ER -= ER.mean(
    axis=0,
    keepdims=True,
)

MR_sd = np.sqrt(
    np.sum(
        MR ** 2,
        axis=0,
    )
)

ER_sd = np.sqrt(
    np.sum(
        ER ** 2,
        axis=0,
    )
)

MR_sd[
    MR_sd == 0
] = np.nan

ER_sd[
    ER_sd == 0
] = np.nan


# Spearman correlation matrix
R = (
    MR.T @ ER
) / (
    MR_sd[:, None]
    *
    ER_sd[None, :]
)


# approximate Spearman p-values
n = len(shared)
df = n - 2

R_clip = np.clip(
    R,
    -0.999999999,
    0.999999999,
)

T = (
    R_clip
    *
    np.sqrt(
        df
        /
        (
            1 - R_clip ** 2
        )
    )
)

P = 2 * t.sf(
    np.abs(T),
    df=df,
)


flat_r = R.ravel()
flat_p = P.ravel()

valid = np.isfinite(
    flat_r
) & np.isfinite(
    flat_p
)

q = np.full(
    flat_p.shape,
    np.nan,
)

q[valid] = bh_fdr(
    flat_p[valid]
)


feature_idx, gene_idx = np.unravel_index(
    np.arange(
        R.size
    ),
    R.shape,
)


pairs = pd.DataFrame(
    {
        "morphology_feature":
            morph_features[
                feature_idx
            ],
        "gene_id":
            gene_ids[
                gene_idx
            ],
        "gene_symbol":
            gene_symbols[
                gene_idx
            ],
        "rho":
            flat_r,
        "p_value":
            flat_p,
        "q_value":
            q,
    }
)

pairs["abs_rho"] = np.abs(
    pairs["rho"]
)

pairs = (
    pairs
    .sort_values(
        "abs_rho",
        ascending=False,
    )
)


pairs.head(
    1000
).to_csv(
    OUT /
    "A549_top1000_feature_gene_pairs.csv",
    index=False,
)


print(
    "\nTOP 20 FEATURE ↔ GENE PAIRS"
)

print(
    pairs[
        [
            "morphology_feature",
            "gene_symbol",
            "rho",
            "p_value",
            "q_value",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nFDR q<0.05 feature-gene pairs:",
    int(
        (
            pairs["q_value"] < 0.05
        ).sum()
    ),
)


print(
    "\nMaximum |rho|:",
    pairs["abs_rho"].max(),
)


print("\nSaved to:", OUT)
print("DONE")
