#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

MORPH = Path(
    "OpenScreen/data/hepg2_morphology_final.parquet"
)

GEX = Path(
    "OpenScreen/data/hepg2_pseudobulk_2000hvg_shared119.parquet"
)

OUT = Path("OpenScreen/task2")
OUT.mkdir(parents=True, exist_ok=True)

N_PCS = 30
N_PERM = 999
RANDOM_STATE = 0


def zscore(df):
    std = df.std(ddof=0).replace(0, 1)
    return (df - df.mean()) / std


print("=" * 78)
print("NOTEBOOK METHOD ON FINAL QC-CONTROLLED DATA")
print("=" * 78)

M = pd.read_parquet(MORPH)
G = pd.read_parquet(GEX)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

shared = sorted(set(M.index) & set(G.index))

M = M.loc[shared]
G = G.loc[shared]

print("\nMorphology:", M.shape)
print("Expression:", G.shape)
print("Shared drugs:", len(shared))


# ------------------------------------------------------------
# PCA exactly at the current drug-profile level
# NO pre-PCA StandardScaler, to match notebook behavior better
# ------------------------------------------------------------

pca_m = PCA(
    n_components=N_PCS,
    random_state=RANDOM_STATE
)

pca_g = PCA(
    n_components=N_PCS,
    random_state=RANDOM_STATE
)

M_pc = pca_m.fit_transform(
    M.to_numpy(dtype=np.float64)
)

G_pc = pca_g.fit_transform(
    G.to_numpy(dtype=np.float64)
)

morph_compound_pca = pd.DataFrame(
    M_pc,
    index=shared
)

rna_compound_pca = pd.DataFrame(
    G_pc,
    index=shared
)

print("\nVariance explained:")
print(
    "Morphology 30 PCs:",
    pca_m.explained_variance_ratio_.sum()
)
print(
    "Expression 30 PCs:",
    pca_g.explained_variance_ratio_.sum()
)


# ------------------------------------------------------------
# Exact notebook-style PC standardisation
# ------------------------------------------------------------

morph_z = zscore(
    morph_compound_pca
)

rna_z = zscore(
    rna_compound_pca
)


# ------------------------------------------------------------
# EXACT NOTEBOOK DISTANCE:
# scipy pdist default = Euclidean
# ------------------------------------------------------------

dist_morph = squareform(
    pdist(morph_z.to_numpy())
)

dist_rna = squareform(
    pdist(rna_z.to_numpy())
)

iu = np.triu_indices(
    len(shared),
    k=1
)

mantel_r, _ = spearmanr(
    dist_morph[iu],
    dist_rna[iu]
)


# ------------------------------------------------------------
# Exact notebook-style permutation
# ------------------------------------------------------------

rng = np.random.RandomState(
    RANDOM_STATE
)

idx = np.arange(
    len(shared)
)

null_rs = np.empty(
    N_PERM
)

for i in range(N_PERM):

    perm = rng.permutation(idx)

    null_rs[i], _ = spearmanr(
        dist_morph[iu],
        dist_rna[
            np.ix_(perm, perm)
        ][iu]
    )

mantel_p = (
    np.sum(
        np.abs(null_rs)
        >= abs(mantel_r)
    ) + 1
) / (
    N_PERM + 1
)


print("\n" + "=" * 78)
print("NOTEBOOK-STYLE RESULT")
print("=" * 78)

print("Drug pairs:", len(dist_morph[iu]))

print(
    "Mantel-style Spearman r:",
    mantel_r
)

print(
    "Permutation p:",
    mantel_p
)

print(
    "Null mean:",
    np.mean(null_rs)
)

print(
    "Null SD:",
    np.std(null_rs)
)

print(
    "Null p95:",
    np.quantile(null_rs, 0.95)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

pd.DataFrame(
    [{
        "analysis":
            "notebook_metric_on_final_QC_data",
        "n_drugs":
            len(shared),
        "n_pcs":
            N_PCS,
        "mantel_spearman_r":
            mantel_r,
        "permutation_p":
            mantel_p,
        "morph_variance_30pc":
            pca_m.explained_variance_ratio_.sum(),
        "gex_variance_30pc":
            pca_g.explained_variance_ratio_.sum(),
    }]
).to_csv(
    OUT /
    "hepg2_notebook_metric_finaldata.csv",
    index=False
)

print("\nSaved:")
print(
    OUT /
    "hepg2_notebook_metric_finaldata.csv"
)
