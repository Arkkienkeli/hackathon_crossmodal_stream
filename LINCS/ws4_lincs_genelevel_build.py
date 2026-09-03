#!/usr/bin/env python3

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from pathlib import Path


FILE = "LINCS/data/lincs_expression_a549.h5ad"

OUT = Path("LINCS/genelevel")
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 5000


print("=" * 100)
print("LINCS A549 — BUILD 2000-HVG SAMPLE/DRUG MATRIX")
print("=" * 100)

a = ad.read_h5ad(FILE, backed="r")

obs = a.obs.copy()

hvg_mask = a.var["highly_variable"].astype(bool).to_numpy()
hvg_idx = np.where(hvg_mask)[0]

genes = (
    a.var.loc[hvg_mask, "gene_symbol"]
    .astype(str)
    .to_numpy()
)

print("cells:", a.n_obs)
print("genes:", a.n_vars)
print("HVGs:", len(hvg_idx))
print("samples:", obs["sample"].nunique())
print("drugs:", obs["Metadata_Drug"].nunique())
print("plates:", obs["plate"].nunique())


# ============================================================
# SAMPLE INDEX
# ============================================================

samples = sorted(
    obs["sample"].astype(str).unique()
)

sample_to_i = {
    s: i
    for i, s in enumerate(samples)
}

n_samples = len(samples)
n_genes = len(hvg_idx)

sums = np.zeros(
    (n_samples, n_genes),
    dtype=np.float64
)

counts = np.zeros(
    n_samples,
    dtype=np.int64
)


sample_labels = (
    obs["sample"]
    .astype(str)
    .to_numpy()
)


# ============================================================
# CHUNKED CELL -> SAMPLE AGGREGATION
# ============================================================

print("\nAggregating processed cell expression to sample means...")

for start in range(0, a.n_obs, CHUNK):

    end = min(
        start + CHUNK,
        a.n_obs
    )

    X = a[start:end, hvg_idx].X

    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(
        X,
        dtype=np.float64
    )

    labels = sample_labels[
        start:end
    ]

    for s in np.unique(labels):

        mask = labels == s
        j = sample_to_i[s]

        sums[j] += X[mask].sum(axis=0)
        counts[j] += mask.sum()

    if start % 50000 == 0:
        print(
            f"  processed {start:,}/{a.n_obs:,}"
        )


sample_X = sums / counts[:, None]


# ============================================================
# SAMPLE METADATA
# ============================================================

sample_meta = (
    obs.assign(
        sample=obs["sample"].astype(str)
    )
    .groupby(
        "sample",
        observed=True
    )
    .agg(
        drug=("Metadata_Drug", "first"),
        plate=("plate", "first"),
        n_cells=("Metadata_Drug", "size"),
    )
    .reindex(samples)
)


print("\nSample matrix:", sample_X.shape)

print(
    "cells/sample median:",
    int(np.median(counts))
)

print(
    "cells/sample min:",
    int(counts.min())
)

print(
    "cells/sample max:",
    int(counts.max())
)


# ============================================================
# BEFORE-CORRECTION PLATE QC
# ============================================================

# standardize each gene across samples
mu = sample_X.mean(axis=0)
sd = sample_X.std(axis=0, ddof=0)
sd[sd == 0] = 1

sample_Z = (
    sample_X - mu
) / sd


pca_before = PCA(
    n_components=10,
    random_state=0
)

PC_before = pca_before.fit_transform(
    sample_Z
)


def eta2(values, groups):

    values = np.asarray(values)
    groups = np.asarray(groups)

    grand = values.mean()

    ss_total = np.sum(
        (values - grand) ** 2
    )

    ss_between = 0.0

    for g in np.unique(groups):

        x = values[
            groups == g
        ]

        ss_between += len(x) * (
            x.mean() - grand
        ) ** 2

    return (
        ss_between / ss_total
        if ss_total > 0
        else 0
    )


plates = (
    sample_meta["plate"]
    .astype(str)
    .to_numpy()
)


print("\nPLATE eta² BEFORE CORRECTION")

before_eta = []

for pc in range(5):

    e = eta2(
        PC_before[:, pc],
        plates
    )

    before_eta.append(e)

    print(
        f"PC{pc+1}: {e:.6f}"
    )


# ============================================================
# PLATE CENTERING
#
# Same conceptual correction as final TAHOE:
# standardize genes across samples,
# then subtract each plate's gene-wise mean.
# ============================================================

sample_Z_corr = sample_Z.copy()

for plate in np.unique(plates):

    mask = plates == plate

    sample_Z_corr[mask] -= (
        sample_Z_corr[mask]
        .mean(
            axis=0,
            keepdims=True
        )
    )


# ============================================================
# AFTER-CORRECTION QC
# ============================================================

pca_after = PCA(
    n_components=10,
    random_state=0
)

PC_after = pca_after.fit_transform(
    sample_Z_corr
)


print("\nPLATE eta² AFTER CORRECTION")

after_eta = []

for pc in range(5):

    e = eta2(
        PC_after[:, pc],
        plates
    )

    after_eta.append(e)

    print(
        f"PC{pc+1}: {e:.8f}"
    )


# ============================================================
# SAMPLE -> DRUG CONSENSUS
# ============================================================

drug_labels = (
    sample_meta["drug"]
    .astype(str)
    .to_numpy()
)

drugs = sorted(
    np.unique(drug_labels)
)


drug_X = np.vstack([
    sample_Z_corr[
        drug_labels == d
    ].mean(axis=0)
    for d in drugs
])


print("\nDrug expression matrix:", drug_X.shape)


# ============================================================
# MATCH FINAL MORPHOLOGY DRUG SET
# ============================================================

morph = pd.read_csv(
    "LINCS/task2_final/"
    "LINCS_A549_FINAL_morphology_86x30PC.csv",
    index_col=0
)

shared = sorted(
    set(drugs)
    &
    set(morph.index.astype(str))
)


print(
    "Shared with final morphology:",
    len(shared)
)


drug_df = pd.DataFrame(
    drug_X,
    index=drugs,
    columns=genes
)


drug_df = drug_df.loc[
    shared
]


# ============================================================
# SAVE
# ============================================================

sample_df = pd.DataFrame(
    sample_Z_corr,
    index=samples,
    columns=genes
)

sample_df.insert(
    0,
    "drug",
    sample_meta["drug"].astype(str)
)

sample_df.insert(
    1,
    "plate",
    sample_meta["plate"].astype(str)
)


sample_df.to_parquet(
    OUT /
    "LINCS_A549_platecorrected_sample_2000hvg.parquet"
)


drug_df.to_parquet(
    OUT /
    "LINCS_A549_platecorrected_drug_2000hvg.parquet"
)


qc = pd.DataFrame({
    "PC": np.arange(1, 6),
    "plate_eta2_before": before_eta,
    "plate_eta2_after": after_eta,
})


qc.to_csv(
    OUT /
    "LINCS_A549_plate_QC.csv",
    index=False
)


print("\nSAVED")

print(
    OUT /
    "LINCS_A549_platecorrected_sample_2000hvg.parquet"
)

print(
    OUT /
    "LINCS_A549_platecorrected_drug_2000hvg.parquet"
)

print(
    OUT /
    "LINCS_A549_plate_QC.csv"
)

a.file.close()

print("\nDONE")
