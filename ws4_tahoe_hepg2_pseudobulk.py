#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

RNA = Path("OpenScreen/data/hepg2_cells.h5ad")
MORPH = Path("OpenScreen/data/hepg2_morphology_final.parquet")
OUT = Path("OpenScreen/data")
OUT.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 100
N_HVG = 2000
TARGET_SUM = 1e4
RANDOM_STATE = 42

print("=" * 78)
print("TAHOE HepG2 SAMPLE-LEVEL PSEUDOBULK")
print("=" * 78)

# ------------------------------------------------------------------
# LOAD BACKED DATA
# ------------------------------------------------------------------

a = ad.read_h5ad(RNA, backed="r")
obs = a.obs.copy()

print("\nInput shape:", a.shape)
print("Cells:", a.n_obs)
print("Genes:", a.n_vars)
print("Drugs:", obs["drug"].nunique())
print("Samples:", obs["sample"].nunique())

# ------------------------------------------------------------------
# SAMPLE QC
# ------------------------------------------------------------------

sample_info = (
    obs.groupby("sample", observed=True)
    .agg(
        cells=("drug", "size"),
        drug=("drug", "first"),
        plate=("plate", "first"),
        moa=("moa-fine", "first"),
    )
)

sample_info["keep"] = (
    sample_info["cells"] >= MIN_CELLS
)

sample_info.to_csv(
    OUT / "hepg2_tahoe_sample_qc.csv"
)

removed = sample_info[
    ~sample_info["keep"]
].sort_values("cells")

print("\n" + "=" * 78)
print("SAMPLE FILTER")
print("=" * 78)

print("Minimum cells/sample:", MIN_CELLS)
print("Total samples:", len(sample_info))
print("Kept samples:", sample_info["keep"].sum())
print("Removed samples:", (~sample_info["keep"]).sum())

print("\nRemoved samples:")
print(removed.to_string())

keep_samples = (
    sample_info.index[
        sample_info["keep"]
    ]
    .astype(str)
    .tolist()
)

sample_to_row = {
    s: i
    for i, s in enumerate(keep_samples)
}

obs_sample = obs["sample"].astype(str).to_numpy()

# ------------------------------------------------------------------
# AGGREGATE RAW COUNTS WITHIN SAMPLE
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("AGGREGATING RAW COUNTS WITHIN SAMPLE")
print("=" * 78)

n_samples = len(keep_samples)
n_genes = a.n_vars

# 409 × 62710 float32 ~100 MB
counts = np.zeros(
    (n_samples, n_genes),
    dtype=np.float32
)

for i, sample in enumerate(keep_samples):

    idx = np.flatnonzero(
        obs_sample == sample
    )

    X = a.X[idx, :]

    row = np.asarray(
        X.sum(axis=0)
    ).ravel()

    counts[i, :] = row.astype(
        np.float32,
        copy=False
    )

    if (
        (i + 1) % 25 == 0
        or i == 0
        or i + 1 == n_samples
    ):
        print(
            f"aggregated {i+1}/{n_samples} samples",
            flush=True
        )

# Sample metadata in exact matrix order
meta = sample_info.loc[
    keep_samples
].copy()

meta.index = meta.index.astype(str)

var = a.var.copy()
var.index = a.var_names.astype(str)

a.file.close()

# ------------------------------------------------------------------
# SAVE RAW SAMPLE PSEUDOBULKS
# ------------------------------------------------------------------

pb_counts = ad.AnnData(
    X=counts.copy(),
    obs=meta.copy(),
    var=var.copy()
)

counts_file = (
    OUT /
    "hepg2_sample_pseudobulk_counts.h5ad"
)

pb_counts.write_h5ad(
    counts_file,
    compression="gzip"
)

print("\nSaved raw sample pseudobulks:")
print(counts_file)
print("Shape:", pb_counts.shape)

# ------------------------------------------------------------------
# NORMALIZE SAMPLE PSEUDOBULKS
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("NORMALIZATION")
print("=" * 78)

library_sizes_before = np.asarray(
    pb_counts.X.sum(axis=1)
).ravel()

print(
    "Raw sample library size:"
)
print(
    pd.Series(
        library_sizes_before
    ).describe()
)

sc.pp.normalize_total(
    pb_counts,
    target_sum=TARGET_SUM
)

sc.pp.log1p(pb_counts)

print(
    f"\nApplied normalize_total(target_sum={TARGET_SUM})"
)
print("Applied log1p")

# ------------------------------------------------------------------
# REMOVE GENES SEEN IN TOO FEW SAMPLE PSEUDOBULKS
# ------------------------------------------------------------------

before_genes = pb_counts.n_vars

sc.pp.filter_genes(
    pb_counts,
    min_cells=3
)

print(
    "\nGenes before expression filter:",
    before_genes
)
print(
    "Genes expressed in >=3 samples:",
    pb_counts.n_vars
)

# ------------------------------------------------------------------
# HVG SELECTION
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("HVG SELECTION")
print("=" * 78)

sc.pp.highly_variable_genes(
    pb_counts,
    n_top_genes=N_HVG,
    flavor="seurat"
)

hvg_mask = (
    pb_counts.var["highly_variable"]
    .to_numpy()
)

hvg_names = (
    pb_counts.var_names[hvg_mask]
    .astype(str)
    .tolist()
)

print("Selected HVGs:", len(hvg_names))

pd.DataFrame(
    {
        "gene": hvg_names
    }
).to_csv(
    OUT / "hepg2_hvg_2000.csv",
    index=False
)

# ------------------------------------------------------------------
# SAMPLE-LEVEL 2000-HVG MATRIX
# ------------------------------------------------------------------

Xh = np.asarray(
    pb_counts[:, hvg_mask].X,
    dtype=np.float32
)

sample_hvg = pd.DataFrame(
    Xh,
    index=pb_counts.obs_names.astype(str),
    columns=hvg_names
)

sample_hvg.to_parquet(
    OUT /
    "hepg2_sample_log1p_2000hvg.parquet"
)

print(
    "Saved sample-level HVG profiles:",
    sample_hvg.shape
)

# ------------------------------------------------------------------
# WITHIN-DRUG REPLICATE QC
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("WITHIN-DRUG SAMPLE REPRODUCIBILITY")
print("=" * 78)

drug_labels = (
    pb_counts.obs["drug"]
    .astype(str)
)

rep_rows = []

for drug in sorted(drug_labels.unique()):

    ids = drug_labels.index[
        drug_labels == drug
    ]

    Z = sample_hvg.loc[ids].to_numpy()

    n = len(Z)

    if n < 2:
        median_r = np.nan
    else:
        C = np.corrcoef(Z)
        iu = np.triu_indices(n, k=1)
        median_r = np.nanmedian(C[iu])

    rep_rows.append(
        {
            "drug": drug,
            "n_samples": n,
            "median_replicate_r": median_r,
            "total_cells": int(
                pb_counts.obs.loc[
                    ids, "cells"
                ].sum()
            )
        }
    )

rep_qc = pd.DataFrame(rep_rows)

rep_qc.to_csv(
    OUT /
    "hepg2_tahoe_within_drug_replicate_qc.csv",
    index=False
)

print(
    "\nReplicate correlation summary:"
)
print(
    rep_qc["median_replicate_r"]
    .describe()
)

print("\nLowest replicate correlations:")
print(
    rep_qc.sort_values(
        "median_replicate_r"
    )
    .head(15)
    .to_string(index=False)
)

print("\nHighest replicate correlations:")
print(
    rep_qc.sort_values(
        "median_replicate_r",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)

# ------------------------------------------------------------------
# AGGREGATE SAMPLE PROFILES TO DRUG LEVEL
# Equal weight per sample
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("DRUG-LEVEL AGGREGATION")
print("=" * 78)

tmp = sample_hvg.copy()
tmp["drug"] = drug_labels.values

drug_profiles = (
    tmp.groupby(
        "drug",
        observed=True
    )
    .mean()
    .sort_index()
)

drug_profiles.to_parquet(
    OUT /
    "hepg2_pseudobulk_2000hvg.parquet"
)

# Drug-level QC
drug_qc = (
    pb_counts.obs
    .groupby("drug", observed=True)
    .agg(
        samples=("cells", "size"),
        cells=("cells", "sum"),
        moa=("moa", "first"),
    )
    .sort_index()
)

drug_qc.to_csv(
    OUT /
    "hepg2_drug_aggregation_qc.csv"
)

print("Drug profile shape:", drug_profiles.shape)
print("\nSamples per drug:")
print(drug_qc["samples"].describe())

# ------------------------------------------------------------------
# MATCH TO OPENSCREEN MORPHOLOGY
# ------------------------------------------------------------------

morph = pd.read_parquet(MORPH)

shared = sorted(
    set(morph.index.astype(str))
    &
    set(drug_profiles.index.astype(str))
)

morph_only = sorted(
    set(morph.index.astype(str))
    -
    set(drug_profiles.index.astype(str))
)

rna_only = sorted(
    set(drug_profiles.index.astype(str))
    -
    set(morph.index.astype(str))
)

shared_profiles = (
    drug_profiles.loc[shared]
)

shared_profiles.to_parquet(
    OUT /
    "hepg2_pseudobulk_2000hvg_shared119.parquet"
)

print("\n" + "=" * 78)
print("FINAL TAHOE OUTPUT")
print("=" * 78)

print("TAHOE drug profiles:", len(drug_profiles))
print("Morphology drugs:", len(morph))
print("Shared drugs:", len(shared))

print("\nMorphology-only:")
print(morph_only)

print("\nTAHOE-only:")
print(rna_only)

print(
    "\nShared matrix shape:",
    shared_profiles.shape
)

print("\nSaved files:")

for f in [
    "hepg2_tahoe_sample_qc.csv",
    "hepg2_sample_pseudobulk_counts.h5ad",
    "hepg2_hvg_2000.csv",
    "hepg2_sample_log1p_2000hvg.parquet",
    "hepg2_tahoe_within_drug_replicate_qc.csv",
    "hepg2_drug_aggregation_qc.csv",
    "hepg2_pseudobulk_2000hvg.parquet",
    "hepg2_pseudobulk_2000hvg_shared119.parquet",
]:
    print(OUT / f)

