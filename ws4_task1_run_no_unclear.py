#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import anndata as ad

from crossmodal_moa import run_all

MORPH = "OpenScreen/data/hepg2_morphology_final.parquet"
GEX = "hepg2_platecorrected_drug_2000hvg.parquet"
META = "OpenScreen/data/hepg2_sample_pseudobulk_counts.h5ad"

OUT_TABLE = "OpenScreen/task1_moa_results_no_unclear.csv"
OUT_RETRIEVAL = "OpenScreen/task1_crossmodal_retrieval_no_unclear.json"
OUT_LABELS = "OpenScreen/task1_moa_labels_no_unclear.csv"

print("=" * 80)
print("WS4 TASK 1 — MoA PREDICTION")
print("=" * 80)

# ------------------------------------------------------------
# Load frozen morphology + plate-corrected expression
# ------------------------------------------------------------

M = pd.read_parquet(MORPH)
G = pd.read_parquet(GEX)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

shared = sorted(set(M.index) & set(G.index))

M = M.loc[shared].copy()
G = G.loc[shared].copy()

print("Morphology:", M.shape)
print("Expression:", G.shape)
print("Shared drugs:", len(shared))

assert list(M.index) == list(G.index)

# ------------------------------------------------------------
# Get MoA annotation from TAHOE retained sample metadata
# ------------------------------------------------------------

a = ad.read_h5ad(META, backed="r")
obs = a.obs.copy()
a.file.close()

print("\nMetadata columns:", list(obs.columns))

drug_col = "drug"

if "moa" in obs.columns:
    moa_col = "moa"
elif "moa-fine" in obs.columns:
    moa_col = "moa-fine"
else:
    raise RuntimeError(
        "Could not find MoA column. Available: "
        + ", ".join(obs.columns.astype(str))
    )

obs[drug_col] = obs[drug_col].astype(str)
obs[moa_col] = obs[moa_col].astype(str)

# Ignore missing-string labels if present
obs = obs[
    obs[moa_col].notna()
    &
    (~obs[moa_col].isin(["nan", "None", ""]))
].copy()

# Verify each drug maps to a single MoA
n_moa = (
    obs.groupby(drug_col)[moa_col]
    .nunique()
)

bad = n_moa[n_moa > 1]

if len(bad):
    print("\nDrugs with conflicting MoA labels:")
    print(bad)
    raise RuntimeError("Conflicting MoA annotation per drug.")

moa_map = (
    obs.drop_duplicates(drug_col)
       .set_index(drug_col)[moa_col]
)

# Keep only compounds with MoA labels
with_moa = [
    d for d in shared
    if d in moa_map.index
]

M = M.loc[with_moa]
G = G.loc[with_moa]

y = moa_map.loc[with_moa].to_numpy()
drugs = np.asarray(with_moa)

# Exclude "unclear": this is a missing/heterogeneous annotation,
# not a biological MoA class.
mask_clear = y != "unclear"

M = M.iloc[mask_clear].copy()
G = G.iloc[mask_clear].copy()
y = y[mask_clear]
drugs = drugs[mask_clear]

print("\nExcluded 'unclear' MoA category")
print("Drugs remaining before rare-class filter:", len(drugs))

print("\nDrugs with MoA:", len(drugs))
print("Raw MoA classes:", pd.Series(y).nunique())

print("\nRaw class counts:")
print(pd.Series(y).value_counts().to_string())

pd.DataFrame({
    "drug": drugs,
    "moa": y
}).to_csv(
    OUT_LABELS,
    index=False
)

# ------------------------------------------------------------
# Run Task 1
#
# crossmodal_moa.py:
# - drops classes with <3 compounds
# - repeated stratified CV
# - preprocessing/PCA/CCA fit inside CV
# - label-permutation controls
# ------------------------------------------------------------

table, retrieval = run_all(
    M.to_numpy(np.float64),
    G.to_numpy(np.float64),
    y,
    drugs=drugs,
    min_count=3,
    n_pcs=30,
    cca_components=10,
    run_permutation=True,
)

table.to_csv(
    OUT_TABLE,
    index=False
)

with open(
    OUT_RETRIEVAL,
    "w"
) as f:
    json.dump(
        retrieval,
        f,
        indent=2
    )

print("\n" + "=" * 80)
print("FINAL TASK 1 RESULTS")
print("=" * 80)

print(
    table.to_string(
        index=False
    )
)

print("\nHeld-out cross-modal retrieval:")
print(
    json.dumps(
        retrieval,
        indent=2
    )
)

# ------------------------------------------------------------
# Simple fusion deltas
# ------------------------------------------------------------

lookup = table.set_index("experiment")

m = lookup.loc[
    "A morphology only",
    "bal_acc"
]

g = lookup.loc[
    "B expression only",
    "bal_acc"
]

fusion = lookup.loc[
    "C concat (early fusion)",
    "bal_acc"
]

cca_concat = lookup.loc[
    "D CCA shared (concat)",
    "bal_acc"
]

cca_mean = lookup.loc[
    "D CCA shared (mean)",
    "bal_acc"
]

best_single = max(m, g)

print("\nFusion comparison:")
print(f"  morphology only : {m:.4f}")
print(f"  expression only : {g:.4f}")
print(f"  best single     : {best_single:.4f}")
print(f"  early fusion    : {fusion:.4f}")
print(f"  fusion delta    : {fusion-best_single:+.4f}")
print(f"  CCA concat      : {cca_concat:.4f}")
print(f"  CCA mean        : {cca_mean:.4f}")

print("\nSaved:")
print(" ", OUT_TABLE)
print(" ", OUT_RETRIEVAL)
print(" ", OUT_LABELS)

print("\nIMPORTANT CAVEAT:")
print(
    "The current 2,000 HVGs were selected globally before this supervised "
    "cross-validation. PCA/CCA/classifier fitting occurs inside folds, but "
    "HVG selection itself is not fold-restricted. Treat Task 1 as an "
    "exploratory workstream check rather than a final leakage-free benchmark."
)
