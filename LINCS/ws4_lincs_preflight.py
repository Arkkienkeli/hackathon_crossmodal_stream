#!/usr/bin/env python3

import anndata as ad
import numpy as np
import pandas as pd


print("=" * 100)
print("LINCS A549 PREFLIGHT")
print("=" * 100)


# =============================================================================
# 1. EXPRESSION
# =============================================================================

p = "LINCS/data/lincs_expression_a549.h5ad"

print("\n" + "=" * 100)
print("EXPRESSION:", p)
print("=" * 100)

a = ad.read_h5ad(p, backed="r")

print("shape:", a.shape)
print("obs columns:", a.obs.columns.tolist())
print("var columns:", a.var.columns.tolist())
print("obsm:", list(a.obsm.keys()))
print("layers:", list(a.layers.keys()))
print("raw:", a.raw is not None)

for c in [
    "Metadata_Drug",
    "drug",
    "sample",
    "plate",
    "cell_line_id",
    "dose",
    "time",
    "pert_dose",
    "pert_time",
]:
    if c in a.obs.columns:
        print("\n---", c, "---")
        print("n unique:", a.obs[c].nunique(dropna=False))
        print(a.obs[c].value_counts(dropna=False).head(20))

if "highly_variable" in a.var.columns:
    hv = a.var["highly_variable"].astype(bool)
    print("\nHVG count:", int(hv.sum()))

print("\nExpression X sample:")
x = np.asarray(a[:500, :1000].X, dtype=np.float64)

print("min:", np.nanmin(x))
print("max:", np.nanmax(x))
print("mean:", np.nanmean(x))
print("median:", np.nanmedian(x))
print("negative fraction:", np.mean(x < 0))
print("zero fraction:", np.mean(x == 0))
print(
    "quantiles:",
    np.nanquantile(x, [0, .01, .25, .5, .75, .99, 1])
)

a.file.close()


# =============================================================================
# 2. MORPHOLOGY CONSENSUS
# =============================================================================

p = "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"

print("\n" + "=" * 100)
print("MORPHOLOGY CONSENSUS:", p)
print("=" * 100)

m = ad.read_h5ad(p)

X = np.asarray(m.X, dtype=np.float64)

print("shape:", m.shape)
print("finite fraction:", np.isfinite(X).mean())

print("\nMetadata:")

for c in [
    "Metadata_Drug",
    "Metadata_pert_iname",
    "Metadata_pert_id",
    "Metadata_dose_recode",
    "Metadata_n_replicates",
]:
    if c in m.obs.columns:
        print("\n---", c, "---")
        print("n unique:", m.obs[c].nunique(dropna=False))
        print(m.obs[c].value_counts(dropna=False).sort_index().head(30))


# =============================================================================
# 3. VALUE DISTRIBUTION
# =============================================================================

finite = X[np.isfinite(X)]

print("\nGLOBAL MORPHOLOGY VALUE DISTRIBUTION")

for q in [
    0,
    .0001,
    .001,
    .01,
    .1,
    .5,
    .9,
    .99,
    .999,
    .9999,
    1,
]:
    print(
        f"q={q:7.4f}:",
        np.quantile(finite, q)
    )


# =============================================================================
# 4. HOW MANY EXTREME VALUES?
# =============================================================================

print("\nEXTREME VALUE COUNTS")

for threshold in [
    10,
    100,
    1e3,
    1e4,
    1e6,
    1e9,
    1e12,
    1e15,
]:
    n = np.sum(np.abs(X) > threshold)

    print(
        f"|X| > {threshold:>10g}: "
        f"{n:>10d} / {X.size} "
        f"({100*n/X.size:.6f}%)"
    )


# =============================================================================
# 5. WHICH FEATURES CAUSE THE EXTREMES?
# =============================================================================

feature_max = np.nanmax(
    np.abs(X),
    axis=0
)

feature_q99 = np.nanquantile(
    np.abs(X),
    .99,
    axis=0
)

feature_sd = np.nanstd(
    X,
    axis=0
)


qc = pd.DataFrame({
    "feature": m.var_names.astype(str),
    "max_abs": feature_max,
    "q99_abs": feature_q99,
    "sd": feature_sd,
})

qc = qc.sort_values(
    "max_abs",
    ascending=False
)

print("\nTOP 30 FEATURES BY MAX ABSOLUTE VALUE")
print(
    qc.head(30).to_string(
        index=False
    )
)


# Count affected features
for threshold in [
    100,
    1e3,
    1e6,
    1e12,
]:
    n = (
        qc["max_abs"]
        >
        threshold
    ).sum()

    print(
        f"\nfeatures with max|X| > {threshold:g}: "
        f"{n}/{len(qc)}"
    )


qc.to_csv(
    "LINCS/data/lincs_morphology_feature_value_qc.csv",
    index=False
)


# =============================================================================
# 6. PCA VARIANCE DIAGNOSTIC — NO ALTERATION
# =============================================================================

from sklearn.decomposition import PCA

print("\n" + "=" * 100)
print("RAW CONSENSUS PCA DIAGNOSTIC")
print("=" * 100)

try:

    P = PCA(
        n_components=30,
        svd_solver="full"
    ).fit(X)

    ev = P.explained_variance_ratio_

    for i in range(10):
        print(
            f"PC{i+1}: "
            f"{ev[i]:.6f}"
        )

    print(
        "First 5 cumulative:",
        ev[:5].sum()
    )

    print(
        "First 30 cumulative:",
        ev[:30].sum()
    )

except Exception as e:

    print(
        "PCA failed:",
        repr(e)
    )


print("\nDONE")
