#!/usr/bin/env python3

"""
LINCS A549 validation of WS4 Task 2.

IMPORTANT
---------
Expression:
    Already processed.
    Uses existing X_pca from lincs_expression_a549.h5ad.
    No renormalization, no log1p, no HVG reselection.

Morphology:
    Already MAD-normalized / feature-selected / consensus-aggregated.
    Uses lincs_morphology_a549_batch1_consensus.h5ad directly.

Because some morphology features contain extreme numerical values,
we run several transparent stability filters instead of silently
choosing one arbitrary preprocessing rule.

Analysis:
    1. morphology PCA -> 30 PCs
    2. average six dose profiles per drug
    3. expression existing 30 PCs -> average cells per drug
    4. match drugs
    5. z-score PC axes
    6. Euclidean drug-pair geometry
    7. drug-label permutation
    8. perturbation-strength correlation
    9. direction-only correlation
    10. PC1-drop direction sensitivity
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

N_PCS = 30
N_PERM = 999
SEED = 0

MORPH_FILE = Path(
    "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"
)

EXPR_FILE = Path(
    "LINCS/data/lincs_expression_a549.h5ad"
)

OUTDIR = Path(
    "LINCS/task2_validation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def zscore_columns(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)

    sd[sd == 0] = 1.0

    return (
        X - mu
    ) / sd


def unit_rows(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return X / norms


def distance_matrix(X):

    return squareform(
        pdist(
            X,
            metric="euclidean"
        )
    )


def geometry_test(Dm, Dg, n_perm=N_PERM, seed=SEED):

    n = Dm.shape[0]

    iu = np.triu_indices(
        n,
        k=1
    )

    x = Dm[iu]
    y = Dg[iu]

    r = float(
        spearmanr(
            x,
            y
        ).statistic
    )

    rng = np.random.default_rng(seed)

    null = np.zeros(
        n_perm,
        dtype=float
    )

    idx = np.arange(n)

    for b in range(n_perm):

        perm = rng.permutation(idx)

        Dgp = Dg[
            np.ix_(
                perm,
                perm
            )
        ]

        null[b] = spearmanr(
            x,
            Dgp[iu]
        ).statistic

    # positive one-sided test
    p_one = (
        1
        +
        np.sum(
            null >= r
        )
    ) / (
        n_perm + 1
    )

    # two-sided test
    p_two = (
        1
        +
        np.sum(
            np.abs(null) >= abs(r)
        )
    ) / (
        n_perm + 1
    )

    return r, p_one, p_two, null


def strength_test(sm, sg, n_perm=N_PERM, seed=SEED):

    rho = float(
        spearmanr(
            sm,
            sg
        ).statistic
    )

    rng = np.random.default_rng(seed)

    null = np.zeros(
        n_perm,
        dtype=float
    )

    for b in range(n_perm):

        perm = rng.permutation(
            len(sg)
        )

        null[b] = spearmanr(
            sm,
            sg[perm]
        ).statistic

    p = (
        1
        +
        np.sum(
            null >= rho
        )
    ) / (
        n_perm + 1
    )

    return rho, p, null


# ============================================================
# LOAD MORPHOLOGY
# ============================================================

print("=" * 90)
print("LINCS A549 — TASK 2 VALIDATION")
print("=" * 90)

m = ad.read_h5ad(
    MORPH_FILE
)

Xm_all = np.asarray(
    m.X,
    dtype=np.float64
)

morph_drugs = (
    m.obs["Metadata_Drug"]
    .astype(str)
    .to_numpy()
)

print("\nMORPHOLOGY")
print("shape:", m.shape)
print(
    "drugs:",
    m.obs["Metadata_Drug"].nunique()
)
print(
    "dose levels:",
    sorted(
        m.obs["Metadata_dose_recode"].unique()
    )
)

print(
    "features:",
    Xm_all.shape[1]
)

print(
    "global max |value|:",
    np.nanmax(
        np.abs(Xm_all)
    )
)


# ============================================================
# LOAD EXPRESSION
# ============================================================

print("\n" + "=" * 90)
print("EXPRESSION — ALREADY PROCESSED")
print("=" * 90)

g = ad.read_h5ad(
    EXPR_FILE,
    backed="r"
)

print(
    "shape:",
    g.shape
)

print(
    "drugs:",
    g.obs["Metadata_Drug"].nunique()
)

print(
    "samples:",
    g.obs["sample"].nunique()
)

print(
    "plates:",
    g.obs["plate"].nunique()
)

print(
    "HVGs:",
    int(
        g.var["highly_variable"].sum()
    )
)

print(
    "obsm:",
    list(
        g.obsm.keys()
    )
)


if "X_pca" not in g.obsm.keys():

    raise RuntimeError(
        "X_pca missing from processed expression file."
    )


# Existing processed expression representation
Gcell = np.asarray(
    g.obsm["X_pca"][:, :N_PCS],
    dtype=np.float64
)

Gtmp = pd.DataFrame(
    Gcell,
    columns=[
        f"PC{i+1}"
        for i in range(Gcell.shape[1])
    ]
)

Gtmp["drug"] = (
    g.obs["Metadata_Drug"]
    .astype(str)
    .to_numpy()
)


Gdrug = (
    Gtmp
    .groupby(
        "drug",
        observed=True
    )
    .mean()
)


print(
    "drug-level expression:",
    Gdrug.shape
)

g.file.close()


# ============================================================
# MORPHOLOGY FEATURE QC
# ============================================================

feature_names = np.asarray(
    m.var_names.astype(str)
)

finite_feature = np.all(
    np.isfinite(Xm_all),
    axis=0
)

feature_max = np.nanmax(
    np.abs(Xm_all),
    axis=0
)


qc = pd.DataFrame({
    "feature": feature_names,
    "finite": finite_feature,
    "max_abs": feature_max,
})


qc.sort_values(
    "max_abs",
    ascending=False
).to_csv(
    OUTDIR /
    "lincs_morphology_feature_numeric_qc.csv",
    index=False
)


print("\n" + "=" * 90)
print("MORPHOLOGY NUMERIC QC")
print("=" * 90)

for threshold in [
    100,
    1_000,
    1_000_000,
    1e12,
]:

    good = (
        finite_feature
        &
        (
            feature_max
            <= threshold
        )
    )

    print(
        f"max|X| <= {threshold:g}: "
        f"{good.sum()} / {len(good)} features"
    )


# ============================================================
# RUN ONE ANALYSIS
# ============================================================

def run_analysis(label, threshold):

    print("\n")
    print("#" * 90)
    print("ANALYSIS:", label)
    print("#" * 90)

    if np.isinf(threshold):

        keep = finite_feature.copy()

    else:

        keep = (
            finite_feature
            &
            (
                feature_max
                <= threshold
            )
        )


    X = Xm_all[
        :,
        keep
    ]


    n_features = X.shape[1]


    print(
        "Morphology features retained:",
        n_features
    )

    if n_features < N_PCS:

        print(
            "SKIP — fewer than 30 features."
        )

        return None


    # --------------------------------------------------------
    # Morphology PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_PCS,
        random_state=SEED,
        svd_solver="full"
    )

    M_dose_pc = pca.fit_transform(
        X
    )


    print(
        "Morph PC1 variance:",
        f"{pca.explained_variance_ratio_[0]:.4f}"
    )

    print(
        "Morph first 5 PCs:",
        f"{pca.explained_variance_ratio_[:5].sum():.4f}"
    )


    # --------------------------------------------------------
    # Average 6 dose profiles -> one morphology profile/drug
    # --------------------------------------------------------

    Mtmp = pd.DataFrame(
        M_dose_pc,
        columns=[
            f"PC{i+1}"
            for i in range(N_PCS)
        ]
    )

    Mtmp["drug"] = morph_drugs


    Mdrug = (
        Mtmp
        .groupby(
            "drug",
            observed=True
        )
        .mean()
    )


    # --------------------------------------------------------
    # Match morphology and GE
    # --------------------------------------------------------

    shared = sorted(
        set(Mdrug.index)
        &
        set(Gdrug.index)
    )


    M = Mdrug.loc[
        shared
    ].to_numpy(
        dtype=np.float64
    )

    G = Gdrug.loc[
        shared
    ].to_numpy(
        dtype=np.float64
    )


    n = len(shared)

    n_pairs = (
        n
        *
        (
            n - 1
        )
    ) // 2


    print(
        "Shared drugs:",
        n
    )

    print(
        "Drug pairs:",
        n_pairs
    )


    # --------------------------------------------------------
    # z-score PC axes exactly as Task 2
    # --------------------------------------------------------

    Mz = zscore_columns(M)
    Gz = zscore_columns(G)


    # --------------------------------------------------------
    # Overall geometry
    # --------------------------------------------------------

    Dm = distance_matrix(Mz)
    Dg = distance_matrix(Gz)

    r_all, p_all, p2_all, null_all = geometry_test(
        Dm,
        Dg
    )


    # --------------------------------------------------------
    # Strength
    # --------------------------------------------------------

    sm = np.linalg.norm(
        Mz,
        axis=1
    )

    sg = np.linalg.norm(
        Gz,
        axis=1
    )

    rho_strength, p_strength, null_strength = strength_test(
        sm,
        sg
    )


    # --------------------------------------------------------
    # Direction only
    # --------------------------------------------------------

    Mdir = unit_rows(
        Mz
    )

    Gdir = unit_rows(
        Gz
    )

    r_dir, p_dir, p2_dir, null_dir = geometry_test(
        distance_matrix(Mdir),
        distance_matrix(Gdir)
    )


    # --------------------------------------------------------
    # PC1 dropped direction
    # --------------------------------------------------------

    M_no1 = unit_rows(
        Mz[:, 1:]
    )

    G_no1 = unit_rows(
        Gz[:, 1:]
    )

    r_no1, p_no1, _, _ = geometry_test(
        distance_matrix(M_no1),
        distance_matrix(G_no1)
    )


    print()
    print(
        "OVERALL:"
    )

    print(
        f"  r = {r_all:.4f}"
    )

    print(
        f"  permutation p = {p_all:.4f}"
    )


    print(
        "STRENGTH:"
    )

    print(
        f"  rho = {rho_strength:.4f}"
    )

    print(
        f"  permutation p = {p_strength:.4f}"
    )


    print(
        "DIRECTION:"
    )

    print(
        f"  r = {r_dir:.4f}"
    )

    print(
        f"  permutation p = {p_dir:.4f}"
    )


    print(
        "DIRECTION PC1 DROP:"
    )

    print(
        f"  r = {r_no1:.4f}"
    )

    print(
        f"  permutation p = {p_no1:.4f}"
    )


    return {
        "analysis": label,
        "max_abs_threshold": threshold,
        "n_morph_features": n_features,
        "morph_pc1_variance": pca.explained_variance_ratio_[0],
        "n_drugs": n,
        "n_pairs": n_pairs,
        "overall_r": r_all,
        "overall_perm_p": p_all,
        "strength_rho": rho_strength,
        "strength_perm_p": p_strength,
        "direction_r": r_dir,
        "direction_perm_p": p_dir,
        "direction_pc1drop_r": r_no1,
        "direction_pc1drop_p": p_no1,
    }


# ============================================================
# RUN SENSITIVITY SERIES
# ============================================================

results = []


tests = [
    (
        "strict_100",
        100.0
    ),
    (
        "moderate_1000",
        1_000.0
    ),
    (
        "pathological_only_1e6",
        1_000_000.0
    ),
    (
        "pathological_only_1e12",
        1e12
    ),
    (
        "all_finite_original",
        np.inf
    ),
]


for label, threshold in tests:

    out = run_analysis(
        label,
        threshold
    )

    if out is not None:

        results.append(
            out
        )


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)


summary.to_csv(
    OUTDIR /
    "lincs_task2_sensitivity_summary.csv",
    index=False
)


print("\n")
print("=" * 110)
print("FINAL SENSITIVITY SUMMARY")
print("=" * 110)

cols = [
    "analysis",
    "n_morph_features",
    "morph_pc1_variance",
    "n_drugs",
    "n_pairs",
    "overall_r",
    "overall_perm_p",
    "strength_rho",
    "strength_perm_p",
    "direction_r",
    "direction_perm_p",
    "direction_pc1drop_r",
]


print(
    summary[
        cols
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(
    OUTDIR /
    "lincs_task2_sensitivity_summary.csv"
)

print(
    OUTDIR /
    "lincs_morphology_feature_numeric_qc.csv"
)

print("\nDONE")
