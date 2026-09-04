#!/usr/bin/env python3

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

HEPG2_MORPH = Path(
    "OpenScreen/data/hepg2_morphology_final.parquet"
)

HEPG2_EXPR = Path(
    "OpenScreen/data/hepg2_platecorrected_drug_2000hvg.parquet"
)

A549_MORPH = Path(
    "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"
)

A549_EXPR = Path(
    "LINCS/final_rebuild/A549_platecorrected_drug_2000HVG.parquet"
)

OUT = Path(
    "cross_dataset_common_compounds"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

N_PCS = 30
N_PERM = 999
SEED = 0


# ============================================================
# HELPERS
# ============================================================

def norm_name(x):
    return (
        str(x)
        .strip()
        .lower()
    )


def zscore_cols(X):

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    mu = X.mean(
        axis=0,
        keepdims=True,
    )

    sd = X.std(
        axis=0,
        ddof=0,
        keepdims=True,
    )

    sd[sd == 0] = 1.0

    return (
        X - mu
    ) / sd


def pca_z(X):

    ncomp = min(
        N_PCS,
        X.shape[0] - 1,
        X.shape[1],
    )

    pc = PCA(
        n_components=ncomp,
        svd_solver="full",
    ).fit_transform(
        X
    )

    return zscore_cols(
        pc
    )


def upper(D):

    return D[
        np.triu_indices(
            D.shape[0],
            1,
        )
    ]


def distance_matrix(Z):

    return squareform(
        pdist(
            Z,
            metric="euclidean",
        )
    )


def direction_distance(Z):

    norms = np.linalg.norm(
        Z,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    U = Z / norms

    return squareform(
        pdist(
            U,
            metric="euclidean",
        )
    )


def cosine_distance(Z):

    norms = np.linalg.norm(
        Z,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    U = Z / norms

    C = 1 - (
        U @ U.T
    )

    np.fill_diagonal(
        C,
        0.0,
    )

    return C


def permutation_p(
    Dm,
    Dg,
    observed,
    n_perm=N_PERM,
    seed=SEED,
):

    rng = np.random.default_rng(
        seed
    )

    n = Dm.shape[0]

    null = np.zeros(
        n_perm,
        dtype=float,
    )

    for b in range(
        n_perm
    ):

        p = rng.permutation(
            n
        )

        Dgp = Dg[
            np.ix_(
                p,
                p,
            )
        ]

        null[b] = spearmanr(
            upper(Dm),
            upper(Dgp),
        ).statistic

    pval = (
        1
        +
        np.sum(
            null >= observed
        )
    ) / (
        n_perm + 1
    )

    return float(
        pval
    )


def strength_perm_p(
    sm,
    sg,
    observed,
    n_perm=N_PERM,
    seed=SEED,
):

    rng = np.random.default_rng(
        seed
    )

    null = np.zeros(
        n_perm,
        dtype=float,
    )

    for b in range(
        n_perm
    ):

        p = rng.permutation(
            len(sg)
        )

        null[b] = spearmanr(
            sm,
            sg[p],
        ).statistic

    return float(
        (
            1
            +
            np.sum(
                null >= observed
            )
        )
        /
        (
            n_perm + 1
        )
    )


def analyze(
    M,
    G,
    drugs,
    label,
):

    Mz = pca_z(
        M
    )

    Gz = pca_z(
        G
    )

    Dm = distance_matrix(
        Mz
    )

    Dg = distance_matrix(
        Gz
    )

    overall = spearmanr(
        upper(Dm),
        upper(Dg),
    ).statistic

    overall_p = permutation_p(
        Dm,
        Dg,
        overall,
    )

    sm = np.linalg.norm(
        Mz,
        axis=1,
    )

    sg = np.linalg.norm(
        Gz,
        axis=1,
    )

    strength = spearmanr(
        sm,
        sg,
    ).statistic

    strength_p = strength_perm_p(
        sm,
        sg,
        strength,
    )

    Dm_dir = direction_distance(
        Mz
    )

    Dg_dir = direction_distance(
        Gz
    )

    direction = spearmanr(
        upper(Dm_dir),
        upper(Dg_dir),
    ).statistic

    direction_p = permutation_p(
        Dm_dir,
        Dg_dir,
        direction,
    )

    Dm_cos = cosine_distance(
        Mz
    )

    Dg_cos = cosine_distance(
        Gz
    )

    cosine = spearmanr(
        upper(Dm_cos),
        upper(Dg_cos),
    ).statistic

    # Per-drug neighbourhood concordance
    rows = []

    n = len(
        drugs
    )

    for i, drug in enumerate(
        drugs
    ):

        keep = (
            np.arange(n)
            != i
        )

        r = spearmanr(
            Dm[i, keep],
            Dg[i, keep],
        ).statistic

        rows.append(
            {
                "drug": drug,
                f"{label}_neighborhood_r":
                    r,
                f"{label}_morph_strength":
                    sm[i],
                f"{label}_expr_strength":
                    sg[i],
            }
        )

    ranking = pd.DataFrame(
        rows
    )

    result = {
        "cell_line": label,
        "n_drugs": n,
        "n_pairs":
            n * (n - 1) // 2,
        "overall_r":
            overall,
        "overall_perm_p":
            overall_p,
        "strength_rho":
            strength,
        "strength_perm_p":
            strength_p,
        "direction_r":
            direction,
        "direction_perm_p":
            direction_p,
        "cosine_r":
            cosine,
    }

    return (
        result,
        ranking,
    )


# ============================================================
# LOAD HEPG2
# ============================================================

print(
    "=" * 100
)

print(
    "FOUR-WAY COMMON COMPOUND ANALYSIS"
)

print(
    "HepG2 morphology ∩ HepG2 GE ∩ "
    "A549 morphology ∩ A549 GE"
)

print(
    "=" * 100
)


H_M = pd.read_parquet(
    HEPG2_MORPH
)

H_G = pd.read_parquet(
    HEPG2_EXPR
)


print(
    "\nHepG2 morphology:",
    H_M.shape,
)

print(
    "HepG2 expression:",
    H_G.shape,
)


# ============================================================
# LOAD / BUILD A549 MORPHOLOGY
# ============================================================

a_m = ad.read_h5ad(
    A549_MORPH
)

MX = np.asarray(
    a_m.X,
    dtype=np.float64,
)


# same pathological-feature exclusion
max_abs = np.nanmax(
    np.abs(MX),
    axis=0,
)

keep_features = (
    np.all(
        np.isfinite(MX),
        axis=0,
    )
    &
    (
        max_abs <= 1e6
    )
)

MX = MX[
    :,
    keep_features
]


drug_labels = (
    a_m.obs[
        "Metadata_Drug"
    ]
    .astype(str)
    .to_numpy()
)


a549_morph_drugs = sorted(
    np.unique(
        drug_labels
    )
)


A_M = pd.DataFrame(
    np.vstack(
        [
            MX[
                drug_labels == d
            ].mean(
                axis=0
            )
            for d in
            a549_morph_drugs
        ]
    ),
    index=a549_morph_drugs,
)


A_G = pd.read_parquet(
    A549_EXPR
)


print(
    "\nA549 morphology:",
    A_M.shape,
)

print(
    "A549 expression:",
    A_G.shape,
)


# ============================================================
# NORMALIZED NAME LOOKUPS
# ============================================================

h_m_lookup = {
    norm_name(x): x
    for x in H_M.index
}

h_g_lookup = {
    norm_name(x): x
    for x in H_G.index
}

a_m_lookup = {
    norm_name(x): x
    for x in A_M.index
}

a_g_lookup = {
    norm_name(x): x
    for x in A_G.index
}


# ============================================================
# FOUR-WAY INTERSECTION
# ============================================================

common_keys = sorted(
    set(
        h_m_lookup
    )
    &
    set(
        h_g_lookup
    )
    &
    set(
        a_m_lookup
    )
    &
    set(
        a_g_lookup
    )
)


print(
    "\nFOUR-WAY COMMON COMPOUNDS:",
    len(
        common_keys
    ),
)


# use HepG2 spelling for display
common_drugs = [
    str(
        h_m_lookup[k]
    )
    for k in common_keys
]


pd.DataFrame(
    {
        "drug": common_drugs,
    }
).to_csv(
    OUT /
    "fourway_common_compounds.csv",
    index=False,
)


print(
    "\nCOMMON COMPOUNDS"
)

print(
    "\n".join(
        common_drugs
    )
)


# ============================================================
# ALIGNED MATRICES
# ============================================================

HM = H_M.loc[
    [
        h_m_lookup[k]
        for k in common_keys
    ]
].to_numpy(
    dtype=np.float64
)


HG = H_G.loc[
    [
        h_g_lookup[k]
        for k in common_keys
    ]
].to_numpy(
    dtype=np.float64
)


AM = A_M.loc[
    [
        a_m_lookup[k]
        for k in common_keys
    ]
].to_numpy(
    dtype=np.float64
)


AG = A_G.loc[
    [
        a_g_lookup[k]
        for k in common_keys
    ]
].to_numpy(
    dtype=np.float64
)


print(
    "\nAligned matrix shapes:"
)

print(
    "HepG2 morphology:",
    HM.shape,
)

print(
    "HepG2 expression:",
    HG.shape,
)

print(
    "A549 morphology:",
    AM.shape,
)

print(
    "A549 expression:",
    AG.shape,
)


# ============================================================
# RUN TASK2 ON IDENTICAL COMPOUNDS
# ============================================================

h_result, h_rank = analyze(
    HM,
    HG,
    common_drugs,
    "HepG2",
)


a_result, a_rank = analyze(
    AM,
    AG,
    common_drugs,
    "A549",
)


summary = pd.DataFrame(
    [
        h_result,
        a_result,
    ]
)


summary.to_csv(
    OUT /
    "fourway_same_compounds_task2_summary.csv",
    index=False,
)


print(
    "\n" + "=" * 100
)

print(
    "TASK 2 — IDENTICAL COMPOUND SET"
)

print(
    "=" * 100
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# COMPARE PER-DRUG CONCORDANCE
# ============================================================

rank = h_rank.merge(
    a_rank,
    on="drug",
    how="inner",
)


rank[
    "mean_neighborhood_r"
] = (
    rank[
        "HepG2_neighborhood_r"
    ]
    +
    rank[
        "A549_neighborhood_r"
    ]
) / 2


rank[
    "minimum_neighborhood_r"
] = np.minimum(
    rank[
        "HepG2_neighborhood_r"
    ],
    rank[
        "A549_neighborhood_r"
    ],
)


rank[
    "cellline_difference"
] = np.abs(
    rank[
        "HepG2_neighborhood_r"
    ]
    -
    rank[
        "A549_neighborhood_r"
    ]
)


rank = rank.sort_values(
    "minimum_neighborhood_r",
    ascending=False,
)


rank.to_csv(
    OUT /
    "fourway_common_drug_concordance.csv",
    index=False,
)


print(
    "\n" + "=" * 100
)

print(
    "TOP COMMON COMPOUNDS — STRONG IN BOTH CELL LINES"
)

print(
    "=" * 100
)

print(
    rank[
        [
            "drug",
            "HepG2_neighborhood_r",
            "A549_neighborhood_r",
            "minimum_neighborhood_r",
            "mean_neighborhood_r",
        ]
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 100
)

print(
    "LARGEST CELL-LINE DIFFERENCES"
)

print(
    "=" * 100
)

print(
    rank.sort_values(
        "cellline_difference",
        ascending=False,
    )[
        [
            "drug",
            "HepG2_neighborhood_r",
            "A549_neighborhood_r",
            "cellline_difference",
        ]
    ]
    .head(
        15
    )
    .to_string(
        index=False
    )
)


print(
    "\nSaved to:",
    OUT
)

print(
    "DONE"
)
