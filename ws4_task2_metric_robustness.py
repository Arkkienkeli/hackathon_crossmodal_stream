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

RANDOM_STATE = 0
N_PERM = 999
MAX_PCS = 50


# ============================================================
# HELPERS
# ============================================================

def zscore_cols(X):
    """
    Match notebook PC z-scoring:
    each PC is centered/scaled across drugs.
    ddof=0, matching the notebook.
    """
    X = np.asarray(X, dtype=np.float64)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0

    return (X - mu) / sd


def distance_matrix(X, metric):
    D = squareform(
        pdist(
            np.asarray(X, dtype=np.float64),
            metric=metric
        )
    )

    if not np.isfinite(D).all():
        raise RuntimeError(
            f"Non-finite distances for metric={metric}"
        )

    return D


def mantel_style(Dm, Dg, n_perm=999, seed=0):
    """
    Same basic permutation logic as original notebook.

    Spearman correlation between upper triangles.
    Drug labels of GEX matrix are permuted.
    """

    iu = np.triu_indices_from(Dm, k=1)

    vm = Dm[iu]
    vg = Dg[iu]

    obs = spearmanr(
        vm,
        vg
    ).statistic

    rng = np.random.RandomState(seed)
    idx = np.arange(Dm.shape[0])

    null = np.empty(
        n_perm,
        dtype=np.float64
    )

    for i in range(n_perm):
        perm = rng.permutation(idx)

        null[i] = spearmanr(
            vm,
            Dg[np.ix_(perm, perm)][iu]
        ).statistic

    p_two = (
        np.sum(
            np.abs(null)
            >= abs(obs)
        ) + 1
    ) / (
        n_perm + 1
    )

    p_one = (
        np.sum(
            null >= obs
        ) + 1
    ) / (
        n_perm + 1
    )

    return {
        "r": obs,
        "p_two": p_two,
        "p_one": p_one,
        "null_mean": np.mean(null),
        "null_sd": np.std(null),
        "null_p95": np.quantile(null, 0.95),
    }


# ============================================================
# LOAD FINAL QC-CONTROLLED DATA
# ============================================================

print("=" * 80)
print("TASK 2 METRIC / PCA ROBUSTNESS ANALYSIS")
print("=" * 80)

M = pd.read_parquet(MORPH)
G = pd.read_parquet(GEX)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

shared = sorted(
    set(M.index)
    & set(G.index)
)

M = M.loc[shared]
G = G.loc[shared]

assert M.index.equals(G.index)

print("\nMorphology:", M.shape)
print("Expression:", G.shape)
print("Shared drugs:", len(shared))


# ============================================================
# PCA
#
# IMPORTANT:
# No pre-PCA StandardScaler.
# This follows original notebook behavior more closely.
# ============================================================

pca_m = PCA(
    n_components=MAX_PCS,
    random_state=RANDOM_STATE
)

pca_g = PCA(
    n_components=MAX_PCS,
    random_state=RANDOM_STATE
)

Mpc50 = pca_m.fit_transform(
    M.to_numpy(dtype=np.float64)
)

Gpc50 = pca_g.fit_transform(
    G.to_numpy(dtype=np.float64)
)

print("\nVariance explained by first 50 PCs:")
print(
    "Morphology:",
    pca_m.explained_variance_ratio_.sum()
)

print(
    "Expression:",
    pca_g.explained_variance_ratio_.sum()
)


# ============================================================
# PART A
# METRIC ROBUSTNESS AT EXACTLY 30 PCs
# ============================================================

print("\n" + "=" * 80)
print("PART A: METRIC ROBUSTNESS AT 30 PCs")
print("=" * 80)

M30 = Mpc50[:, :30]
G30 = Gpc50[:, :30]

M30z = zscore_cols(M30)
G30z = zscore_cols(G30)

conditions = [
    (
        "zscore_PC + euclidean",
        M30z,
        G30z,
        "euclidean",
    ),
    (
        "raw_PC + euclidean",
        M30,
        G30,
        "euclidean",
    ),
    (
        "zscore_PC + cosine",
        M30z,
        G30z,
        "cosine",
    ),
    (
        "zscore_PC + correlation",
        M30z,
        G30z,
        "correlation",
    ),
    (
        "raw_PC + cosine",
        M30,
        G30,
        "cosine",
    ),
    (
        "raw_PC + correlation",
        M30,
        G30,
        "correlation",
    ),
]

metric_rows = []

for name, Xm, Xg, metric in conditions:

    Dm = distance_matrix(
        Xm,
        metric
    )

    Dg = distance_matrix(
        Xg,
        metric
    )

    res = mantel_style(
        Dm,
        Dg,
        n_perm=N_PERM,
        seed=RANDOM_STATE
    )

    metric_rows.append({
        "analysis": name,
        "n_pcs": 30,
        "metric": metric,
        "mantel_spearman_r": res["r"],
        "p_two_sided": res["p_two"],
        "p_one_sided": res["p_one"],
        "null_mean": res["null_mean"],
        "null_sd": res["null_sd"],
        "null_p95": res["null_p95"],
    })

    print("\n", name)
    print("  r      =", res["r"])
    print("  p(two) =", res["p_two"])
    print("  p(one) =", res["p_one"])


metric_df = pd.DataFrame(
    metric_rows
)

metric_df.to_csv(
    OUT /
    "hepg2_task2_metric_robustness_30pc.csv",
    index=False
)


# ============================================================
# PART B
# NUMBER-OF-PC SENSITIVITY
#
# Keep exact notebook metric:
# z-score PCs + Euclidean distance
# ============================================================

print("\n" + "=" * 80)
print("PART B: NUMBER-OF-PC SENSITIVITY")
print("z-scored PCs + Euclidean distance")
print("=" * 80)

pc_rows = []

for k in [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
]:

    Mk = zscore_cols(
        Mpc50[:, :k]
    )

    Gk = zscore_cols(
        Gpc50[:, :k]
    )

    Dm = distance_matrix(
        Mk,
        "euclidean"
    )

    Dg = distance_matrix(
        Gk,
        "euclidean"
    )

    res = mantel_style(
        Dm,
        Dg,
        n_perm=N_PERM,
        seed=RANDOM_STATE
    )

    morph_var = (
        pca_m.explained_variance_ratio_[
            :k
        ].sum()
    )

    gex_var = (
        pca_g.explained_variance_ratio_[
            :k
        ].sum()
    )

    pc_rows.append({
        "n_pcs": k,
        "morph_variance_explained":
            morph_var,
        "gex_variance_explained":
            gex_var,
        "mantel_spearman_r":
            res["r"],
        "p_two_sided":
            res["p_two"],
        "p_one_sided":
            res["p_one"],
        "null_p95":
            res["null_p95"],
    })

    print(
        f"{k:2d} PCs | "
        f"M var={morph_var:.3f} | "
        f"G var={gex_var:.3f} | "
        f"r={res['r']:.4f} | "
        f"p={res['p_two']:.4f}"
    )


pc_df = pd.DataFrame(
    pc_rows
)

pc_df.to_csv(
    OUT /
    "hepg2_task2_pc_number_sensitivity.csv",
    index=False
)


# ============================================================
# PART C
# COMPACT FINAL TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL METRIC ROBUSTNESS TABLE")
print("=" * 80)

print(
    metric_df[
        [
            "analysis",
            "mantel_spearman_r",
            "p_two_sided",
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("FINAL PC-NUMBER SENSITIVITY TABLE")
print("=" * 80)

print(
    pc_df[
        [
            "n_pcs",
            "morph_variance_explained",
            "gex_variance_explained",
            "mantel_spearman_r",
            "p_two_sided",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE ONE COMPACT SUMMARY
# ============================================================

primary = metric_df[
    metric_df["analysis"]
    == "zscore_PC + euclidean"
].iloc[0]

summary = pd.DataFrame(
    [{
        "n_drugs":
            len(shared),
        "n_drug_pairs":
            len(shared)
            * (len(shared) - 1)
            // 2,
        "primary_method":
            "30 PC, PC-zscore, Euclidean",
        "primary_r":
            primary[
                "mantel_spearman_r"
            ],
        "primary_p":
            primary[
                "p_two_sided"
            ],
        "original_notebook_target_r":
            0.334,
        "original_notebook_target_p":
            0.001,
    }]
)

summary.to_csv(
    OUT /
    "hepg2_task2_robustness_summary.csv",
    index=False
)

print("\nSaved:")
for f in [
    "hepg2_task2_metric_robustness_30pc.csv",
    "hepg2_task2_pc_number_sensitivity.csv",
    "hepg2_task2_robustness_summary.csv",
]:
    print(OUT / f)

