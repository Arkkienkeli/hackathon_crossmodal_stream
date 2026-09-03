#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MORPH = Path("OpenScreen/data/hepg2_morphology_final.parquet")
GEX = Path("OpenScreen/data/hepg2_pseudobulk_2000hvg_shared119.parquet")

OUT = Path("OpenScreen/task2")
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = 9999
RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def row_corr_distance(X):
    """
    Pairwise Pearson-correlation distance between drug profiles.
    Distance = 1 - Pearson r
    """
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    return 1.0 - C


def upper(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu], iu


def mantel_spearman(A, B, n_perm=9999, seed=42):
    """
    Mantel-style Spearman correlation between upper triangles.

    Drug labels of B are permuted while preserving the B
    distance matrix structure.
    """

    va, iu = upper(A)
    vb, _ = upper(B)

    # ranks for Spearman
    ra = rankdata(va)
    rb = rankdata(vb)

    obs = pearsonr(ra, rb).statistic

    # Make symmetric matrix containing global upper-triangle ranks.
    RB = np.zeros_like(B, dtype=float)
    RB[iu] = rb
    RB[(iu[1], iu[0])] = rb

    rng = np.random.default_rng(seed)

    null = np.empty(n_perm, dtype=float)

    ra0 = ra - ra.mean()
    ra_den = np.sqrt(np.sum(ra0 ** 2))

    n = A.shape[0]

    for i in range(n_perm):
        p = rng.permutation(n)

        rperm = RB[p][:, p][iu]
        rp0 = rperm - rperm.mean()

        den = ra_den * np.sqrt(np.sum(rp0 ** 2))

        if den == 0:
            null[i] = np.nan
        else:
            null[i] = np.sum(ra0 * rp0) / den

    null = null[np.isfinite(null)]

    p_one = (
        1 + np.sum(null >= obs)
    ) / (
        1 + len(null)
    )

    p_two = (
        1 + np.sum(np.abs(null) >= abs(obs))
    ) / (
        1 + len(null)
    )

    return {
        "spearman_r": obs,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "null_mean": np.mean(null),
        "null_sd": np.std(null),
        "null_p95": np.quantile(null, 0.95),
        "null": null,
    }


# ============================================================
# LOAD AND ALIGN
# ============================================================

print("=" * 78)
print("TASK 2: OPENSCREEN MORPHOLOGY ↔ TAHOE GENE EXPRESSION")
print("=" * 78)

M = pd.read_parquet(MORPH)
G = pd.read_parquet(GEX)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

print("\nMorphology:", M.shape)
print("Gene expression:", G.shape)

shared = sorted(
    set(M.index) & set(G.index)
)

print("Shared drugs:", len(shared))

m_only = sorted(set(M.index) - set(G.index))
g_only = sorted(set(G.index) - set(M.index))

print("Morphology-only:", m_only)
print("GEX-only:", g_only)

M = M.loc[shared]
G = G.loc[shared]

assert M.shape[0] == 119
assert G.shape[0] == 119
assert M.index.equals(G.index)

print("\nAligned matrices:")
print("M:", M.shape)
print("G:", G.shape)


# ============================================================
# STANDARDISE EACH FEATURE ACROSS DRUGS
# ============================================================

# Important:
# We want relative drug perturbation patterns, not the common
# HepG2 baseline or different feature scales.

Mz = StandardScaler().fit_transform(
    M.to_numpy(dtype=np.float64)
)

Gz = StandardScaler().fit_transform(
    G.to_numpy(dtype=np.float64)
)

print("\nFeature standardisation complete.")


# ============================================================
# PRIMARY ANALYSIS: FULL STANDARDISED FEATURE SPACES
# ============================================================

Dm = row_corr_distance(Mz)
Dg = row_corr_distance(Gz)

vm, _ = upper(Dm)
vg, _ = upper(Dg)

pearson_distance_r = pearsonr(vm, vg).statistic

result = mantel_spearman(
    Dm,
    Dg,
    n_perm=N_PERM,
    seed=RANDOM_STATE,
)

print("\n" + "=" * 78)
print("PRIMARY CROSS-MODAL RESULT")
print("=" * 78)

print(
    "Drug pairs:",
    len(vm)
)

print(
    "Pearson correlation between distance matrices:",
    pearson_distance_r
)

print(
    "Mantel-style Spearman r:",
    result["spearman_r"]
)

print(
    "Permutation p, one-sided:",
    result["p_one_sided"]
)

print(
    "Permutation p, two-sided:",
    result["p_two_sided"]
)

print(
    "Null mean:",
    result["null_mean"]
)

print(
    "Null SD:",
    result["null_sd"]
)

print(
    "Null 95th percentile:",
    result["null_p95"]
)


# ============================================================
# 30-PC SENSITIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 78)
print("30-PC SENSITIVITY ANALYSIS")
print("=" * 78)

Mp = PCA(
    n_components=30,
    random_state=RANDOM_STATE
).fit_transform(Mz)

Gp = PCA(
    n_components=30,
    random_state=RANDOM_STATE
).fit_transform(Gz)

Dm_pc = row_corr_distance(Mp)
Dg_pc = row_corr_distance(Gp)

res_pc = mantel_spearman(
    Dm_pc,
    Dg_pc,
    n_perm=N_PERM,
    seed=RANDOM_STATE,
)

vmp, _ = upper(Dm_pc)
vgp, _ = upper(Dg_pc)

print(
    "Morphology variance explained by 30 PCs:",
    PCA(
        n_components=30,
        random_state=RANDOM_STATE
    ).fit(Mz).explained_variance_ratio_.sum()
)

print(
    "GEX variance explained by 30 PCs:",
    PCA(
        n_components=30,
        random_state=RANDOM_STATE
    ).fit(Gz).explained_variance_ratio_.sum()
)

print(
    "Pearson distance correlation:",
    pearsonr(vmp, vgp).statistic
)

print(
    "Mantel-style Spearman r:",
    res_pc["spearman_r"]
)

print(
    "Permutation p, one-sided:",
    res_pc["p_one_sided"]
)

print(
    "Permutation p, two-sided:",
    res_pc["p_two_sided"]
)


# ============================================================
# PER-DRUG CROSS-MODAL CONCORDANCE
# ============================================================

rows = []

for i, drug in enumerate(shared):

    keep = np.arange(len(shared)) != i

    morph_dist = Dm[i, keep]
    gex_dist = Dg[i, keep]

    r = pd.Series(
        morph_dist
    ).corr(
        pd.Series(gex_dist),
        method="spearman"
    )

    rows.append({
        "drug": drug,
        "cross_modal_neighbourhood_spearman": r
    })

per_drug = (
    pd.DataFrame(rows)
    .sort_values(
        "cross_modal_neighbourhood_spearman",
        ascending=False
    )
)

per_drug.to_csv(
    OUT / "hepg2_per_drug_crossmodal_concordance.csv",
    index=False
)

print("\n" + "=" * 78)
print("PER-DRUG CROSS-MODAL CONCORDANCE")
print("=" * 78)

print("\nSummary:")
print(
    per_drug[
        "cross_modal_neighbourhood_spearman"
    ].describe()
)

print("\nHighest 15:")
print(
    per_drug.head(15).to_string(index=False)
)

print("\nLowest 15:")
print(
    per_drug.tail(15)
    .sort_values(
        "cross_modal_neighbourhood_spearman"
    )
    .to_string(index=False)
)


# ============================================================
# SAVE MATRICES
# ============================================================

pd.DataFrame(
    Dm,
    index=shared,
    columns=shared
).to_parquet(
    OUT / "hepg2_morphology_drug_distance.parquet"
)

pd.DataFrame(
    Dg,
    index=shared,
    columns=shared
).to_parquet(
    OUT / "hepg2_gex_drug_distance.parquet"
)

pd.DataFrame({
    "null_spearman_r": result["null"]
}).to_csv(
    OUT / "hepg2_crossmodal_mantel_null.csv",
    index=False
)

summary = pd.DataFrame([
    {
        "analysis": "full_636_vs_2000",
        "n_drugs": len(shared),
        "n_drug_pairs": len(vm),
        "pearson_distance_r":
            pearson_distance_r,
        "mantel_spearman_r":
            result["spearman_r"],
        "p_one_sided":
            result["p_one_sided"],
        "p_two_sided":
            result["p_two_sided"],
    },
    {
        "analysis": "30PC_vs_30PC",
        "n_drugs": len(shared),
        "n_drug_pairs": len(vmp),
        "pearson_distance_r":
            pearsonr(vmp, vgp).statistic,
        "mantel_spearman_r":
            res_pc["spearman_r"],
        "p_one_sided":
            res_pc["p_one_sided"],
        "p_two_sided":
            res_pc["p_two_sided"],
    }
])

summary.to_csv(
    OUT / "hepg2_crossmodal_summary.csv",
    index=False
)

print("\n" + "=" * 78)
print("SAVED OUTPUTS")
print("=" * 78)

for f in [
    "hepg2_crossmodal_summary.csv",
    "hepg2_morphology_drug_distance.parquet",
    "hepg2_gex_drug_distance.parquet",
    "hepg2_crossmodal_mantel_null.csv",
    "hepg2_per_drug_crossmodal_concordance.csv",
]:
    print(OUT / f)

