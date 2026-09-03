#!/usr/bin/env python3
"""
What is the LARGEST cross-modal direction agreement this data could show?

The problem
-----------
Observed direction-only Mantel r = 0.040. That is small, but a correlation
between two noisy measurements is attenuated by roughly

    sqrt(reliability_M * reliability_G)

Morphology replicates agree at r ~ 0.5 and same-drug expression samples at
r ~ 0.355, so a TRUE directional correspondence of 0.10 would be observed as
about 0.04. "Small" and "absent" are different claims, and only the ratio to
the ceiling distinguishes them.

Method
------
Split each modality's replicates into two independent halves, build direction-
only distance matrices from each half, and Mantel them against each other
WITHIN modality. That is the reliability of the direction structure itself --
the same quantity the cross-modal statistic is attenuated by, measured in the
same units, rather than borrowed from a profile-level replicate correlation.

    ceiling = sqrt(rel_M * rel_G)
    disattenuated = observed / ceiling

Report the observed value, the ceiling, and the ratio. A ratio near 1 means the
modalities agree as well as the data permits. A ratio near 0 with a healthy
ceiling is real evidence of absence.

Inputs: four parquet files of drugs x features, same drug index --
morphology half A and B, expression half A and B. For morphology, the natural
split is by acquisition site (e.g. FMP vs IMTM). For expression, split the
sample-level pseudobulks per drug into two groups.

Usage:
    python ws4_noise_ceiling.py M_a.parquet M_b.parquet G_a.parquet G_b.parquet
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.decomposition import PCA

N_PCS = 30
N_PERM = 999
RANDOM_STATE = 0


def zscore(df):
    return (df - df.mean()) / df.std(ddof=0).replace(0, 1)


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def direction_matrix(X, n_pcs=N_PCS, seed=RANDOM_STATE):
    """PCA -> z-score PCs -> centre -> unit-normalise -> Euclidean distances."""
    X = np.asarray(X, dtype=np.float64)
    k = min(n_pcs, X.shape[0] - 1, X.shape[1])
    P = PCA(n_components=k, random_state=seed).fit_transform(X)
    Z = zscore(pd.DataFrame(P)).to_numpy()
    U = Z - Z.mean(0)
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    return squareform(pdist(U))


def mantel(D1, D2, n_perm=N_PERM, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    r1 = rankdata(D1[iu])
    obs = _pearson(r1, rankdata(D2[iu]))
    null = np.array([_pearson(r1, rankdata(D2[np.ix_(p, p)][iu]))
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    return obs, float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))


def split_by_group(long_df, drug_col, group_col, feature_cols, seed=RANDOM_STATE):
    """Split replicate rows per drug into two halves and average each."""
    rng = np.random.default_rng(seed)
    a_rows, b_rows = [], []
    for drug, g in long_df.groupby(drug_col, observed=True):
        groups = list(g[group_col].unique())
        if len(groups) < 2:
            continue
        rng.shuffle(groups)
        half = max(1, len(groups) // 2)
        ga, gb = set(groups[:half]), set(groups[half:])
        a_rows.append(g[g[group_col].isin(ga)][feature_cols].mean().rename(drug))
        b_rows.append(g[g[group_col].isin(gb)][feature_cols].mean().rename(drug))
    return pd.DataFrame(a_rows), pd.DataFrame(b_rows)


def main(ma, mb, ga, gb):
    Ma, Mb, Ga, Gb = (pd.read_parquet(p) for p in (ma, mb, ga, gb))
    for d in (Ma, Mb, Ga, Gb):
        d.index = d.index.astype(str)
    shared = sorted(set(Ma.index) & set(Mb.index) & set(Ga.index) & set(Gb.index))
    Ma, Mb, Ga, Gb = (d.loc[shared] for d in (Ma, Mb, Ga, Gb))
    print(f"{len(shared)} drugs present in all four halves\n")

    DMa, DMb = direction_matrix(Ma), direction_matrix(Mb)
    DGa, DGb = direction_matrix(Ga), direction_matrix(Gb)

    rel_m, p_m = mantel(DMa, DMb)
    rel_g, p_g = mantel(DGa, DGb)
    print(f"morphology direction reliability  r = {rel_m:+.4f}  p = {p_m:.4f}")
    print(f"expression direction reliability  r = {rel_g:+.4f}  p = {p_g:.4f}")

    if rel_m <= 0 or rel_g <= 0:
        print("\n  A reliability is at or below zero: the direction structure "
              "is not reproducible within that modality at all. No cross-modal "
              "directional result can be interpreted, in either direction.")
        return

    ceiling = float(np.sqrt(rel_m * rel_g))
    print(f"\nattenuation ceiling sqrt(rel_M * rel_G) = {ceiling:.4f}")

    # observed cross-modal direction agreement, averaged over the four
    # half-to-half comparisons so it is on the same footing as the ceiling
    cross = [mantel(DMa, DGa)[0], mantel(DMa, DGb)[0],
             mantel(DMb, DGa)[0], mantel(DMb, DGb)[0]]
    obs = float(np.mean(cross))
    print(f"observed cross-modal direction (mean of 4 half-pairs) = {obs:+.4f}")

    ratio = obs / ceiling
    disatt = obs / ceiling
    print(f"\nfraction of the achievable ceiling reached = {ratio:.3f}")
    print(f"disattenuated estimate of true direction agreement = {disatt:+.4f}")

    print("\nreading:")
    if ceiling < 0.15:
        print(f"  The ceiling itself is only {ceiling:.3f}. Neither modality "
              f"resolves compound-specific DIRECTION reproducibly enough to "
              f"detect a modest correspondence. Report this as insufficient "
              f"power, NOT as evidence that no correspondence exists.")
    elif ratio < 0.25:
        print(f"  Observed agreement reaches only {ratio:.0%} of what the data "
              f"permits, and the ceiling ({ceiling:.3f}) is high enough that a "
              f"real correspondence would have shown. This is evidence of "
              f"absence, not just absence of evidence.")
    else:
        print(f"  Observed agreement reaches {ratio:.0%} of the ceiling. The "
              f"modalities agree on direction about as well as the measurement "
              f"noise allows -- the raw r understates the correspondence.")
    return {"rel_morph": rel_m, "rel_expr": rel_g, "ceiling": ceiling,
            "observed": obs, "ratio": ratio}


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:5])