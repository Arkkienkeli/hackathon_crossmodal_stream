#!/usr/bin/env python3
"""
Is the r=0.328 cross-modal result mechanism correspondence, or potency agreement?

The observation to explain
--------------------------
    30 PCs, z-scored, Euclidean      r = 0.328   p = 0.001
    30 PCs, correlation distance     r = 0.032   p = 0.018
    full features, correlation dist  r = 0.018   p = 0.063

Euclidean distance on z-scored PCs keeps each drug's distance from the origin --
the MAGNITUDE of its perturbation. Correlation distance removes magnitude and
compares only DIRECTION. So the table already says the signal is nearly all in
magnitude and nearly none in direction.

If that is right, "morphology and expression agree" reduces to "both modalities
distinguish potent compounds from inert ones", which is true but is not evidence
that a transcriptional program corresponds to a morphological phenotype. The
notebook's own PC1 interpretation (stressed vs unstressed poles) points the same
way.

This script separates the two contributions:
  1. reproduce the notebook number
  2. agreement of the two activity axes on their own
  3. how much of each modality's distance matrix is pure magnitude
  4. partial Mantel controlling for activity
  5. direction-only Mantel (unit-normalised PC vectors)
  6. the same with PC1 dropped from both blocks

Usage:
    python ws4_task2_decompose.py \\
        OpenScreen/data/hepg2_morphology_final.parquet \\
        OpenScreen/data/hepg2_pseudobulk_2000hvg_shared119.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA

N_PCS = 30
N_PERM = 999
RANDOM_STATE = 0


def zscore(df):
    std = df.std(ddof=0).replace(0, 1)
    return (df - df.mean()) / std


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mantel(D1, D2, n_perm=N_PERM, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    r1 = rankdata(D1[iu])
    obs = _pearson(r1, rankdata(D2[iu]))
    null = np.array([_pearson(r1, rankdata(D2[np.ix_(p, p)][iu]))
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    return obs, float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))


def partial_mantel(D1, D2, nuisance, n_perm=N_PERM, seed=RANDOM_STATE):
    """Mantel between D1 and D2 controlling for several nuisance matrices.

    Residualises both rank vectors on the nuisance ranks by least squares, then
    correlates the residuals. Multiple covariates are needed because Euclidean
    distance under random directions scales as sqrt(a_i^2 + a_j^2), not as
    |a_i - a_j| -- a difference-only nuisance cannot absorb it.
    """
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    N = np.column_stack([rankdata(D[iu]) for D in nuisance])
    N = np.column_stack([np.ones(len(N)), N])

    def resid(v):
        beta, *_ = np.linalg.lstsq(N, v, rcond=None)
        return v - N @ beta

    r1 = resid(rankdata(D1[iu]))
    obs = _pearson(r1, resid(rankdata(D2[iu])))
    null = np.array([_pearson(r1, resid(rankdata(D2[np.ix_(p, p)][iu])))
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    return obs, float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))


def magnitude_nuisance(a):
    """Distance matrices implied by magnitudes alone, both functional forms."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    quad = np.sqrt(np.add.outer(a**2, a**2))
    diff = np.abs(np.subtract.outer(a, a))
    return [quad, diff]


def embed(X, n_pcs=N_PCS, drop_pc1=False, seed=RANDOM_STATE):
    """Notebook representation: PCA on drug profiles, then z-score each PC."""
    P = PCA(n_components=n_pcs, random_state=seed).fit_transform(
        np.asarray(X, dtype=np.float64))
    Z = zscore(pd.DataFrame(P)).to_numpy()
    return Z[:, 1:] if drop_pc1 else Z


def report(Mz, Gz, label):
    print(f"\n--- {label} ---")
    Dm = squareform(pdist(Mz))
    Dg = squareform(pdist(Gz))

    r, p = mantel(Dm, Dg)
    print(f"  Euclidean (notebook metric)     r = {r:+.4f}  p = {p:.4f}")

    # activity = distance from the centroid in each modality
    am = np.linalg.norm(Mz - Mz.mean(0), axis=1)
    ag = np.linalg.norm(Gz - Gz.mean(0), axis=1)
    rho = spearmanr(am, ag).statistic
    print(f"  activity axes agree             rho = {rho:+.4f}")

    nuis = magnitude_nuisance(am) + magnitude_nuisance(ag)
    Da = nuis[0]
    rm, _ = mantel(Dm, magnitude_nuisance(am)[0])
    rg, _ = mantel(Dg, magnitude_nuisance(ag)[0])
    print(f"  morphology distance vs activity r = {rm:+.4f}")
    print(f"  expression distance vs activity r = {rg:+.4f}")

    rp, pp = partial_mantel(Dm, Dg, nuis)
    print(f"  Euclidean controlling activity  r = {rp:+.4f}  p = {pp:.4f}")

    # direction only: project every drug onto the unit sphere
    Mu = (Mz - Mz.mean(0)); Mu /= np.linalg.norm(Mu, axis=1, keepdims=True) + 1e-12
    Gu = (Gz - Gz.mean(0)); Gu /= np.linalg.norm(Gu, axis=1, keepdims=True) + 1e-12
    rd, pd_ = mantel(squareform(pdist(Mu)), squareform(pdist(Gu)))
    print(f"  direction only (unit-normalised) r = {rd:+.4f}  p = {pd_:.4f}")

    return {"variant": label, "euclidean_r": r, "euclidean_p": p,
            "activity_agreement": float(rho),
            "morph_vs_activity": rm, "expr_vs_activity": rg,
            "partial_r": rp, "partial_p": pp,
            "direction_r": rd, "direction_p": pd_}


def main(morph_path, gex_path, out="task2_decomposition.csv"):
    M = pd.read_parquet(morph_path)
    G = pd.read_parquet(gex_path)
    M.index = M.index.astype(str)
    G.index = G.index.astype(str)
    shared = sorted(set(M.index) & set(G.index))
    M, G = M.loc[shared], G.loc[shared]
    print(f"{len(shared)} drugs | morph {M.shape[1]}d | expr {G.shape[1]}d")

    rows = [report(embed(M), embed(G), "30 PCs, z-scored"),
            report(embed(M, drop_pc1=True), embed(G, drop_pc1=True),
                   "PCs 2-30 (PC1 dropped)")]

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))

    base = rows[0]
    print("\nreading (use direction_r, NOT partial_r):")
    print("  NOTE: on simulated data where ONLY potency is shared, the partial")
    print("  Mantel still returns ~0.86 -- rank residualisation cannot absorb")
    print("  the sqrt(a_i^2 + a_j^2) form of Euclidean distance. The")
    print("  direction-only statistic separates the cases cleanly (0.03 vs")
    print("  0.68 on the same simulations), so read that one.\n")
    d_r, d_p = base["direction_r"], base["direction_p"]
    if d_p >= 0.05 or abs(d_r) < 0.08:
        print(f"  direction-only r = {d_r:+.4f} (p = {d_p:.4f}). The headline "
              f"r = {base['euclidean_r']:.3f} is carried by perturbation "
              f"MAGNITUDE. The two modalities agree on how strongly each "
              f"compound acted, not on which mechanism it acted through. "
              f"Report it as potency agreement.")
    elif abs(d_r) < 0.4 * abs(base["euclidean_r"]):
        print(f"  direction-only r = {d_r:+.4f} against Euclidean "
              f"{base['euclidean_r']:.3f}. There is real directional "
              f"correspondence, but most of the headline number is magnitude. "
              f"Report both.")
    else:
        print(f"  direction-only r = {d_r:+.4f} holds up. Genuine "
              f"correspondence beyond perturbation strength.")
    print(f"\nsaved {out}")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])