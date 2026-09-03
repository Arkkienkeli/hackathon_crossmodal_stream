#!/usr/bin/env python3

import mudata as md
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

FILE = "LINCS/raw_recovery/a549_mdata.h5mu"
N_PCS = 30
N_PERM = 999
SEED = 0


def zscore_cols(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    return (X - mu) / sd


def unit_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


def dist(X, metric="euclidean"):
    return squareform(pdist(X, metric=metric))


def perm_test(D1, D2):
    n = len(D1)
    iu = np.triu_indices(n, 1)

    obs = spearmanr(D1[iu], D2[iu]).statistic

    rng = np.random.default_rng(SEED)
    null = []

    for _ in range(N_PERM):
        p = rng.permutation(n)
        Dp = D2[np.ix_(p, p)]
        null.append(
            spearmanr(D1[iu], Dp[iu]).statistic
        )

    null = np.asarray(null)

    pval = (
        1 + np.sum(null >= obs)
    ) / (
        N_PERM + 1
    )

    return float(obs), float(pval)


print("=" * 90)
print("LINCS A549 MuData — 94-DRUG SENSITIVITY")
print("=" * 90)

m = md.read_h5mu(FILE)

M = np.asarray(m.mod["morphology"].X, dtype=float)
G = np.asarray(m.mod["expression"].X, dtype=float)

drugs = m.obs["drug"].astype(str).to_numpy()

print("drugs:", len(drugs))
print("morphology:", M.shape)
print("expression:", G.shape)

# ------------------------------------------------------------
# Morphology numerical QC
# ------------------------------------------------------------

max_abs = np.nanmax(np.abs(M), axis=0)

keep_m = (
    np.all(np.isfinite(M), axis=0)
    &
    (max_abs <= 1e6)
)

print("\nMorphology")
print("original features:", M.shape[1])
print("excluded pathological:", (~keep_m).sum())
print("retained:", keep_m.sum())

M = M[:, keep_m]


# ------------------------------------------------------------
# Expression QC
# ------------------------------------------------------------

keep_g = (
    np.all(np.isfinite(G), axis=0)
    &
    (np.std(G, axis=0) > 0)
)

G = G[:, keep_g]

print("\nExpression")
print("finite/nonconstant genes:", G.shape[1])


# ------------------------------------------------------------
# PCA separately in each modality
# ------------------------------------------------------------

Mpca = PCA(
    n_components=N_PCS,
    svd_solver="full",
    random_state=SEED
).fit_transform(M)

Gpca = PCA(
    n_components=N_PCS,
    svd_solver="full",
    random_state=SEED
).fit_transform(G)

Mz = zscore_cols(Mpca)
Gz = zscore_cols(Gpca)


# ------------------------------------------------------------
# Overall Euclidean geometry
# ------------------------------------------------------------

Dm = dist(Mz)
Dg = dist(Gz)

overall_r, overall_p = perm_test(Dm, Dg)


# ------------------------------------------------------------
# Strength
# ------------------------------------------------------------

sm = np.linalg.norm(Mz, axis=1)
sg = np.linalg.norm(Gz, axis=1)

strength_rho = spearmanr(sm, sg).statistic

rng = np.random.default_rng(SEED)

null = np.array([
    spearmanr(sm, sg[rng.permutation(len(sg))]).statistic
    for _ in range(N_PERM)
])

strength_p = (
    1 + np.sum(null >= strength_rho)
) / (
    N_PERM + 1
)


# ------------------------------------------------------------
# Direction-only
# ------------------------------------------------------------

Mdir = unit_rows(Mz)
Gdir = unit_rows(Gz)

direction_r, direction_p = perm_test(
    dist(Mdir),
    dist(Gdir)
)


# ------------------------------------------------------------
# Explicit cosine
# ------------------------------------------------------------

cos_r, cos_p = perm_test(
    dist(Mz, "cosine"),
    dist(Gz, "cosine")
)


print("\n" + "=" * 90)
print("94-DRUG RESULTS")
print("=" * 90)

print(
    f"Overall Euclidean : "
    f"r={overall_r:.6f}, p={overall_p:.4f}"
)

print(
    f"Strength          : "
    f"rho={strength_rho:.6f}, p={strength_p:.4f}"
)

print(
    f"Direction-only    : "
    f"r={direction_r:.6f}, p={direction_p:.4f}"
)

print(
    f"Cosine            : "
    f"r={cos_r:.6f}, p={cos_p:.4f}"
)

print("\nDONE")
