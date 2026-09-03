#!/usr/bin/env python3

"""
LINCS A549 final same-logic rebuild + directional noise ceiling.

Expression:
- Uses already normalized/log1p expression in X.
- Does NOT normalize/log1p again.
- Uses the existing 2000 HVG annotation.
- Cells -> equal-weight sample mean expression.
- Gene-wise standardize across samples.
- Within-plate centering.
- Samples -> equal-weight drug consensus.

Morphology:
- Uses already pycytominer-processed compound-dose consensus.
- Excludes 41 numerically pathological features with max|X| > 1e6.
- Six dose profiles -> drug consensus BEFORE PCA.

Task 2:
- match drugs
- PCA 30 independently
- z-score PC axes across drugs
- Euclidean drug-pair geometry
- drug-label permutation
- strength
- direction

Noise ceiling:
- morphology: balanced 3-dose vs 3-dose halves
- expression: sample splits within drug
- independent PCA per half
- directional geometry reliability
- attenuation ceiling sqrt(r_morph * r_expr)
- 4 cross-modal half-pair correlations
"""

from pathlib import Path
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


SEED = 0
N_PCS = 30
N_PERM = 999
CHUNK = 5000
MORPH_THRESHOLD = 1e6

MORPH_FILE = "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"
EXPR_FILE = "LINCS/data/lincs_expression_a549.h5ad"

OUT = Path("LINCS/final_rebuild")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def stable_rng(label):
    h = hashlib.sha256(
        f"{SEED}|{label}".encode()
    ).digest()
    seed = int.from_bytes(h[:8], "little") % (2**32 - 1)
    return np.random.default_rng(seed)


def zscore_cols(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def unit_rows(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def distmat(X):
    return squareform(pdist(X, metric="euclidean"))


def geometry_from_features(X, n_pcs=N_PCS, direction=False):
    ncomp = min(n_pcs, X.shape[0]-1, X.shape[1])

    pc = PCA(
        n_components=ncomp,
        svd_solver="full",
        random_state=SEED
    ).fit_transform(np.asarray(X, dtype=np.float64))

    pc = zscore_cols(pc)

    if direction:
        pc = unit_rows(pc)

    return distmat(pc), pc


def corr_distance(D1, D2):
    iu = np.triu_indices(D1.shape[0], k=1)
    return float(
        spearmanr(D1[iu], D2[iu]).statistic
    )


def permutation_geometry(D1, D2):
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)

    obs = float(
        spearmanr(D1[iu], D2[iu]).statistic
    )

    rng = np.random.default_rng(SEED)
    idx = np.arange(n)

    null = np.zeros(N_PERM)

    for b in range(N_PERM):
        p = rng.permutation(idx)
        Dp = D2[np.ix_(p, p)]

        null[b] = spearmanr(
            D1[iu],
            Dp[iu]
        ).statistic

    p_one = (
        1 + np.sum(null >= obs)
    ) / (
        N_PERM + 1
    )

    p_two = (
        1 + np.sum(np.abs(null) >= abs(obs))
    ) / (
        N_PERM + 1
    )

    return obs, p_one, p_two, null


def strength_perm(x, y):
    obs = float(spearmanr(x, y).statistic)

    rng = np.random.default_rng(SEED)
    null = np.zeros(N_PERM)

    for b in range(N_PERM):
        p = rng.permutation(len(y))
        null[b] = spearmanr(x, y[p]).statistic

    p = (
        1 + np.sum(null >= obs)
    ) / (
        N_PERM + 1
    )

    return obs, p


def eta2(x, groups):
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)

    grand = x.mean()
    total = np.sum((x - grand) ** 2)

    between = 0.0

    for g in np.unique(groups):
        y = x[groups == g]
        between += len(y) * (y.mean() - grand) ** 2

    return between / total if total > 0 else 0.0


# ============================================================
# 1. MORPHOLOGY
# ============================================================

print("=" * 100)
print("A549 SAME-LOGIC REBUILD + NOISE CEILING")
print("=" * 100)

m = ad.read_h5ad(MORPH_FILE)

Xm = np.asarray(m.X, dtype=np.float64)

feature_max = np.nanmax(np.abs(Xm), axis=0)
finite = np.all(np.isfinite(Xm), axis=0)

keep = finite & (feature_max <= MORPH_THRESHOLD)

print("\nMORPHOLOGY")
print("dose profiles:", m.n_obs)
print("original features:", m.n_vars)
print("excluded pathological:", (~keep).sum())
print("retained:", keep.sum())

assert keep.sum() == 574

Xm = Xm[:, keep]

mobs = m.obs.reset_index(drop=True).copy()
mobs["drug"] = mobs["Metadata_Drug"].astype(str)
mobs["dose"] = mobs["Metadata_dose_recode"].astype(int)


# Full drug consensus BEFORE PCA
morph_drugs = sorted(mobs["drug"].unique())

Mdrug = []

for drug in morph_drugs:
    idx = np.where(mobs["drug"].to_numpy() == drug)[0]
    Mdrug.append(Xm[idx].mean(axis=0))

Mdrug = pd.DataFrame(
    np.vstack(Mdrug),
    index=morph_drugs
)

print("drug-level morphology:", Mdrug.shape)


# ============================================================
# 2. EXPRESSION: CELLS -> SAMPLE MEANS
# ============================================================

g = ad.read_h5ad(EXPR_FILE, backed="r")

obs = g.obs.copy()

hvg_mask = g.var["highly_variable"].astype(bool).to_numpy()
hvg_idx = np.where(hvg_mask)[0]

print("\nEXPRESSION")
print("cells:", g.n_obs)
print("genes:", g.n_vars)
print("HVGs:", len(hvg_idx))
print("samples:", obs["sample"].nunique())
print("drugs:", obs["Metadata_Drug"].nunique())
print("plates:", obs["plate"].nunique())

assert len(hvg_idx) == 2000


sample_labels = obs["sample"].astype(str).to_numpy()

sample_codes, sample_names = pd.factorize(
    sample_labels,
    sort=True
)

n_samples = len(sample_names)
n_hvg = len(hvg_idx)

sample_sums = np.zeros(
    (n_samples, n_hvg),
    dtype=np.float64
)

sample_counts = np.zeros(
    n_samples,
    dtype=np.int64
)


print("\nAggregating cells -> samples...")

for start in range(0, g.n_obs, CHUNK):

    end = min(start + CHUNK, g.n_obs)

    # Row slice FIRST.
    # This avoids two-axis fancy indexing on the backed object.
    chunk = g.X[start:end]

    # Subset HVGs after row slice.
    if sparse.issparse(chunk):
        Xh = chunk[:, hvg_idx]
    else:
        Xh = np.asarray(chunk)[:, hvg_idx]

    codes = sample_codes[start:end]

    for c in np.unique(codes):
        rows = np.where(codes == c)[0]

        if sparse.issparse(Xh):
            v = np.asarray(
                Xh[rows].sum(axis=0)
            ).ravel()
        else:
            v = Xh[rows].sum(axis=0)

        sample_sums[c] += v
        sample_counts[c] += len(rows)

    if start % 50000 == 0:
        print(f"  {start:,}/{g.n_obs:,}")


sample_X = sample_sums / sample_counts[:, None]


# Sample metadata
meta = (
    obs.assign(
        sample=obs["sample"].astype(str)
    )
    .groupby("sample", observed=True)
    .agg(
        drug=("Metadata_Drug", "first"),
        plate=("plate", "first"),
        n_cells=("Metadata_Drug", "size")
    )
    .reindex(sample_names)
)

meta["drug"] = meta["drug"].astype(str)
meta["plate"] = meta["plate"].astype(str)

print("sample matrix:", sample_X.shape)
print("median cells/sample:", int(np.median(sample_counts)))

# ============================================================
# SAMPLE QC — SAME RULE AS HEPG2
# ============================================================

good = sample_counts >= 100

print("\nSAMPLE QC")
print("samples before:", len(sample_counts))
print("samples <100 cells:", (~good).sum())

if (~good).sum() > 0:
    print(
        meta.loc[~good, ["drug", "plate", "n_cells"]]
        .to_string()
    )

sample_X = sample_X[good]
meta = meta.iloc[np.where(good)[0]].copy()
sample_counts = sample_counts[good]
sample_names = sample_names[good]

print("samples after >=100-cell QC:", len(sample_counts))
print("drugs retained:", meta["drug"].nunique())
print("minimum cells/sample:", sample_counts.min())

# ============================================================
# 3. PLATE QC + CORRECTION
# ============================================================

sample_Z = zscore_cols(sample_X)

pca0 = PCA(
    n_components=10,
    random_state=SEED
).fit_transform(sample_Z)

plates = meta["plate"].to_numpy()

before_eta = [
    eta2(pca0[:, i], plates)
    for i in range(5)
]

print("\nPLATE eta2 BEFORE")
for i, e in enumerate(before_eta, 1):
    print(f"PC{i}: {e:.6f}")


sample_corr = sample_Z.copy()

for plate in np.unique(plates):
    mask = plates == plate

    sample_corr[mask] -= sample_corr[mask].mean(
        axis=0,
        keepdims=True
    )


pca1 = PCA(
    n_components=10,
    random_state=SEED
).fit_transform(sample_corr)

after_eta = [
    eta2(pca1[:, i], plates)
    for i in range(5)
]

print("\nPLATE eta2 AFTER")
for i, e in enumerate(after_eta, 1):
    print(f"PC{i}: {e:.8f}")


# ============================================================
# 4. EXPRESSION SAMPLE -> DRUG
# ============================================================

expr_drugs = sorted(meta["drug"].unique())

Gdrug = []

for drug in expr_drugs:
    mask = meta["drug"].to_numpy() == drug

    # equal-weight sample consensus
    Gdrug.append(
        sample_corr[mask].mean(axis=0)
    )

Gdrug = pd.DataFrame(
    np.vstack(Gdrug),
    index=expr_drugs
)

print("\ndrug-level expression:", Gdrug.shape)


# ============================================================
# 5. SAME-LOGIC FULL TASK 2
# ============================================================

shared = sorted(
    set(Mdrug.index) &
    set(Gdrug.index)
)

print("\nFULL TASK 2")
print("shared drugs:", len(shared))

M = Mdrug.loc[shared].to_numpy()
G = Gdrug.loc[shared].to_numpy()

n = len(shared)
n_pairs = n * (n - 1) // 2

print("drug pairs:", n_pairs)


# independently PCA each modality
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

Dm = distmat(Mz)
Dg = distmat(Gz)

r_all, p_all, _, _ = permutation_geometry(Dm, Dg)


# strength
sm = np.linalg.norm(Mz, axis=1)
sg = np.linalg.norm(Gz, axis=1)

rho_strength, p_strength = strength_perm(sm, sg)


# direction
Mdir = unit_rows(Mz)
Gdir = unit_rows(Gz)

Dm_dir = distmat(Mdir)
Dg_dir = distmat(Gdir)

r_dir, p_dir, _, _ = permutation_geometry(
    Dm_dir,
    Dg_dir
)


# PC1 drop
Mno1 = unit_rows(Mz[:, 1:])
Gno1 = unit_rows(Gz[:, 1:])

r_no1, p_no1, _, _ = permutation_geometry(
    distmat(Mno1),
    distmat(Gno1)
)


print("\nREBUILT PRIMARY RESULT")
print(f"overall   r={r_all:.6f}, p={p_all:.4f}")
print(f"strength  rho={rho_strength:.6f}, p={p_strength:.4f}")
print(f"direction r={r_dir:.6f}, p={p_dir:.4f}")
print(f"PC1-drop  r={r_no1:.6f}, p={p_no1:.4f}")


# ============================================================
# 6. MORPHOLOGY HALF MATRICES
#
# One dose from each adjacent pair goes into each half:
# (1,2), (3,4), (5,6)
#
# This ensures BOTH halves span low / mid / high dose.
# ============================================================

morph_half_A = {}
morph_half_B = {}

for drug in morph_drugs:

    sub_idx = np.where(
        mobs["drug"].to_numpy() == drug
    )[0]

    doses_here = mobs.loc[sub_idx, "dose"].to_numpy()

    dose_to_vec = {}

    for dose in np.unique(doses_here):
        ii = sub_idx[doses_here == dose]
        dose_to_vec[int(dose)] = Xm[ii].mean(axis=0)

    if not all(d in dose_to_vec for d in range(1, 7)):
        continue

    A = []
    B = []

    rng = stable_rng("morph|" + drug)

    for d1, d2 in [(1, 2), (3, 4), (5, 6)]:

        if rng.random() < 0.5:
            A.append(dose_to_vec[d1])
            B.append(dose_to_vec[d2])
        else:
            A.append(dose_to_vec[d2])
            B.append(dose_to_vec[d1])

    morph_half_A[drug] = np.mean(A, axis=0)
    morph_half_B[drug] = np.mean(B, axis=0)


# ============================================================
# 7. EXPRESSION HALF MATRICES
# ============================================================

expr_half_A = {}
expr_half_B = {}

drug_arr = meta["drug"].to_numpy()

for drug in expr_drugs:

    idx = np.where(drug_arr == drug)[0]

    if len(idx) < 2:
        continue

    rng = stable_rng("expr|" + drug)

    idx = rng.permutation(idx)

    Aidx = idx[::2]
    Bidx = idx[1::2]

    if len(Aidx) == 0 or len(Bidx) == 0:
        continue

    expr_half_A[drug] = sample_corr[Aidx].mean(axis=0)
    expr_half_B[drug] = sample_corr[Bidx].mean(axis=0)


eligible = sorted(
    set(morph_half_A) &
    set(morph_half_B) &
    set(expr_half_A) &
    set(expr_half_B)
)

print("\nNOISE CEILING")
print("eligible split-half drugs:", len(eligible))

print(
    "expression sample counts among eligible:",
    meta[
        meta["drug"].isin(eligible)
    ].groupby("drug").size().describe().to_dict()
)


MA = np.vstack([morph_half_A[d] for d in eligible])
MB = np.vstack([morph_half_B[d] for d in eligible])

GA = np.vstack([expr_half_A[d] for d in eligible])
GB = np.vstack([expr_half_B[d] for d in eligible])


# ============================================================
# 8. DIRECTIONAL GEOMETRY FOR EACH HALF
# ============================================================

DmA, _ = geometry_from_features(
    MA,
    direction=True
)

DmB, _ = geometry_from_features(
    MB,
    direction=True
)

DgA, _ = geometry_from_features(
    GA,
    direction=True
)

DgB, _ = geometry_from_features(
    GB,
    direction=True
)


# Within-modality directional reliability
r_morph, p_morph, _, _ = permutation_geometry(
    DmA,
    DmB
)

r_expr, p_expr, _, _ = permutation_geometry(
    DgA,
    DgB
)


ceiling = np.sqrt(
    max(r_morph, 0) *
    max(r_expr, 0)
)


# Four cross-modal half pairings
cross = {
    "M_A__G_A": corr_distance(DmA, DgA),
    "M_A__G_B": corr_distance(DmA, DgB),
    "M_B__G_A": corr_distance(DmB, DgA),
    "M_B__G_B": corr_distance(DmB, DgB),
}

cross_mean = float(
    np.mean(list(cross.values()))
)

fraction_ceiling = (
    cross_mean / ceiling
    if ceiling > 0
    else np.nan
)


# Composite cross-modal permutation
rng = np.random.default_rng(SEED)
idx = np.arange(len(eligible))

null_cross = np.zeros(N_PERM)

for b in range(N_PERM):

    p = rng.permutation(idx)

    DgAp = DgA[np.ix_(p, p)]
    DgBp = DgB[np.ix_(p, p)]

    null_cross[b] = np.mean([
        corr_distance(DmA, DgAp),
        corr_distance(DmA, DgBp),
        corr_distance(DmB, DgAp),
        corr_distance(DmB, DgBp),
    ])


p_cross = (
    1 + np.sum(null_cross >= cross_mean)
) / (
    N_PERM + 1
)


print("\nDIRECTIONAL RELIABILITY")
print(
    f"Morphology reliability : "
    f"r={r_morph:.6f}, p={p_morph:.4f}"
)

print(
    f"Expression reliability : "
    f"r={r_expr:.6f}, p={p_expr:.4f}"
)

print(
    f"Attenuation ceiling     : "
    f"{ceiling:.6f}"
)

print("\nCROSS-MODAL HALF PAIRS")

for k, v in cross.items():
    print(f"{k}: {v:.6f}")

print(
    f"\nMean cross-modal direction : "
    f"{cross_mean:.6f}"
)

print(
    f"Permutation p              : "
    f"{p_cross:.4f}"
)

print(
    f"Fraction of ceiling        : "
    f"{fraction_ceiling:.6f}"
)

print(
    f"Percent of ceiling         : "
    f"{100*fraction_ceiling:.2f}%"
)


# ============================================================
# 9. SAVE EVERYTHING
# ============================================================

pd.DataFrame(
    sample_corr,
    index=sample_names,
    columns=g.var_names[hvg_idx].astype(str)
).to_parquet(
    OUT /
    "A549_platecorrected_sample_2000HVG.parquet"
)


Gdrug.loc[shared].to_parquet(
    OUT /
    "A549_platecorrected_drug_2000HVG.parquet"
)


pd.DataFrame(
    MA,
    index=eligible
).to_parquet(
    OUT /
    "A549_morph_halfA_574.parquet"
)

pd.DataFrame(
    MB,
    index=eligible
).to_parquet(
    OUT /
    "A549_morph_halfB_574.parquet"
)

pd.DataFrame(
    GA,
    index=eligible
).to_parquet(
    OUT /
    "A549_expr_halfA_2000HVG.parquet"
)

pd.DataFrame(
    GB,
    index=eligible
).to_parquet(
    OUT /
    "A549_expr_halfB_2000HVG.parquet"
)


pd.DataFrame({
    "PC": np.arange(1, 6),
    "plate_eta2_before": before_eta,
    "plate_eta2_after": after_eta,
}).to_csv(
    OUT /
    "A549_plate_QC.csv",
    index=False
)


summary = pd.DataFrame([{
    "n_full_drugs": n,
    "n_full_pairs": n_pairs,

    "overall_r": r_all,
    "overall_perm_p": p_all,

    "strength_rho": rho_strength,
    "strength_perm_p": p_strength,

    "direction_r": r_dir,
    "direction_perm_p": p_dir,

    "direction_PC1drop_r": r_no1,
    "direction_PC1drop_p": p_no1,

    "noise_n_drugs": len(eligible),

    "morph_direction_reliability": r_morph,
    "morph_reliability_p": p_morph,

    "expr_direction_reliability": r_expr,
    "expr_reliability_p": p_expr,

    "attenuation_ceiling": ceiling,

    "crossmodal_halfpair_mean": cross_mean,
    "crossmodal_halfpair_perm_p": p_cross,

    "fraction_of_ceiling": fraction_ceiling,
}])


summary.to_csv(
    OUT /
    "A549_FINAL_rebuild_noise_ceiling_summary.csv",
    index=False
)


print("\nSaved to:", OUT)
print("DONE")

g.file.close()
