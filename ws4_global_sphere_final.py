#!/usr/bin/env python3

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import anndata as ad

from pycytominer.operations import Spherize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_NULL = 10000
K = 5

PARQUET = Path(
    "OpenScreen/sphering/hepg2_mad_selected.parquet"
)

REPORT_SITES = [
    "FMP_HepG2",
    "IMTM_HepG2",
    "MEDINA_HepG2",
]

H5ADS = {
    "FMP_HepG2":
        "OpenScreen/data/morphology_FMP_HepG2.h5ad",
    "IMTM_HepG2":
        "OpenScreen/data/morphology_IMTM_HepG2.h5ad",
    "MEDINA_HepG2":
        "OpenScreen/data/morphology_MEDINA_HepG2.h5ad",
}

OUT = Path("OpenScreen/sphering")
OUT.mkdir(exist_ok=True, parents=True)


# ============================================================
# HELPERS
# ============================================================

def row_corr(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)

    den = np.sqrt(
        np.sum(A * A, axis=1) *
        np.sum(B * B, axis=1)
    )

    den[den == 0] = np.nan

    return np.sum(A * B, axis=1) / den


def dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


# ============================================================
# LOAD MAD-NORMALIZED WELL DATA
# ============================================================

print("=" * 80)
print("LOAD MAD-ROBUSTIZED OPENSCREEN")
print("=" * 80)

raw = pd.read_parquet(PARQUET)

features = [
    c for c in raw.columns
    if not c.startswith("Metadata_")
]

print("wells:", len(raw))
print("features:", len(features))
print(
    "sites:",
    raw["Metadata_Site"].value_counts().to_dict()
)


# ============================================================
# POOLED DMSO REFERENCE
# ============================================================

dmso = raw[
    raw["Metadata_negcon"].astype(bool)
][features].copy()

print("\n" + "=" * 80)
print("POOLED DMSO REFERENCE")
print("=" * 80)

print("DMSO wells:", len(dmso))
print("features:", len(features))
print("n/p:", len(dmso) / len(features))

print("\nDMSO by site:")
print(
    raw[
        raw["Metadata_negcon"].astype(bool)
    ]["Metadata_Site"].value_counts()
)


# Conditioning diagnostic
Xc = dmso.to_numpy(np.float64)

Z = Xc - Xc.mean(0)
sd = Z.std(0, ddof=1)
sd[sd == 0] = 1
Z /= sd

C = np.corrcoef(Z, rowvar=False)
C = np.nan_to_num(C)

ev = np.linalg.eigvalsh(C)
ev = np.clip(ev, 0, None)

print("\nPooled covariance conditioning:")
print("largest eigenvalue:", ev[-1])
print("smallest eigenvalue:", ev[0])
print(
    "condition number:",
    ev[-1] / max(ev[0], 1e-12)
)
print("eigenvalues < 1e-3:", (ev < 1e-3).sum())
print("eigenvalues < 1e-2:", (ev < 1e-2).sum())

# ============================================================
# LOAD EXACT TASK-2 DRUG / PDID MAPPING
# ============================================================

print("\n" + "=" * 80)
print("LOAD EXISTING TASK-2 DRUG / PDID MAPPING")
print("=" * 80)

pdid_sets = {}
pdid_to_drug = {}

for site, path in H5ADS.items():

    a = ad.read_h5ad(path)

    if "Metadata_Drug" not in a.obs.columns:
        raise RuntimeError(
            f"{site}: Metadata_Drug missing"
        )

    if "Metadata_pdid" not in a.obs.columns:
        raise RuntimeError(
            f"{site}: Metadata_pdid missing"
        )

    obs = a.obs[
        ["Metadata_pdid", "Metadata_Drug"]
    ].copy()

    obs = obs.dropna(
        subset=["Metadata_pdid", "Metadata_Drug"]
    )

    obs["Metadata_pdid"] = (
        obs["Metadata_pdid"].astype(str)
    )

    obs["Metadata_Drug"] = (
        obs["Metadata_Drug"].astype(str)
    )

    pdid_sets[site] = set(
        obs["Metadata_pdid"]
    )

    for _, r in obs.iterrows():
        pdid_to_drug[r["Metadata_pdid"]] = (
            r["Metadata_Drug"]
        )

    print(
        f"{site}: "
        f"{len(pdid_sets[site])} pdids"
    )


# Build baseline drug consensus from the RECONSTRUCTED
# 636-feature MAD-normalized profiles.
baseline_consensus = {}

for site in REPORT_SITES:

    x = raw[
        (raw["Metadata_Site"] == site)
        & (~raw["Metadata_negcon"].astype(bool))
        & (raw["Metadata_pdid"].notna())
    ].copy()

    x["Metadata_pdid"] = (
        x["Metadata_pdid"].astype(str)
    )

    x = x[
        x["Metadata_pdid"].isin(
            pdid_sets[site]
        )
    ].copy()

    x["Metadata_Drug"] = (
        x["Metadata_pdid"].map(
            pdid_to_drug
        )
    )

    x = x.dropna(
        subset=["Metadata_Drug"]
    )

    baseline_consensus[site] = (
        x.groupby(
            "Metadata_Drug",
            observed=True
        )[features]
        .mean()
        .sort_index()
    )

    print(
        f"{site}: "
        f"{len(x)} treatment wells -> "
        f"{len(baseline_consensus[site])} drugs"
    )


print("\nPairwise shared drugs:")

for a, b in combinations(
    REPORT_SITES, 2
):

    shared = (
        set(baseline_consensus[a].index)
        & set(baseline_consensus[b].index)
    )

    print(
        a, "vs", b, ":",
        len(shared)
    )


all_shared = set.intersection(
    *[
        set(baseline_consensus[s].index)
        for s in REPORT_SITES
    ]
)

print(
    "\nShared across all three sites:",
    len(all_shared)
)

# ============================================================
# TRANSFORMS
# ============================================================

def identity_fit():
    return lambda X: np.asarray(
        X,
        dtype=np.float64
    )


def global_zca_fit(epsilon):

    sph = Spherize(
        epsilon=epsilon,
        center=True,
        method="ZCA-cor",
        return_numpy=True,
    )

    # IMPORTANT:
    # fit ONCE on all 3136 DMSO controls.
    sph.fit(dmso)

    def transform(X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=features
            )

        return np.asarray(
            sph.transform(X[features]),
            dtype=np.float64
        )

    return transform


# ============================================================
# CROSS-SITE QC ON EXACT DRUG NAMES
# ============================================================

def cross_site_qc(consensus, label):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rows = []

    for s1, s2 in combinations(
        REPORT_SITES, 2
    ):

        shared = sorted(
            set(consensus[s1].index)
            & set(consensus[s2].index)
        )

        A = consensus[s1].loc[
            shared
        ].to_numpy(np.float64)

        B = consensus[s2].loc[
            shared
        ].to_numpy(np.float64)

        matched = row_corr(A, B)

        null = []

        while len(null) < N_NULL:

            perm = rng.permutation(
                len(shared)
            )

            vals = row_corr(
                A,
                B[perm]
            )

            valid = (
                (perm != np.arange(len(shared)))
                & np.isfinite(vals)
            )

            null.extend(
                vals[valid].tolist()
            )

        null = np.asarray(
            null[:N_NULL]
        )

        p95 = np.quantile(
            null,
            0.95
        )

        rows.append({
            "transform": label,
            "pair": f"{s1} vs {s2}",
            "n_drugs": len(shared),
            "median_matched_r":
                np.nanmedian(matched),
            "median_null_r":
                np.nanmedian(null),
            "null_p95": p95,
            "percent_replicating":
                np.nanmean(matched > p95),
        })

    return pd.DataFrame(rows)


# ============================================================
# WITHIN-SITE QC USING RAW REPLICATE WELLS
# ============================================================

def within_site_qc(
    transform,
    label,
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rows = []

    for site in REPORT_SITES:

        x = raw[
            (raw["Metadata_Site"] == site)
            & (~raw["Metadata_negcon"].astype(bool))
            & raw["Metadata_pdid"].notna()
        ].copy()

        x["Metadata_pdid"] = (
            x["Metadata_pdid"]
            .astype(str)
        )

        if pdid_sets[site]:

            x = x[
                x["Metadata_pdid"].isin(
                    pdid_sets[site]
                )
            ].copy()

        Y = transform(
            x[features]
        )

        within = []

        labels = (
            x["Metadata_pdid"]
            .to_numpy()
        )

        for pdid in np.unique(labels):

            idx = np.where(
                labels == pdid
            )[0]

            if len(idx) < 2:
                continue

            C = np.corrcoef(
                Y[idx]
            )

            iu = np.triu_indices(
                len(idx),
                k=1
            )

            within.append(
                np.nanmedian(C[iu])
            )

        within = np.asarray(within)

        null = []

        while len(null) < N_NULL:

            i = rng.integers(
                0,
                len(Y),
                size=N_NULL
            )

            j = rng.integers(
                0,
                len(Y),
                size=N_NULL
            )

            valid = (
                (i != j)
                & (labels[i] != labels[j])
            )

            vals = row_corr(
                Y[i[valid]],
                Y[j[valid]]
            )

            null.extend(
                vals[
                    np.isfinite(vals)
                ].tolist()
            )

        null = np.asarray(
            null[:N_NULL]
        )

        p95 = np.quantile(
            null,
            0.95
        )

        rows.append({
            "transform": label,
            "site": site,
            "n_compounds":
                len(within),
            "median_within_site_r":
                np.nanmedian(within),
            "null_p95": p95,
            "percent_replicating":
                np.nanmean(
                    within > p95
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# WELL-LEVEL SITE/COMPOUND NEIGHBOUR QC
# ============================================================

def neighbour_qc(
    transform,
    label,
):

    frames = []
    ys = []

    for site in REPORT_SITES:

        x = raw[
            (raw["Metadata_Site"] == site)
            & (~raw["Metadata_negcon"].astype(bool))
            & raw["Metadata_pdid"].notna()
        ].copy()

        x["Metadata_pdid"] = (
            x["Metadata_pdid"]
            .astype(str)
        )

        if pdid_sets[site]:

            x = x[
                x["Metadata_pdid"].isin(
                    pdid_sets[site]
                )
            ].copy()

        Y = transform(
            x[features]
        )

        frames.append(
            x[
                [
                    "Metadata_Site",
                    "Metadata_pdid"
                ]
            ].reset_index(drop=True)
        )

        ys.append(Y)

    meta = pd.concat(
        frames,
        ignore_index=True
    )

    Y = np.vstack(ys)

    Y = StandardScaler().fit_transform(
        Y
    )

    nn = NearestNeighbors(
        n_neighbors=K + 1,
        metric="euclidean",
        n_jobs=-1
    )

    nn.fit(Y)

    idx = nn.kneighbors(
        Y,
        return_distance=False
    )[:, 1:]

    sites = (
        meta["Metadata_Site"]
        .to_numpy()
    )

    pdids = (
        meta["Metadata_pdid"]
        .to_numpy()
    )

    same_site = (
        sites[idx] ==
        sites[:, None]
    ).mean()

    same_compound = (
        pdids[idx] ==
        pdids[:, None]
    ).mean()

    return {
        "transform": label,
        "n_wells": len(Y),
        "same_site_nn":
            same_site,
        "same_compound_nn":
            same_compound,
    }


# ============================================================
# RUN ONLY THREE REPRESENTATIONS
# ============================================================

experiments = [
    (
        "baseline_MAD",
        identity_fit()
    ),
    (
        "global_ZCA_eps_0.1",
        global_zca_fit(0.1)
    ),
    (
        "global_ZCA_eps_1",
        global_zca_fit(1.0)
    ),
]


all_cross = []
all_within = []
all_nn = []


for label, transform in experiments:

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    transformed_consensus = {}

    for site in REPORT_SITES:

        base = baseline_consensus[site]

        Y = transform(
            base[features]
        )

        transformed_consensus[site] = (
            pd.DataFrame(
                Y,
                index=base.index,
                columns=features
            )
        )

    cross = cross_site_qc(
        transformed_consensus,
        label
    )

    within = within_site_qc(
        transform,
        label
    )

    nn = neighbour_qc(
        transform,
        label
    )

    all_cross.append(cross)
    all_within.append(within)
    all_nn.append(nn)

    print("\nCROSS-SITE:")
    print(
        cross.to_string(
            index=False
        )
    )

    print("\nWITHIN-SITE:")
    print(
        within.to_string(
            index=False
        )
    )

    print("\nNEIGHBOURS:")
    print(nn)

    # Save pooled drug representation
    pooled = pd.concat(
        [
            transformed_consensus[s]
            .assign(
                Metadata_Site=s,
                Metadata_Drug=
                    transformed_consensus[s].index
            )
            for s in REPORT_SITES
        ],
        ignore_index=True
    )

    pooled_drug = (
        pooled
        .groupby(
            "Metadata_Drug",
            observed=True
        )[features]
        .mean()
        .sort_index()
    )

    safe = (
        label
        .replace(".", "p")
        .replace(" ", "_")
    )

    pooled_drug.to_parquet(
        OUT /
        f"{safe}_pooled_drug_consensus.parquet"
    )


# ============================================================
# SUMMARY
# ============================================================

cross = pd.concat(
    all_cross,
    ignore_index=True
)

within = pd.concat(
    all_within,
    ignore_index=True
)

nn = pd.DataFrame(
    all_nn
)

cross.to_csv(
    OUT /
    "global_sphere_cross_site.csv",
    index=False
)

within.to_csv(
    OUT /
    "global_sphere_within_site.csv",
    index=False
)

nn.to_csv(
    OUT /
    "global_sphere_neighbours.csv",
    index=False
)


summary = (
    cross
    .groupby("transform")
    .agg(
        cross_r_mean=(
            "median_matched_r",
            "mean"
        ),
        cross_PR_mean=(
            "percent_replicating",
            "mean"
        ),
        cross_PR_min=(
            "percent_replicating",
            "min"
        ),
    )
)

wsummary = (
    within
    .groupby("transform")
    .agg(
        within_r_mean=(
            "median_within_site_r",
            "mean"
        ),
        within_PR_mean=(
            "percent_replicating",
            "mean"
        ),
    )
)

summary = (
    summary
    .join(wsummary)
    .reset_index()
    .merge(
        nn[
            [
                "transform",
                "same_site_nn",
                "same_compound_nn"
            ]
        ],
        on="transform"
    )
)


print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)

summary.to_csv(
    OUT /
    "global_sphere_final_summary.csv",
    index=False
)

print("\nSaved:")
print(
    OUT /
    "global_sphere_final_summary.csv"
)
print(
    OUT /
    "global_sphere_cross_site.csv"
)
print(
    OUT /
    "global_sphere_within_site.csv"
)
print(
    OUT /
    "global_sphere_neighbours.csv"
)
