#!/usr/bin/env python3

"""
Exploratory OpenScreen morphology <-> TAHOE gene-expression mapping.

IMPORTANT
---------
These are NOT paired single-cell measurements.

The two modalities are aligned only by perturbation identity:

    OpenScreen morphology: one consensus profile per drug
    TAHOE expression:      one consensus profile per drug

Associations therefore mean:

    "Across matched drug perturbations, morphology feature X
     covaries with expression of gene Y."

They do NOT mean:
    - same-cell association
    - causal regulation
    - gene Y causes morphology X

Primary analyses
----------------
1. Raw drug-level morphology-feature <-> gene Spearman map.
2. Direction-normalized sensitivity map:
       z-score features across drugs
       unit-normalize each drug vector
       correlate individual coordinates across drugs
   This reduces the influence of overall perturbation magnitude.
3. BH FDR across all feature-gene tests.
4. Leave-one-drug-out sensitivity for strongest associations.
5. Identify drugs contributing most strongly to selected associations.
6. Export ranked gene lists for downstream pathway enrichment.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, t as student_t
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

N_TOP_ASSOC = 500
N_LOO_ASSOC = 100
N_DRIVER_DRUGS = 5
N_TOP_MORPH_FOR_PATHWAYS = 25

FDR_THRESHOLD = 0.05
RHO_DISPLAY_THRESHOLD = 0.30

RANDOM_STATE = 0


# ============================================================
# HELPERS
# ============================================================

def bh_fdr(p):
    """Benjamini-Hochberg FDR."""
    p = np.asarray(p, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = ranked * n / np.arange(1, n + 1)

    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty(n, dtype=float)
    out[order] = q

    return out


def zscore_columns(X):
    X = np.asarray(X, dtype=np.float64)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)

    sd[sd == 0] = 1.0

    return (X - mu) / sd


def unit_normalize_rows(X):
    X = np.asarray(X, dtype=np.float64)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return X / norms


def spearman_matrix(X, Y):
    """
    Compute all column-wise Spearman correlations:
        X: n x p
        Y: n x q

    Returns p x q matrix.
    """

    n = X.shape[0]

    RX = rankdata(
        X,
        axis=0,
        method="average"
    )

    RY = rankdata(
        Y,
        axis=0,
        method="average"
    )

    RX = RX - RX.mean(axis=0, keepdims=True)
    RY = RY - RY.mean(axis=0, keepdims=True)

    sx = np.sqrt(
        np.sum(RX ** 2, axis=0)
    )

    sy = np.sqrt(
        np.sum(RY ** 2, axis=0)
    )

    sx[sx == 0] = np.nan
    sy[sy == 0] = np.nan

    corr = (
        RX.T @ RY
    ) / (
        sx[:, None] * sy[None, :]
    )

    corr = np.clip(
        corr,
        -1,
        1
    )

    return corr


def correlation_pvalues(R, n):
    """
    Approximate two-sided p-values using the standard
    t approximation for correlation.

    At n=119 this is suitable for exploratory Spearman testing.
    """

    R = np.asarray(R, dtype=float)

    denom = np.maximum(
        1.0 - R ** 2,
        1e-15
    )

    T = (
        R
        *
        np.sqrt(
            (n - 2) / denom
        )
    )

    P = 2 * student_t.sf(
        np.abs(T),
        df=n - 2
    )

    return P


def association_table(
    R,
    morph_names,
    gene_names,
    n,
    map_name
):
    P = correlation_pvalues(
        R,
        n
    )

    feature_idx, gene_idx = np.indices(
        R.shape
    )

    out = pd.DataFrame({
        "morphology_feature":
            np.asarray(morph_names)[feature_idx.ravel()],

        "gene":
            np.asarray(gene_names)[gene_idx.ravel()],

        "rho":
            R.ravel(),

        "p":
            P.ravel(),
    })

    out["q_bh"] = bh_fdr(
        out["p"].to_numpy()
    )

    out["abs_rho"] = np.abs(
        out["rho"]
    )

    out["association_map"] = map_name

    return out


def single_spearman(x, y):
    rx = rankdata(
        x,
        method="average"
    )

    ry = rankdata(
        y,
        method="average"
    )

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt(
        np.sum(rx ** 2)
        *
        np.sum(ry ** 2)
    )

    if denom == 0:
        return np.nan

    return np.sum(
        rx * ry
    ) / denom


def association_drivers(
    x,
    y,
    drugs,
    n_driver=5
):
    """
    Contribution to Pearson correlation of ranks.
    Used only descriptively to identify compounds supporting
    or opposing an association.
    """

    rx = rankdata(
        x,
        method="average"
    ).astype(float)

    ry = rankdata(
        y,
        method="average"
    ).astype(float)

    rx = (
        rx - rx.mean()
    ) / (
        rx.std(ddof=1)
        if rx.std(ddof=1) != 0
        else 1
    )

    ry = (
        ry - ry.mean()
    ) / (
        ry.std(ddof=1)
        if ry.std(ddof=1) != 0
        else 1
    )

    contribution = (
        rx * ry
    )

    order_support = np.argsort(
        contribution
    )[::-1]

    order_oppose = np.argsort(
        contribution
    )

    support = [
        str(drugs[i])
        for i in order_support[:n_driver]
    ]

    oppose = [
        str(drugs[i])
        for i in order_oppose[:n_driver]
    ]

    return (
        support,
        oppose,
        contribution
    )


def loo_sensitivity(x, y, drugs):
    full = single_spearman(
        x,
        y
    )

    loo = []

    for i in range(
        len(drugs)
    ):

        keep = np.ones(
            len(drugs),
            dtype=bool
        )

        keep[i] = False

        r = single_spearman(
            x[keep],
            y[keep]
        )

        loo.append(
            (
                str(drugs[i]),
                r,
                r - full
            )
        )

    loo = sorted(
        loo,
        key=lambda z: abs(z[2]),
        reverse=True
    )

    most_influential = loo[0]

    return {
        "rho_full": full,
        "rho_loo_min": np.nanmin(
            [z[1] for z in loo]
        ),
        "rho_loo_max": np.nanmax(
            [z[1] for z in loo]
        ),
        "most_influential_drug":
            most_influential[0],
        "max_abs_loo_change":
            abs(most_influential[2]),
    }


def make_heatmap(
    matrix,
    row_labels,
    col_labels,
    title,
    xlabel,
    ylabel,
    path
):
    plt.figure(
        figsize=(15, 10)
    )

    im = plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(
        im,
        label="Spearman rho"
    )

    plt.xticks(
        np.arange(
            len(col_labels)
        ),
        col_labels,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        np.arange(
            len(row_labels)
        ),
        row_labels,
        fontsize=7
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "morphology",
    help="119 x 636 morphology parquet"
)

parser.add_argument(
    "expression",
    help="plate-corrected drug expression parquet"
)

parser.add_argument(
    "--out",
    default="OpenScreen/feature_gene_exploratory",
    help="output directory"
)

args = parser.parse_args()


OUT = Path(
    args.out
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOAD DATA")
print("=" * 80)

M = pd.read_parquet(
    args.morphology
)

G = pd.read_parquet(
    args.expression
)

M.index = M.index.astype(str)
G.index = G.index.astype(str)

M.columns = M.columns.astype(str)
G.columns = G.columns.astype(str)


shared = sorted(
    set(M.index)
    &
    set(G.index)
)


print(
    "Morphology original:",
    M.shape
)

print(
    "Expression original:",
    G.shape
)

print(
    "Shared drugs:",
    len(shared)
)


M = M.loc[
    shared
].copy()

G = G.loc[
    shared
].copy()


print(
    "Aligned morphology:",
    M.shape
)

print(
    "Aligned expression:",
    G.shape
)


if len(shared) < 100:
    raise RuntimeError(
        "Unexpectedly small drug overlap. "
        "Check drug identifiers before continuing."
    )


if not np.isfinite(
    M.to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Morphology matrix contains NaN/Inf."
    )


if not np.isfinite(
    G.to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Expression matrix contains NaN/Inf."
    )


drugs = np.asarray(
    shared
)

morph_names = np.asarray(
    M.columns
)

gene_names = np.asarray(
    G.columns
)


# ============================================================
# SAVE ALIGNMENT
# ============================================================

pd.DataFrame({
    "drug": shared
}).to_csv(
    OUT / "matched_drugs.csv",
    index=False
)


# ============================================================
# ACTIVITY / PERTURBATION MAGNITUDE
# for descriptive sensitivity
# ============================================================

print("\n" + "=" * 80)
print("PERTURBATION MAGNITUDE")
print("=" * 80)


pca_m = PCA(
    n_components=30,
    random_state=RANDOM_STATE
)

pca_g = PCA(
    n_components=30,
    random_state=RANDOM_STATE
)


Mpc = pca_m.fit_transform(
    M.to_numpy(
        dtype=np.float64
    )
)

Gpc = pca_g.fit_transform(
    G.to_numpy(
        dtype=np.float64
    )
)


Mpc_z = zscore_columns(
    Mpc
)

Gpc_z = zscore_columns(
    Gpc
)


morph_activity = np.linalg.norm(
    Mpc_z,
    axis=1
)

expr_activity = np.linalg.norm(
    Gpc_z,
    axis=1
)


activity = pd.DataFrame({
    "drug": shared,
    "morphology_activity":
        morph_activity,
    "expression_activity":
        expr_activity,
})


activity.to_csv(
    OUT / "drug_perturbation_strength.csv",
    index=False
)


# ============================================================
# MAP A:
# RAW DRUG-LEVEL FEATURE <-> GENE SPEARMAN
# ============================================================

print("\n" + "=" * 80)
print("MAP A — RAW DRUG-LEVEL FEATURE ↔ GENE ASSOCIATIONS")
print("=" * 80)

MA = M.to_numpy(
    dtype=np.float64
)

GA = G.to_numpy(
    dtype=np.float64
)


R_raw = spearman_matrix(
    MA,
    GA
)


raw = association_table(
    R_raw,
    morph_names,
    gene_names,
    len(shared),
    "raw"
)


raw.to_parquet(
    OUT / "feature_gene_raw_all.parquet",
    index=False
)


raw_top = (
    raw
    .sort_values(
        "abs_rho",
        ascending=False
    )
    .head(
        N_TOP_ASSOC
    )
)


raw_top.to_csv(
    OUT / "feature_gene_raw_top500.csv",
    index=False
)


print(
    "Total tests:",
    len(raw)
)

print(
    "FDR q<0.05:",
    int(
        (
            raw["q_bh"]
            <
            FDR_THRESHOLD
        ).sum()
    )
)

print(
    f"|rho| >= {RHO_DISPLAY_THRESHOLD}:",
    int(
        (
            raw["abs_rho"]
            >=
            RHO_DISPLAY_THRESHOLD
        ).sum()
    )
)


print("\nTop 20 raw associations:")

print(
    raw_top[
        [
            "morphology_feature",
            "gene",
            "rho",
            "p",
            "q_bh"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# MAP B:
# DIRECTION-NORMALIZED FEATURE <-> GENE MAP
#
# 1. z-score each feature/gene across drugs
# 2. unit-normalize every drug profile separately
#
# This reduces global perturbation-strength effects.
# ============================================================

print("\n" + "=" * 80)
print("MAP B — DIRECTION-NORMALIZED ASSOCIATIONS")
print("=" * 80)


M_std = zscore_columns(
    MA
)

G_std = zscore_columns(
    GA
)


M_direction = unit_normalize_rows(
    M_std
)

G_direction = unit_normalize_rows(
    G_std
)


R_direction = spearman_matrix(
    M_direction,
    G_direction
)


direction = association_table(
    R_direction,
    morph_names,
    gene_names,
    len(shared),
    "direction_normalized"
)


direction.to_parquet(
    OUT / "feature_gene_direction_all.parquet",
    index=False
)


direction_top = (
    direction
    .sort_values(
        "abs_rho",
        ascending=False
    )
    .head(
        N_TOP_ASSOC
    )
)


direction_top.to_csv(
    OUT / "feature_gene_direction_top500.csv",
    index=False
)


print(
    "FDR q<0.05:",
    int(
        (
            direction["q_bh"]
            <
            FDR_THRESHOLD
        ).sum()
    )
)

print(
    f"|rho| >= {RHO_DISPLAY_THRESHOLD}:",
    int(
        (
            direction["abs_rho"]
            >=
            RHO_DISPLAY_THRESHOLD
        ).sum()
    )
)


print(
    "\nTop 20 direction-normalized associations:"
)

print(
    direction_top[
        [
            "morphology_feature",
            "gene",
            "rho",
            "p",
            "q_bh"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# COMBINE RAW + DIRECTION RESULTS
# ============================================================

print("\n" + "=" * 80)
print("RAW vs DIRECTION COMPARISON")
print("=" * 80)


comparison = raw[
    [
        "morphology_feature",
        "gene",
        "rho",
        "q_bh"
    ]
].rename(
    columns={
        "rho": "rho_raw",
        "q_bh": "q_raw"
    }
).merge(
    direction[
        [
            "morphology_feature",
            "gene",
            "rho",
            "q_bh"
        ]
    ].rename(
        columns={
            "rho": "rho_direction",
            "q_bh": "q_direction"
        }
    ),
    on=[
        "morphology_feature",
        "gene"
    ],
    how="inner"
)


comparison[
    "abs_rho_raw"
] = np.abs(
    comparison[
        "rho_raw"
    ]
)


comparison[
    "abs_rho_direction"
] = np.abs(
    comparison[
        "rho_direction"
    ]
)


comparison[
    "direction_retention"
] = (
    comparison[
        "abs_rho_direction"
    ]
    /
    comparison[
        "abs_rho_raw"
    ].replace(
        0,
        np.nan
    )
)


comparison[
    "delta_abs_rho"
] = (
    comparison[
        "abs_rho_direction"
    ]
    -
    comparison[
        "abs_rho_raw"
    ]
)


comparison.to_parquet(
    OUT / "feature_gene_raw_vs_direction.parquet",
    index=False
)


comparison.sort_values(
    "abs_rho_direction",
    ascending=False
).head(
    N_TOP_ASSOC
).to_csv(
    OUT / "feature_gene_raw_vs_direction_top500.csv",
    index=False
)


# ============================================================
# LEAVE-ONE-DRUG-OUT + DRIVER ANALYSIS
# strongest 100 associations from both maps
# ============================================================

print("\n" + "=" * 80)
print("DRUG DRIVER / OUTLIER SENSITIVITY")
print("=" * 80)


candidate_pairs = pd.concat([
    raw_top[
        [
            "morphology_feature",
            "gene"
        ]
    ].head(
        N_LOO_ASSOC
    ),

    direction_top[
        [
            "morphology_feature",
            "gene"
        ]
    ].head(
        N_LOO_ASSOC
    ),
]).drop_duplicates()


morph_lookup = {
    x: i
    for i, x
    in enumerate(
        morph_names
    )
}

gene_lookup = {
    x: i
    for i, x
    in enumerate(
        gene_names
    )
}


driver_rows = []


for k, row in enumerate(
    candidate_pairs.itertuples(
        index=False
    ),
    start=1
):

    mf = row.morphology_feature
    gene = row.gene

    mi = morph_lookup[
        mf
    ]

    gi = gene_lookup[
        gene
    ]


    x_raw = MA[
        :,
        mi
    ]

    y_raw = GA[
        :,
        gi
    ]


    x_dir = M_direction[
        :,
        mi
    ]

    y_dir = G_direction[
        :,
        gi
    ]


    raw_loo = loo_sensitivity(
        x_raw,
        y_raw,
        drugs
    )

    dir_loo = loo_sensitivity(
        x_dir,
        y_dir,
        drugs
    )


    raw_support, raw_oppose, _ = (
        association_drivers(
            x_raw,
            y_raw,
            drugs,
            N_DRIVER_DRUGS
        )
    )


    dir_support, dir_oppose, _ = (
        association_drivers(
            x_dir,
            y_dir,
            drugs,
            N_DRIVER_DRUGS
        )
    )


    driver_rows.append({
        "morphology_feature": mf,
        "gene": gene,

        "rho_raw":
            raw_loo[
                "rho_full"
            ],

        "raw_loo_min":
            raw_loo[
                "rho_loo_min"
            ],

        "raw_loo_max":
            raw_loo[
                "rho_loo_max"
            ],

        "raw_most_influential_drug":
            raw_loo[
                "most_influential_drug"
            ],

        "raw_max_abs_loo_change":
            raw_loo[
                "max_abs_loo_change"
            ],

        "raw_top_supporting_drugs":
            "; ".join(
                raw_support
            ),

        "raw_top_opposing_drugs":
            "; ".join(
                raw_oppose
            ),

        "rho_direction":
            dir_loo[
                "rho_full"
            ],

        "direction_loo_min":
            dir_loo[
                "rho_loo_min"
            ],

        "direction_loo_max":
            dir_loo[
                "rho_loo_max"
            ],

        "direction_most_influential_drug":
            dir_loo[
                "most_influential_drug"
            ],

        "direction_max_abs_loo_change":
            dir_loo[
                "max_abs_loo_change"
            ],

        "direction_top_supporting_drugs":
            "; ".join(
                dir_support
            ),

        "direction_top_opposing_drugs":
            "; ".join(
                dir_oppose
            ),
    })


drivers = pd.DataFrame(
    driver_rows
)


drivers.to_csv(
    OUT / "top_association_drug_drivers_and_LOO.csv",
    index=False
)


print(
    "Associations checked:",
    len(drivers)
)


print(
    "\nLargest leave-one-drug-out changes:"
)


print(
    drivers.sort_values(
        "direction_max_abs_loo_change",
        ascending=False
    )[
        [
            "morphology_feature",
            "gene",
            "rho_direction",
            "direction_most_influential_drug",
            "direction_max_abs_loo_change"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# PATHWAY-READY GENE RANKINGS
#
# Find morphology features with strongest direction-normalized
# associations and save all 2000 genes ranked by rho.
# ============================================================

print("\n" + "=" * 80)
print("EXPORT PATHWAY-READY GENE RANKINGS")
print("=" * 80)


feature_strength = (
    direction
    .groupby(
        "morphology_feature"
    )["abs_rho"]
    .max()
    .sort_values(
        ascending=False
    )
)


top_morph_features = (
    feature_strength
    .head(
        N_TOP_MORPH_FOR_PATHWAYS
    )
    .index
    .tolist()
)


safe_manifest = []


for rank_i, feature in enumerate(
    top_morph_features,
    start=1
):

    temp = (
        direction[
            direction[
                "morphology_feature"
            ]
            ==
            feature
        ][
            [
                "gene",
                "rho",
                "p",
                "q_bh"
            ]
        ]
        .sort_values(
            "rho",
            ascending=False
        )
    )


    filename = (
        f"gene_ranking_"
        f"{rank_i:02d}.tsv"
    )


    temp.to_csv(
        OUT / filename,
        sep="\t",
        index=False
    )


    safe_manifest.append({
        "rank": rank_i,
        "morphology_feature": feature,
        "max_abs_direction_rho":
            feature_strength[
                feature
            ],
        "ranking_file":
            filename
    })


pd.DataFrame(
    safe_manifest
).to_csv(
    OUT / "pathway_ranking_manifest.csv",
    index=False
)


# ============================================================
# FIGURE 1
# RAW vs DIRECTION rho
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.hexbin(
    comparison[
        "rho_raw"
    ],
    comparison[
        "rho_direction"
    ],
    gridsize=60,
    mincnt=1
)


plt.colorbar(
    label="Feature-gene pairs"
)


plt.axhline(
    0,
    linewidth=1
)

plt.axvline(
    0,
    linewidth=1
)


plt.xlabel(
    "Raw drug-level Spearman rho"
)

plt.ylabel(
    "Direction-normalized Spearman rho"
)

plt.title(
    "Morphology-gene associations before and after\n"
    "reducing perturbation-magnitude effects"
)


plt.tight_layout()

plt.savefig(
    OUT / "01_raw_vs_direction_associations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# TOP DIRECTION-NORMALIZED ASSOCIATIONS
# ============================================================

topplot = (
    direction_top
    .head(30)
    .copy()
)


labels = (
    topplot[
        "morphology_feature"
    ].str.slice(
        0,
        45
    )
    +
    " | "
    +
    topplot[
        "gene"
    ]
)


plt.figure(
    figsize=(10, 10)
)


y = np.arange(
    len(topplot)
)


plt.barh(
    y,
    topplot[
        "rho"
    ]
)


plt.yticks(
    y,
    labels,
    fontsize=7
)


plt.gca().invert_yaxis()


plt.xlabel(
    "Direction-normalized Spearman rho"
)

plt.title(
    "Top exploratory morphology-gene associations"
)


plt.tight_layout()

plt.savefig(
    OUT / "02_top_direction_associations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# HEATMAP:
# strongest morphology features x strongest genes
# ============================================================

top_features_heat = (
    direction
    .groupby(
        "morphology_feature"
    )["abs_rho"]
    .max()
    .nlargest(25)
    .index
)


top_genes_heat = (
    direction
    .groupby(
        "gene"
    )["abs_rho"]
    .max()
    .nlargest(40)
    .index
)


feature_pos = {
    f: i
    for i, f
    in enumerate(
        morph_names
    )
}

gene_pos = {
    g: i
    for i, g
    in enumerate(
        gene_names
    )
}


heat = np.empty(
    (
        len(
            top_features_heat
        ),
        len(
            top_genes_heat
        )
    ),
    dtype=float
)


for i, f in enumerate(
    top_features_heat
):

    for j, g in enumerate(
        top_genes_heat
    ):

        heat[
            i,
            j
        ] = R_direction[
            feature_pos[
                f
            ],
            gene_pos[
                g
            ]
        ]


make_heatmap(
    heat,
    list(
        top_features_heat
    ),
    list(
        top_genes_heat
    ),
    "Direction-normalized morphology ↔ gene association map",
    "Genes",
    "Morphology features",
    OUT / "03_direction_feature_gene_heatmap.png"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "n_drugs":
        len(shared),

    "n_morphology_features":
        M.shape[1],

    "n_genes":
        G.shape[1],

    "n_feature_gene_tests":
        M.shape[1]
        *
        G.shape[1],

    "raw_max_abs_rho":
        raw["abs_rho"].max(),

    "raw_n_fdr_005":
        int(
            (
                raw[
                    "q_bh"
                ]
                <
                0.05
            ).sum()
        ),

    "direction_max_abs_rho":
        direction[
            "abs_rho"
        ].max(),

    "direction_n_fdr_005":
        int(
            (
                direction[
                    "q_bh"
                ]
                <
                0.05
            ).sum()
        ),

    "raw_n_absrho_ge_030":
        int(
            (
                raw[
                    "abs_rho"
                ]
                >=
                0.30
            ).sum()
        ),

    "direction_n_absrho_ge_030":
        int(
            (
                direction[
                    "abs_rho"
                ]
                >=
                0.30
            ).sum()
        ),
}


pd.DataFrame(
    [summary]
).to_csv(
    OUT / "feature_gene_summary.csv",
    index=False
)


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


for k, v in summary.items():
    print(
        f"{k:32s}: {v}"
    )


print("\nCreated:")

for f in sorted(
    OUT.iterdir()
):
    print(
        " ",
        f
    )


print("\nINTERPRETATION RULE:")
print(
    "These are exploratory associations ACROSS matched drugs. "
    "They are not paired-cell associations and do not establish causality."
)

