#!/usr/bin/env python3

"""
FINAL LINCS A549 Task-2 validation.

PRIMARY MORPHOLOGY RULE
-----------------------
Use already processed LINCS morphology consensus.
Exclude only numerically pathological features where:

    max absolute value across profiles > 1e6

This removes 41 extreme features and retains 574.

Expression is ALREADY processed:
- normalize_total(1e4)
- log1p
- 2000 HVGs
- PCA

Therefore we use the stored expression X_pca and DO NOT preprocess it again.

Final analysis:
- morphology PCA = 30 PCs
- average morphology dose profiles by drug
- average existing expression PCs by drug
- match drugs
- z-score PC axes
- all drug-pair Euclidean distances
- Spearman + drug-label permutation
- perturbation strength
- direction only
- PC1-drop sensitivity
- presentation figures
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ============================================================
# SETTINGS
# ============================================================

SEED = 0
N_PCS = 30
N_PERM = 999

PATHOLOGICAL_THRESHOLD = 1e6

MORPH_FILE = Path(
    "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad"
)

EXPR_FILE = Path(
    "LINCS/data/lincs_expression_a549.h5ad"
)

OUT = Path(
    "LINCS/task2_final"
)

FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def zscore_columns(X):

    X = np.asarray(X, dtype=float)

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)

    sd[sd == 0] = 1

    return (X - mu) / sd


def unit_rows(X):

    X = np.asarray(X, dtype=float)

    norm = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    norm[norm == 0] = 1

    return X / norm


def distance_matrix(X):

    return squareform(
        pdist(
            X,
            metric="euclidean"
        )
    )


def geometry_permutation(Dm, Dg):

    n = Dm.shape[0]

    iu = np.triu_indices(
        n,
        k=1
    )

    x = Dm[iu]
    y = Dg[iu]

    r = float(
        spearmanr(x, y).statistic
    )

    rng = np.random.default_rng(SEED)

    null = np.zeros(
        N_PERM
    )

    idx = np.arange(n)

    for b in range(N_PERM):

        perm = rng.permutation(idx)

        Dgp = Dg[
            np.ix_(perm, perm)
        ]

        null[b] = spearmanr(
            x,
            Dgp[iu]
        ).statistic


    p_one = (
        1
        +
        np.sum(null >= r)
    ) / (
        N_PERM + 1
    )


    p_two = (
        1
        +
        np.sum(
            np.abs(null)
            >=
            abs(r)
        )
    ) / (
        N_PERM + 1
    )


    return r, p_one, p_two, null


def strength_permutation(x, y):

    rho = float(
        spearmanr(
            x,
            y
        ).statistic
    )

    rng = np.random.default_rng(
        SEED
    )

    null = np.zeros(
        N_PERM
    )

    for b in range(N_PERM):

        perm = rng.permutation(
            len(y)
        )

        null[b] = spearmanr(
            x,
            y[perm]
        ).statistic


    p = (
        1
        +
        np.sum(
            null >= rho
        )
    ) / (
        N_PERM + 1
    )

    return rho, p, null


# ============================================================
# LOAD MORPHOLOGY
# ============================================================

print("=" * 100)
print("FINAL LINCS A549 TASK-2 VALIDATION")
print("=" * 100)

m = ad.read_h5ad(
    MORPH_FILE
)

Xm = np.asarray(
    m.X,
    dtype=float
)

feature_names = np.asarray(
    m.var_names.astype(str)
)

feature_max = np.nanmax(
    np.abs(Xm),
    axis=0
)

finite = np.all(
    np.isfinite(Xm),
    axis=0
)


keep = (
    finite
    &
    (
        feature_max
        <=
        PATHOLOGICAL_THRESHOLD
    )
)


print("\nMORPHOLOGY")

print(
    "Original:",
    m.shape
)

print(
    "Original features:",
    m.n_vars
)

print(
    "Pathological features excluded:",
    (~keep).sum()
)

print(
    "Final features:",
    keep.sum()
)


# Freeze expected result
assert keep.sum() == 574, (
    f"Expected 574 morphology features, "
    f"found {keep.sum()}."
)


excluded = pd.DataFrame({
    "feature": feature_names[~keep],
    "max_abs": feature_max[~keep],
}).sort_values(
    "max_abs",
    ascending=False
)


excluded.to_csv(
    OUT /
    "excluded_pathological_morphology_features.csv",
    index=False
)


included = pd.DataFrame({
    "feature": feature_names[keep],
    "max_abs": feature_max[keep],
})


included.to_csv(
    OUT /
    "retained_morphology_features_574.csv",
    index=False
)


Xm = Xm[
    :,
    keep
]


# ============================================================
# MORPHOLOGY PCA
# ============================================================

pca_m = PCA(
    n_components=N_PCS,
    random_state=SEED,
    svd_solver="full"
)

M_dose = pca_m.fit_transform(
    Xm
)


print(
    "\nMorphology PC1 variance:",
    f"{pca_m.explained_variance_ratio_[0]:.6f}"
)

print(
    "Morphology first 5 cumulative:",
    f"{pca_m.explained_variance_ratio_[:5].sum():.6f}"
)


# Compound-dose -> drug mean
Mdf = pd.DataFrame(
    M_dose,
    columns=[
        f"PC{i+1}"
        for i in range(N_PCS)
    ]
)

Mdf["drug"] = (
    m.obs[
        "Metadata_Drug"
    ]
    .astype(str)
    .to_numpy()
)


Mdrug = (
    Mdf
    .groupby(
        "drug",
        observed=True
    )
    .mean()
)


print(
    "Morphology drug profiles:",
    Mdrug.shape
)


# ============================================================
# LOAD EXISTING PROCESSED EXPRESSION PCA
# ============================================================

print("\nEXPRESSION")

g = ad.read_h5ad(
    EXPR_FILE,
    backed="r"
)


print(
    "Cells:",
    g.n_obs
)

print(
    "Genes:",
    g.n_vars
)

print(
    "Drugs:",
    g.obs[
        "Metadata_Drug"
    ].nunique()
)

print(
    "HVGs:",
    int(
        g.var[
            "highly_variable"
        ].sum()
    )
)


if "X_pca" not in g.obsm:

    raise RuntimeError(
        "Expression X_pca missing."
    )


G_cells = np.asarray(
    g.obsm[
        "X_pca"
    ][:, :N_PCS],
    dtype=float
)


Gdf = pd.DataFrame(
    G_cells,
    columns=[
        f"PC{i+1}"
        for i in range(N_PCS)
    ]
)

Gdf["drug"] = (
    g.obs[
        "Metadata_Drug"
    ]
    .astype(str)
    .to_numpy()
)


Gdrug = (
    Gdf
    .groupby(
        "drug",
        observed=True
    )
    .mean()
)


g.file.close()


print(
    "Expression drug profiles:",
    Gdrug.shape
)


# ============================================================
# MATCH
# ============================================================

shared = sorted(
    set(Mdrug.index)
    &
    set(Gdrug.index)
)


print("\nMATCHING")

print(
    "Shared drugs:",
    len(shared)
)


assert len(shared) == 86, (
    f"Expected 86 shared drugs, "
    f"found {len(shared)}."
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
    "Unique drug pairs:",
    n_pairs
)


M = Mdrug.loc[
    shared
].to_numpy()

G = Gdrug.loc[
    shared
].to_numpy()


# ============================================================
# STANDARDIZE PC AXES
# ============================================================

Mz = zscore_columns(
    M
)

Gz = zscore_columns(
    G
)


# ============================================================
# OVERALL GEOMETRY
# ============================================================

Dm = distance_matrix(
    Mz
)

Dg = distance_matrix(
    Gz
)


r_all, p_all, p2_all, null_all = geometry_permutation(
    Dm,
    Dg
)


# ============================================================
# PERTURBATION STRENGTH
# ============================================================

strength_m = np.linalg.norm(
    Mz,
    axis=1
)

strength_g = np.linalg.norm(
    Gz,
    axis=1
)


rho_strength, p_strength, null_strength = strength_permutation(
    strength_m,
    strength_g
)


# ============================================================
# DIRECTION ONLY
# ============================================================

Mdir = unit_rows(
    Mz
)

Gdir = unit_rows(
    Gz
)


Dm_dir = distance_matrix(
    Mdir
)

Dg_dir = distance_matrix(
    Gdir
)


r_dir, p_dir, p2_dir, null_dir = geometry_permutation(
    Dm_dir,
    Dg_dir
)


# ============================================================
# DROP PC1
# ============================================================

M_no1 = unit_rows(
    Mz[:, 1:]
)

G_no1 = unit_rows(
    Gz[:, 1:]
)


r_no1, p_no1, _, _ = geometry_permutation(
    distance_matrix(
        M_no1
    ),
    distance_matrix(
        G_no1
    )
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print("\n" + "=" * 100)
print("FINAL PRIMARY RESULT")
print("=" * 100)

print(
    f"Drugs               : {n}"
)

print(
    f"Drug pairs          : {n_pairs}"
)

print(
    f"Morphology features : {keep.sum()}"
)

print()

print(
    f"Overall geometry     : "
    f"r={r_all:.6f}, "
    f"perm p={p_all:.4f}"
)

print(
    f"Strength             : "
    f"rho={rho_strength:.6f}, "
    f"perm p={p_strength:.4f}"
)

print(
    f"Direction only       : "
    f"r={r_dir:.6f}, "
    f"perm p={p_dir:.4f}"
)

print(
    f"Direction PC1 drop   : "
    f"r={r_no1:.6f}, "
    f"perm p={p_no1:.4f}"
)


# ============================================================
# SAVE FINAL DRUG MATRICES
# ============================================================

Mfinal = pd.DataFrame(
    Mz,
    index=shared,
    columns=[
        f"PC{i+1}"
        for i in range(N_PCS)
    ]
)

Gfinal = pd.DataFrame(
    Gz,
    index=shared,
    columns=[
        f"PC{i+1}"
        for i in range(N_PCS)
    ]
)


Mfinal.index.name = "drug"
Gfinal.index.name = "drug"


Mfinal.to_csv(
    OUT /
    "LINCS_A549_FINAL_morphology_86x30PC.csv"
)

Gfinal.to_csv(
    OUT /
    "LINCS_A549_FINAL_expression_86x30PC.csv"
)


# ============================================================
# SAVE STRENGTH
# ============================================================

strength_df = pd.DataFrame({
    "drug": shared,
    "morphology_strength": strength_m,
    "expression_strength": strength_g,
})


strength_df.to_csv(
    OUT /
    "LINCS_A549_FINAL_strength.csv",
    index=False
)


# ============================================================
# SAVE PAIRWISE TABLE
# ============================================================

iu = np.triu_indices(
    n,
    k=1
)


pairs = pd.DataFrame({
    "drug_A": [
        shared[i]
        for i in iu[0]
    ],
    "drug_B": [
        shared[j]
        for j in iu[1]
    ],
    "morphology_distance": Dm[iu],
    "expression_distance": Dg[iu],
    "morphology_direction_distance": Dm_dir[iu],
    "expression_direction_distance": Dg_dir[iu],
})


pairs.to_csv(
    OUT /
    "LINCS_A549_FINAL_3655_drug_pairs.csv",
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "dataset": "LINCS_A549",
    "morphology_rule": "max_abs_value<=1e6",
    "excluded_pathological_features": int((~keep).sum()),
    "n_morphology_features": int(keep.sum()),
    "n_expression_HVGs": 2000,
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
    "morph_PC1_variance": pca_m.explained_variance_ratio_[0],
}])


summary.to_csv(
    OUT /
    "LINCS_A549_FINAL_summary.csv",
    index=False
)


# ============================================================
# FIGURE 1
# EASY-TO-UNDERSTAND BINNED CROSS-MODAL RELATIONSHIP
# ============================================================

plot_df = pairs[
    [
        "morphology_distance",
        "expression_distance"
    ]
].copy()


plot_df["bin"] = pd.qcut(
    plot_df[
        "morphology_distance"
    ],
    q=10,
    labels=False,
    duplicates="drop"
)


bin_df = (
    plot_df
    .groupby(
        "bin",
        observed=True
    )
    .agg(
        morph_median=(
            "morphology_distance",
            "median"
        ),
        ge_median=(
            "expression_distance",
            "median"
        ),
        ge_q25=(
            "expression_distance",
            lambda x:
                np.quantile(
                    x,
                    .25
                )
        ),
        ge_q75=(
            "expression_distance",
            lambda x:
                np.quantile(
                    x,
                    .75
                )
        ),
        n=(
            "expression_distance",
            "size"
        )
    )
    .reset_index()
)


bin_df.to_csv(
    OUT /
    "LINCS_A549_FINAL_visualization_bins.csv",
    index=False
)


x = bin_df[
    "morph_median"
].to_numpy()

y = bin_df[
    "ge_median"
].to_numpy()

lower = (
    y
    -
    bin_df[
        "ge_q25"
    ].to_numpy()
)

upper = (
    bin_df[
        "ge_q75"
    ].to_numpy()
    -
    y
)


fig, ax = plt.subplots(
    figsize=(10.5, 7.5)
)


ax.plot(
    x,
    y,
    marker="o",
    markersize=10,
    linewidth=3
)


ax.errorbar(
    x,
    y,
    yerr=[
        lower,
        upper
    ],
    fmt="none",
    capsize=5,
    linewidth=2
)


ax.set_xlabel(
    "Morphology difference between drug pairs",
    fontsize=16
)

ax.set_ylabel(
    "Gene-expression difference between the same drug pairs",
    fontsize=16
)


ax.set_title(
    "LINCS A549: morphology and gene-expression\n"
    "drug relationships are reproducibly associated",
    fontsize=20,
    pad=15
)


ax.text(
    .05,
    .95,
    (
        f"{n} matched A549 drugs\n"
        f"{n_pairs:,} unique drug pairs\n\n"
        f"Spearman r = {r_all:.3f}\n"
        f"Permutation p = {p_all:.3f}"
    ),
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
    bbox=dict(
        boxstyle="round,pad=.5",
        facecolor="white",
        edgecolor=".7",
        alpha=.95
    )
)


ax.text(
    .97,
    .05,
    (
        "Each displayed point = one morphology-distance decile\n"
        "Line = median GE distance\n"
        "Bars = interquartile range\n"
        "Statistics use all 3,655 drug pairs"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
)


plt.tight_layout()


plt.savefig(
    FIG /
    "01_LINCS_A549_crossmodal_relationship.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIG /
    "01_LINCS_A549_crossmodal_relationship.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# HEPG2 vs LINCS REPLICATION
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Overall geometry",
        "Perturbation strength",
        "Direction only",
    ],
    "HepG2 / TAHOE": [
        0.370,
        0.438,
        0.019,
    ],
    "A549 / LINCS": [
        r_all,
        rho_strength,
        r_dir,
    ],
})


comparison.to_csv(
    OUT /
    "HepG2_vs_A549_crossmodal_comparison.csv",
    index=False
)


metrics = comparison[
    "Metric"
].tolist()

hep = comparison[
    "HepG2 / TAHOE"
].to_numpy()

lincs = comparison[
    "A549 / LINCS"
].to_numpy()


xx = np.arange(
    len(metrics)
)

width = .36


fig, ax = plt.subplots(
    figsize=(10.5, 7)
)


bars1 = ax.bar(
    xx - width/2,
    hep,
    width,
    label="HepG2 / TAHOE"
)

bars2 = ax.bar(
    xx + width/2,
    lincs,
    width,
    label="A549 / LINCS"
)


ax.set_xticks(
    xx
)

ax.set_xticklabels(
    metrics,
    fontsize=13
)

ax.set_ylabel(
    "Cross-modal correlation",
    fontsize=15
)

ax.set_title(
    "The strength-dominated cross-modal pattern\n"
    "reproduces in independent A549 / LINCS data",
    fontsize=20
)

ax.axhline(
    0,
    linewidth=1
)

ax.legend(
    frameon=False,
    fontsize=12
)


for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        h = bar.get_height()

        ax.text(
            bar.get_x()
            +
            bar.get_width()/2,
            h + .012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=11
        )


ax.text(
    .5,
    -.16,
    (
        "Overall: permutation p=.001 in both datasets   |   "
        "Direction: nonsignificant in both datasets"
    ),
    transform=ax.transAxes,
    ha="center",
    fontsize=11
)


plt.tight_layout()


plt.savefig(
    FIG /
    "02_HepG2_vs_LINCS_replication.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIG /
    "02_HepG2_vs_LINCS_replication.pdf",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# NUMERIC FILTER SENSITIVITY
# ============================================================

sensitivity_file = Path(
    "LINCS/task2_validation/"
    "lincs_task2_sensitivity_summary.csv"
)


if sensitivity_file.exists():

    sens = pd.read_csv(
        sensitivity_file
    )

    # Keep the interpretable stable sets + original
    order = [
        "strict_100",
        "moderate_1000",
        "pathological_only_1e6",
        "all_finite_original",
    ]

    sens = (
        sens[
            sens[
                "analysis"
            ].isin(order)
        ]
        .set_index(
            "analysis"
        )
        .reindex(
            order
        )
        .reset_index()
    )


    labels = [
        "≤100\n569 feat.",
        "≤1,000\n572 feat.",
        "≤1e6\n574 feat.",
        "All\n615 feat.",
    ]


    x2 = np.arange(
        len(labels)
    )


    fig, ax = plt.subplots(
        figsize=(10.5, 7)
    )


    ax.plot(
        x2,
        sens[
            "overall_r"
        ],
        marker="o",
        linewidth=2.5,
        label="Overall geometry"
    )

    ax.plot(
        x2,
        sens[
            "strength_rho"
        ],
        marker="o",
        linewidth=2.5,
        label="Perturbation strength"
    )

    ax.plot(
        x2,
        sens[
            "direction_r"
        ],
        marker="o",
        linewidth=2.5,
        label="Direction only"
    )


    ax.set_xticks(
        x2
    )

    ax.set_xticklabels(
        labels,
        fontsize=12
    )

    ax.set_ylabel(
        "Cross-modal correlation",
        fontsize=15
    )

    ax.set_title(
        "LINCS conclusion is stable after excluding\n"
        "numerically pathological morphology features",
        fontsize=20
    )

    ax.legend(
        frameon=False
    )


    ax.axhline(
        0,
        linewidth=1
    )


    plt.tight_layout()


    plt.savefig(
        FIG /
        "03_LINCS_numeric_sensitivity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        FIG /
        "03_LINCS_numeric_sensitivity.pdf",
        bbox_inches="tight"
    )

    plt.close()


print("\n" + "=" * 100)
print("FILES CREATED")
print("=" * 100)

for p in sorted(
    OUT.rglob("*")
):
    if p.is_file():
        print(p)

print("\nDONE")
