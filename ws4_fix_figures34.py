#!/usr/bin/env python3

"""
Regenerate stale TAHOE Figures 3 and 4 after discovery of plate structure.

FIGURE 3
--------
Cross-plate replicate QC only:
  same-drug / cross-plate pairs
  versus
  different-drug / cross-plate null

Shown before and after plate correction.

FIGURE 4
--------
Sample-level PCA coloured by experimental plate,
before and after plate correction.

No new biological inference is performed here.
This script only replaces superseded presentation QC figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

DATA = Path("OpenScreen/data")
OUT = Path("OpenScreen/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

RAW_SAMPLE = DATA / "hepg2_sample_log1p_2000hvg.parquet"

CORRECTED_SAMPLE = Path(
    "hepg2_platecorrected_sample_2000hvg.parquet"
)

META_H5AD = DATA / "hepg2_sample_pseudobulk_counts.h5ad"

FIG3 = OUT / "03_tahoe_replicate_qc.png"
FIG4 = OUT / "04_tahoe_drug_pca.png"

RANDOM_STATE = 0


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
})


# ============================================================
# LOAD / ALIGN
# ============================================================

print("=" * 80)
print("LOAD SAMPLE-LEVEL DATA")
print("=" * 80)

Xraw = pd.read_parquet(
    RAW_SAMPLE
)

Xcorr = pd.read_parquet(
    CORRECTED_SAMPLE
)

a = ad.read_h5ad(
    META_H5AD,
    backed="r"
)

meta = a.obs.copy()

try:
    a.file.close()
except Exception:
    pass


Xraw.index = Xraw.index.astype(str)
Xcorr.index = Xcorr.index.astype(str)
meta.index = meta.index.astype(str)


shared = sorted(
    set(Xraw.index)
    &
    set(Xcorr.index)
    &
    set(meta.index)
)


Xraw = Xraw.loc[
    shared
].copy()

Xcorr = Xcorr.loc[
    shared
].copy()

meta = meta.loc[
    shared
].copy()


# If a keep flag exists, honour it.
if "keep" in meta.columns:

    k = meta["keep"]

    if k.dtype == bool:
        keep = k.to_numpy()

    else:
        keep = (
            k.astype(str)
            .str.lower()
            .isin([
                "true",
                "1",
                "yes"
            ])
            .to_numpy()
        )

    if keep.sum() > 0:
        Xraw = Xraw.iloc[keep]
        Xcorr = Xcorr.iloc[keep]
        meta = meta.iloc[keep]


if "drug" not in meta.columns:
    raise RuntimeError(
        f"'drug' missing from metadata: {list(meta.columns)}"
    )

if "plate" not in meta.columns:
    raise RuntimeError(
        f"'plate' missing from metadata: {list(meta.columns)}"
    )


print(
    "Aligned retained samples:",
    len(meta)
)

print(
    "Drugs:",
    meta["drug"].astype(str).nunique()
)

print(
    "Plates:",
    meta["plate"].astype(str).nunique()
)

print(
    "Raw matrix:",
    Xraw.shape
)

print(
    "Corrected matrix:",
    Xcorr.shape
)


assert list(Xraw.index) == list(Xcorr.index)
assert list(Xraw.index) == list(meta.index)


# ============================================================
# REPRESENTATIONS
# ============================================================

# This reproduces the pre-correction QC representation:
# standardize each gene across the 409 samples.
Z_before = StandardScaler().fit_transform(
    Xraw.to_numpy(
        dtype=np.float64
    )
)

# Plate-corrected sample matrix was generated in the
# standardized 2000-HVG space, followed by within-plate
# centering. Use it directly.
Z_after = Xcorr.to_numpy(
    dtype=np.float64
)


if not np.isfinite(
    Z_before
).all():
    raise RuntimeError(
        "Non-finite values in pre-correction matrix."
    )

if not np.isfinite(
    Z_after
).all():
    raise RuntimeError(
        "Non-finite values in corrected matrix."
    )


drugs = (
    meta["drug"]
    .astype(str)
    .to_numpy()
)

plates = (
    meta["plate"]
    .astype(str)
    .to_numpy()
)


# ============================================================
# PAIRWISE CROSS-PLATE QC
# ============================================================

def crossplate_pair_stats(Z):

    C = np.corrcoef(
        Z
    )

    ii, jj = np.triu_indices(
        len(Z),
        k=1
    )

    r = C[
        ii,
        jj
    ]

    finite = np.isfinite(
        r
    )

    ii = ii[finite]
    jj = jj[finite]
    r = r[finite]

    same_drug = (
        drugs[ii]
        ==
        drugs[jj]
    )

    cross_plate = (
        plates[ii]
        !=
        plates[jj]
    )

    same_cross = r[
        same_drug
        &
        cross_plate
    ]

    diff_cross = r[
        (~same_drug)
        &
        cross_plate
    ]

    null95 = float(
        np.quantile(
            diff_cross,
            0.95
        )
    )

    fraction_above = float(
        np.mean(
            same_cross
            >
            null95
        )
    )

    return {
        "same": same_cross,
        "null": diff_cross,
        "same_n": len(same_cross),
        "null_n": len(diff_cross),
        "same_median": float(
            np.median(
                same_cross
            )
        ),
        "null_median": float(
            np.median(
                diff_cross
            )
        ),
        "null95": null95,
        "fraction_above": fraction_above,
    }


before = crossplate_pair_stats(
    Z_before
)

after = crossplate_pair_stats(
    Z_after
)


print("\n" + "=" * 80)
print("FIGURE 3 — CROSS-PLATE REPLICATE QC")
print("=" * 80)


def print_stats(
    name,
    s
):
    print(name)
    print(
        f"  same-drug cross-plate n     = "
        f"{s['same_n']}"
    )
    print(
        f"  different-drug cross-plate n= "
        f"{s['null_n']}"
    )
    print(
        f"  same-drug median r          = "
        f"{s['same_median']:+.4f}"
    )
    print(
        f"  null median r               = "
        f"{s['null_median']:+.4f}"
    )
    print(
        f"  null p95                    = "
        f"{s['null95']:+.4f}"
    )
    print(
        f"  same-drug > null p95        = "
        f"{100*s['fraction_above']:.1f}%"
    )


print_stats(
    "BEFORE PLATE CORRECTION",
    before
)

print()

print_stats(
    "AFTER PLATE CORRECTION",
    after
)


# Save numerical record
stats_df = pd.DataFrame([
    {
        "condition":
            "before_plate_correction",

        "same_drug_crossplate_n":
            before["same_n"],

        "different_drug_crossplate_n":
            before["null_n"],

        "same_drug_crossplate_median_r":
            before["same_median"],

        "different_drug_crossplate_median_r":
            before["null_median"],

        "crossplate_null_p95":
            before["null95"],

        "same_drug_fraction_above_null_p95":
            before["fraction_above"],
    },

    {
        "condition":
            "after_plate_correction",

        "same_drug_crossplate_n":
            after["same_n"],

        "different_drug_crossplate_n":
            after["null_n"],

        "same_drug_crossplate_median_r":
            after["same_median"],

        "different_drug_crossplate_median_r":
            after["null_median"],

        "crossplate_null_p95":
            after["null95"],

        "same_drug_fraction_above_null_p95":
            after["fraction_above"],
    },
])


stats_df.to_csv(
    OUT / "figure3_crossplate_stats.csv",
    index=False
)


# ============================================================
# FIGURE 3
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.5),
    sharex=True,
    sharey=True
)


for ax, s, title in [
    (
        axes[0],
        before,
        "Before plate correction"
    ),
    (
        axes[1],
        after,
        "After plate correction"
    ),
]:

    bins = np.linspace(
        -1,
        1,
        70
    )

    ax.hist(
        s["null"],
        bins=bins,
        density=True,
        alpha=0.50,
        label=(
            "Different-drug,\n"
            "cross-plate null"
        )
    )

    ax.hist(
        s["same"],
        bins=bins,
        density=True,
        alpha=0.65,
        label=(
            "Same-drug,\n"
            "cross-plate"
        )
    )

    ax.axvline(
        s["null95"],
        linestyle=":",
        linewidth=2.2,
        label=(
            "Cross-plate null p95\n"
            f"r={s['null95']:.3f}"
        )
    )

    ax.axvline(
        s["same_median"],
        linestyle="--",
        linewidth=2.2,
        label=(
            "Same-drug median\n"
            f"r={s['same_median']:.3f}"
        )
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Pearson correlation"
    )

    ax.text(
        0.04,
        0.96,
        (
            f"same-drug n={s['same_n']}\n"
            f"null n={s['null_n']}\n"
            f"{100*s['fraction_above']:.1f}% "
            f"same-drug pairs > null p95"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.80
        )
    )


axes[0].set_ylabel(
    "Density"
)

axes[1].legend(
    fontsize=8,
    loc="upper right"
)


fig.suptitle(
    "TAHOE HepG2 replicate reproducibility using a cross-plate null",
    y=1.02
)

fig.text(
    0.5,
    -0.01,
    (
        "Cross-plate comparisons prevent same-plate batch similarity "
        "from defining the replicate null."
    ),
    ha="center",
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    FIG3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nsaved:",
    FIG3
)


# ============================================================
# FIGURE 4 — PCA BY PLATE
# ============================================================

print("\n" + "=" * 80)
print("FIGURE 4 — PCA COLOURED BY PLATE")
print("=" * 80)


pca_before = PCA(
    n_components=10,
    random_state=RANDOM_STATE
)

P_before = pca_before.fit_transform(
    Z_before
)


pca_after = PCA(
    n_components=10,
    random_state=RANDOM_STATE
)

P_after = pca_after.fit_transform(
    Z_after
)


# ============================================================
# Plate eta-squared
# ============================================================

def eta_squared(
    values,
    groups
):

    values = np.asarray(
        values,
        dtype=float
    )

    groups = np.asarray(
        groups
    )

    overall = values.mean()

    ss_total = np.sum(
        (
            values
            -
            overall
        ) ** 2
    )

    ss_between = 0.0

    for g in np.unique(
        groups
    ):

        x = values[
            groups == g
        ]

        ss_between += (
            len(x)
            *
            (
                x.mean()
                -
                overall
            ) ** 2
        )

    if ss_total == 0:
        return np.nan

    return float(
        ss_between
        /
        ss_total
    )


eta_rows = []

for condition, P in [
    (
        "before_plate_correction",
        P_before
    ),
    (
        "after_plate_correction",
        P_after
    ),
]:

    for j in range(
        min(
            10,
            P.shape[1]
        )
    ):

        eta_rows.append({
            "condition":
                condition,

            "PC":
                j + 1,

            "plate_eta_squared":
                eta_squared(
                    P[:, j],
                    plates
                ),
        })


eta_df = pd.DataFrame(
    eta_rows
)

eta_df.to_csv(
    OUT / "figure4_plate_pca_eta_squared.csv",
    index=False
)


for condition in [
    "before_plate_correction",
    "after_plate_correction",
]:

    sub = eta_df[
        eta_df["condition"]
        ==
        condition
    ]

    print(
        "\n",
        condition,
        sep=""
    )

    print(
        sub.head(5).to_string(
            index=False
        )
    )


# ============================================================
# DRAW PCA
# ============================================================

plate_levels = sorted(
    pd.unique(
        plates
    )
)

cmap = plt.get_cmap(
    "tab20",
    len(plate_levels)
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


for ax, P, pca, title in [
    (
        axes[0],
        P_before,
        pca_before,
        "Before plate correction"
    ),

    (
        axes[1],
        P_after,
        pca_after,
        "After plate correction"
    ),
]:

    for idx, plate in enumerate(
        plate_levels
    ):

        mask = (
            plates
            ==
            plate
        )

        ax.scatter(
            P[
                mask,
                0
            ],
            P[
                mask,
                1
            ],
            s=38,
            alpha=0.75,
            color=cmap(idx),
            label=plate,
            edgecolors="none"
        )


    eta1 = eta_squared(
        P[:, 0],
        plates
    )

    eta2 = eta_squared(
        P[:, 1],
        plates
    )


    ax.set_xlabel(
        (
            f"PC1 "
            f"({100*pca.explained_variance_ratio_[0]:.1f}% variance)"
        )
    )

    ax.set_ylabel(
        (
            f"PC2 "
            f"({100*pca.explained_variance_ratio_[1]:.1f}% variance)"
        )
    )

    ax.set_title(
        title
    )

    ax.text(
        0.03,
        0.97,
        (
            f"plate eta²\n"
            f"PC1 = {eta1:.3f}\n"
            f"PC2 = {eta2:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.82
        )
    )


# One shared legend
handles, labels = axes[1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Plate",
    loc="center left",
    bbox_to_anchor=(0.995, 0.5),
    fontsize=8,
    title_fontsize=9
)


fig.suptitle(
    "TAHOE HepG2 sample-level transcriptomic PCA reveals plate structure",
    y=1.02
)

plt.tight_layout(
    rect=[
        0,
        0,
        0.88,
        1
    ]
)

plt.savefig(
    FIG4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nsaved:",
    FIG4
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    "Figure 3:",
    FIG3
)

print(
    "Figure 4:",
    FIG4
)

print(
    "Stats:",
    OUT / "figure3_crossplate_stats.csv"
)

print(
    "Stats:",
    OUT / "figure4_plate_pca_eta_squared.csv"
)

