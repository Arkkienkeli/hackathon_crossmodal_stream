#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

A549 = Path(
    "LINCS/scanpy_sensitivity/batch_plate/rankings/"
    "A549_drug_crossmodal_concordance.csv"
)

HEPG2 = Path(
    "OpenScreen/data/hepg2_rankings/"
    "HepG2_drug_crossmodal_concordance.csv"
)

OUT = Path("cross_dataset_common_compounds")
OUT.mkdir(exist_ok=True)


def norm(x):
    return (
        str(x)
        .strip()
        .lower()
    )


a = pd.read_csv(A549)
h = pd.read_csv(HEPG2)

a["drug_key"] = a["drug"].map(norm)
h["drug_key"] = h["drug"].map(norm)

a = a.rename(
    columns={
        "drug": "drug_A549",
        "crossmodal_neighborhood_r":
            "A549_crossmodal_r",
        "morphology_strength":
            "A549_morph_strength",
        "expression_strength":
            "A549_expr_strength",
    }
)

h = h.rename(
    columns={
        "drug": "drug_HepG2",
        "crossmodal_neighborhood_r":
            "HepG2_crossmodal_r",
        "morphology_strength":
            "HepG2_morph_strength",
        "expression_strength":
            "HepG2_expr_strength",
    }
)

keep_a = [
    "drug_key",
    "drug_A549",
    "A549_crossmodal_r",
    "A549_morph_strength",
    "A549_expr_strength",
]

keep_h = [
    "drug_key",
    "drug_HepG2",
    "HepG2_crossmodal_r",
    "HepG2_morph_strength",
    "HepG2_expr_strength",
]

m = a[keep_a].merge(
    h[keep_h],
    on="drug_key",
    how="inner",
)

# canonical displayed drug name
m["drug"] = m[
    "drug_HepG2"
].fillna(
    m["drug_A549"]
)

# Average concordance across the two cell lines
m["mean_crossmodal_r"] = (
    m["A549_crossmodal_r"]
    +
    m["HepG2_crossmodal_r"]
) / 2

# Conservative score:
# how strong is the WORSE of the two?
m["minimum_crossmodal_r"] = np.minimum(
    m["A549_crossmodal_r"],
    m["HepG2_crossmodal_r"],
)

m["cellline_difference"] = np.abs(
    m["A549_crossmodal_r"]
    -
    m["HepG2_crossmodal_r"]
)

# ranks within common set
m["A549_rank"] = (
    m["A549_crossmodal_r"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

m["HepG2_rank"] = (
    m["HepG2_crossmodal_r"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

m["mean_rank"] = (
    m["A549_rank"]
    +
    m["HepG2_rank"]
) / 2


# ==========================================================
# Ranking 1: strongest average agreement
# ==========================================================

avg = m.sort_values(
    "mean_crossmodal_r",
    ascending=False,
)

avg.to_csv(
    OUT /
    "common_compounds_ranked_by_mean_concordance.csv",
    index=False,
)


# ==========================================================
# Ranking 2: consistently strong in BOTH cell lines
# ==========================================================

consistent = m.sort_values(
    "minimum_crossmodal_r",
    ascending=False,
)

consistent.to_csv(
    OUT /
    "common_compounds_ranked_by_minimum_concordance.csv",
    index=False,
)


# ==========================================================
# Ranking 3: biggest cell-line differences
# ==========================================================

different = m.sort_values(
    "cellline_difference",
    ascending=False,
)

different.to_csv(
    OUT /
    "common_compounds_largest_cellline_difference.csv",
    index=False,
)


print("=" * 100)
print("COMMON COMPOUNDS: MORPHOLOGY + GE IN BOTH HEPG2 AND A549")
print("=" * 100)

print("\nNumber of common compounds:", len(m))

print("\nTOP 20 — STRONGEST MEAN CROSS-MODAL CONCORDANCE")
print(
    avg[
        [
            "drug",
            "HepG2_crossmodal_r",
            "A549_crossmodal_r",
            "mean_crossmodal_r",
            "minimum_crossmodal_r",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nTOP 20 — MOST CONSISTENTLY STRONG IN BOTH CELL LINES")
print(
    consistent[
        [
            "drug",
            "HepG2_crossmodal_r",
            "A549_crossmodal_r",
            "minimum_crossmodal_r",
            "mean_crossmodal_r",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nTOP 15 — LARGEST CELL-LINE DIFFERENCE")
print(
    different[
        [
            "drug",
            "HepG2_crossmodal_r",
            "A549_crossmodal_r",
            "cellline_difference",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nSAVED TO:", OUT)
print("DONE")
