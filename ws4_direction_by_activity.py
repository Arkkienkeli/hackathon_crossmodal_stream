#!/usr/bin/env python3
"""
Is the weak global direction-only signal concentrated among strongly
perturbing compounds?

Background
----------
For the 119 matched OpenScreen HepG2 / TAHOE HepG2 drugs:

    notebook Euclidean Mantel             r ~ 0.328
    direction-only Mantel                 r ~ 0.040

The strong Euclidean result contains perturbation magnitude, whereas the
direction-only analysis removes magnitude by projecting each drug's 30-PC
profile onto the unit sphere.

This script asks whether the small global direction-only correspondence is
diluted by weakly perturbing compounds.

A compound is considered highly active only when it has high perturbation
magnitude in BOTH modalities.

For several top-k activity subsets we calculate:

  1. direction-only Mantel r
  2. label-permutation Mantel p within that subset
  3. the distribution of direction-only r from 1000 RANDOM subsets of
     exactly the same size
  4. an empirical p-value asking whether the activity-selected subset is
     more concordant than random subsets of that size

The random-subset comparison is essential because smaller subsets naturally
have wider correlation distributions.

Usage
-----
python ws4_direction_by_activity.py \
    OpenScreen/data/hepg2_morphology_final.parquet \
    OpenScreen/data/hepg2_pseudobulk_2000hvg_shared119.parquet

Optional MoA file:

python ws4_direction_by_activity.py \
    OpenScreen/data/hepg2_morphology_final.parquet \
    OpenScreen/data/hepg2_pseudobulk_2000hvg_shared119.parquet \
    OpenScreen/data/hepg2_drug_aggregation_qc.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

N_PCS = 30

# Permutations for Mantel test inside each selected subset
N_PERM = 999

# Number of size-matched random subsets
N_RANDOM_SUBSETS = 1000

RANDOM_STATE = 0

TOP_K_VALUES = [
    30,
    40,
    50,
    60,
    80,
    100,
]


# ============================================================
# BASIC HELPERS
# ============================================================

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match notebook behaviour:
    z-score each PC across drugs using population SD (ddof=0).
    """
    std = df.std(ddof=0).replace(0, 1)

    return (
        df - df.mean()
    ) / std


def _pearson(a, b):
    """
    Fast Pearson correlation between two 1-D vectors.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a - a.mean()
    b = b - b.mean()

    den = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if den == 0:
        return np.nan

    return float(
        a @ b / den
    )


# ============================================================
# PCA REPRESENTATION
# ============================================================

def embed(X):
    """
    Notebook representation:

        drug profiles
        -> PCA 30
        -> z-score each PC across drugs
    """

    P = PCA(
        n_components=N_PCS,
        random_state=RANDOM_STATE
    ).fit_transform(
        np.asarray(
            X,
            dtype=np.float64
        )
    )

    return zscore(
        pd.DataFrame(P)
    ).to_numpy()


def directions(Z):
    """
    Remove perturbation magnitude.

    Centre the 30-PC drug cloud, then project every drug vector
    onto the unit sphere.

    After this step every drug has norm = 1, so distances depend
    on direction rather than radial perturbation magnitude.
    """

    U = (
        Z - Z.mean(axis=0)
    )

    norm = np.linalg.norm(
        U,
        axis=1,
        keepdims=True
    )

    norm[norm == 0] = 1

    return U / norm


def activity(Z):
    """
    Perturbation magnitude = distance from the centroid in the
    standardized 30-PC representation.
    """

    Zc = (
        Z - Z.mean(axis=0)
    )

    return np.linalg.norm(
        Zc,
        axis=1
    )


# ============================================================
# MANTEL-STYLE TEST
# ============================================================

def mantel(
    D1,
    D2,
    n_perm=N_PERM,
    seed=RANDOM_STATE,
):
    """
    Spearman/Mantel-style correlation between upper triangles.

    Significance is assessed by permuting drug labels in D2.
    """

    rng = np.random.default_rng(
        seed
    )

    n = D1.shape[0]

    iu = np.triu_indices(
        n,
        k=1
    )

    r1 = rankdata(
        D1[iu]
    )

    r2 = rankdata(
        D2[iu]
    )

    obs = _pearson(
        r1,
        r2
    )

    null = np.empty(
        n_perm,
        dtype=np.float64
    )

    for i in range(n_perm):

        p = rng.permutation(n)

        null[i] = _pearson(
            r1,
            rankdata(
                D2[
                    np.ix_(p, p)
                ][iu]
            )
        )

    # Two-sided permutation p-value
    p_two = (
        1
        + np.sum(
            np.abs(null)
            >= abs(obs)
        )
    ) / (
        n_perm + 1
    )

    return (
        float(obs),
        float(p_two),
        null
    )


# ============================================================
# DIRECTION-ONLY STATISTICS
# ============================================================

def direction_r_only(
    Um,
    Ug,
    idx,
):
    """
    Direction-only Mantel r without permutations.

    Used for the 1000 random subset controls.
    """

    idx = np.asarray(idx)

    Dm = squareform(
        pdist(
            Um[idx],
            metric="euclidean"
        )
    )

    Dg = squareform(
        pdist(
            Ug[idx],
            metric="euclidean"
        )
    )

    iu = np.triu_indices(
        len(idx),
        k=1
    )

    return float(
        spearmanr(
            Dm[iu],
            Dg[iu]
        ).statistic
    )


def direction_mantel(
    Um,
    Ug,
    idx,
    n_perm=N_PERM,
    seed=RANDOM_STATE,
):
    """
    Full direction-only Mantel test for a selected subset.
    """

    idx = np.asarray(idx)

    Dm = squareform(
        pdist(
            Um[idx],
            metric="euclidean"
        )
    )

    Dg = squareform(
        pdist(
            Ug[idx],
            metric="euclidean"
        )
    )

    return mantel(
        Dm,
        Dg,
        n_perm=n_perm,
        seed=seed,
    )


# ============================================================
# PER-DRUG DIRECTIONAL CONCORDANCE
# ============================================================

def per_drug_direction(
    Um,
    Ug,
    names,
):
    """
    For each drug, compare its rank ordering of all other drugs
    in morphology direction-space versus expression direction-space.
    """

    Dm = squareform(
        pdist(
            Um,
            metric="euclidean"
        )
    )

    Dg = squareform(
        pdist(
            Ug,
            metric="euclidean"
        )
    )

    rows = []

    n = len(names)

    for i in range(n):

        keep = (
            np.arange(n) != i
        )

        r = spearmanr(
            Dm[i, keep],
            Dg[i, keep]
        ).statistic

        rows.append({
            "drug": names[i],
            "direction_concordance":
                float(r),
        })

    return pd.DataFrame(rows)


# ============================================================
# MULTIPLE-TEST CORRECTION
# ============================================================

def bh_adjust(pvalues):
    """
    Benjamini-Hochberg FDR correction.
    """

    p = np.asarray(
        pvalues,
        dtype=np.float64
    )

    n = len(p)

    order = np.argsort(p)

    ranked = p[order]

    q = (
        ranked
        * n
        / np.arange(1, n + 1)
    )

    # Enforce monotonicity
    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.minimum(
        q,
        1.0
    )

    out = np.empty_like(q)

    out[order] = q

    return out


# ============================================================
# OPTIONAL MOA LOADING
# ============================================================

def load_moa(path):
    """
    Supports:

    1. existing hepg2_drug_aggregation_qc.csv:
           index = drug
           column = moa

    OR

    2. a simple one-column CSV indexed by drug.
    """

    df = pd.read_csv(
        path,
        index_col=0
    )

    df.index = (
        df.index.astype(str)
    )

    if "moa" in df.columns:
        return df["moa"].astype(str)

    if "moa-fine" in df.columns:
        return df["moa-fine"].astype(str)

    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str)

    raise ValueError(
        "Could not identify MoA column. "
        "Expected 'moa', 'moa-fine', "
        "or a one-column CSV."
    )


# ============================================================
# MAIN
# ============================================================

def main(
    m_path,
    g_path,
    moa_path=None,
):

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    M = pd.read_parquet(
        m_path
    )

    G = pd.read_parquet(
        g_path
    )

    M.index = (
        M.index.astype(str)
    )

    G.index = (
        G.index.astype(str)
    )

    shared = sorted(
        set(M.index)
        & set(G.index)
    )

    M = M.loc[shared]
    G = G.loc[shared]

    names = np.asarray(
        shared
    )

    print("=" * 80)
    print("DIRECTIONAL CONCORDANCE BY PERTURBATION STRENGTH")
    print("=" * 80)

    print(
        f"\n{len(names)} matched drugs"
    )

    print(
        f"Morphology dimensions: {M.shape[1]}"
    )

    print(
        f"Expression dimensions: {G.shape[1]}"
    )


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    Zm = embed(M)
    Zg = embed(G)

    Um = directions(Zm)
    Ug = directions(Zg)

    am = activity(Zm)
    ag = activity(Zg)


    # ========================================================
    # ACTIVITY SCORE
    #
    # Key correction:
    #
    # A drug is considered highly active only when it is
    # high in BOTH modalities.
    #
    # rankdata gives high rank to high magnitude.
    # Taking MINIMUM requires both modalities to be high.
    # ========================================================

    n = len(names)

    morph_activity_pct = (
        rankdata(am)
        / n
    )

    gex_activity_pct = (
        rankdata(ag)
        / n
    )

    joint_activity = np.minimum(
        morph_activity_pct,
        gex_activity_pct
    )


    activity_df = pd.DataFrame({
        "drug":
            names,

        "morphology_magnitude":
            am,

        "expression_magnitude":
            ag,

        "morphology_activity_percentile":
            morph_activity_pct,

        "expression_activity_percentile":
            gex_activity_pct,

        "joint_activity_score":
            joint_activity,
    })

    activity_df = (
        activity_df
        .sort_values(
            "joint_activity_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    activity_df[
        "joint_activity_rank"
    ] = (
        np.arange(
            1,
            len(activity_df) + 1
        )
    )

    activity_df.to_csv(
        "direction_activity_ranking.csv",
        index=False
    )


    # --------------------------------------------------------
    # GLOBAL DIRECTION-ONLY RESULT
    # --------------------------------------------------------

    all_idx = np.arange(
        len(names)
    )

    r_all, p_all, _ = (
        direction_mantel(
            Um,
            Ug,
            all_idx
        )
    )

    print("\n" + "=" * 80)
    print("GLOBAL DIRECTION-ONLY RESULT")
    print("=" * 80)

    print(
        f"r = {r_all:+.4f}"
    )

    print(
        f"permutation p = {p_all:.4f}"
    )


    # ========================================================
    # ACTIVITY-RESTRICTED ANALYSIS
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "DIRECTIONAL AGREEMENT AMONG "
        "MOST ACTIVE COMPOUNDS"
    )
    print("=" * 80)

    print(
        "\nActivity criterion:"
    )

    print(
        "minimum of morphology and "
        "expression activity percentiles"
    )

    print(
        "\nFor each k:"
    )

    print(
        "  Mantel p = correspondence "
        "inside selected subset"
    )

    print(
        "  active-vs-random p = whether "
        "selected subset beats random "
        "subsets of same size\n"
    )


    # Order by high activity in BOTH modalities
    order = np.argsort(
        -joint_activity
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rows = []

    k_values = TOP_K_VALUES + [n]


    for k in k_values:

        if k > n:
            continue

        idx = np.sort(
            order[:k]
        )

        # -----------------------------------------------
        # Actual active subset
        # -----------------------------------------------

        r, mantel_p, _ = (
            direction_mantel(
                Um,
                Ug,
                idx,
                n_perm=N_PERM,
                seed=RANDOM_STATE,
            )
        )


        # -----------------------------------------------
        # Size-matched RANDOM subset null
        #
        # No nested Mantel permutations are required.
        # We want the distribution of the statistic itself
        # for random groups of identical size.
        # -----------------------------------------------

        ctrl = np.empty(
            N_RANDOM_SUBSETS,
            dtype=np.float64
        )

        for b in range(
            N_RANDOM_SUBSETS
        ):

            random_idx = np.sort(
                rng.choice(
                    n,
                    size=k,
                    replace=False
                )
            )

            ctrl[b] = (
                direction_r_only(
                    Um,
                    Ug,
                    random_idx
                )
            )


        ctrl_mean = float(
            np.mean(ctrl)
        )

        ctrl_sd = float(
            np.std(ctrl)
        )

        ctrl_p95 = float(
            np.quantile(
                ctrl,
                0.95
            )
        )

        # One-sided:
        # Is active-subset r unusually HIGH?
        active_vs_random_p = (
            1
            + np.sum(ctrl >= r)
        ) / (
            N_RANDOM_SUBSETS + 1
        )

        if ctrl_sd > 0:
            active_vs_random_z = (
                r - ctrl_mean
            ) / ctrl_sd
        else:
            active_vs_random_z = np.nan


        rows.append({
            "top_k_active":
                k,

            "direction_r":
                r,

            "mantel_p":
                mantel_p,

            "random_subset_mean_r":
                ctrl_mean,

            "random_subset_sd":
                ctrl_sd,

            "random_subset_p95":
                ctrl_p95,

            "active_vs_random_z":
                active_vs_random_z,

            "active_vs_random_p":
                active_vs_random_p,
        })


        print(
            f"k={k:>3} | "
            f"r={r:+.4f} "
            f"(Mantel p={mantel_p:.4f}) | "
            f"random={ctrl_mean:+.4f} "
            f"+/- {ctrl_sd:.4f} | "
            f"random p95={ctrl_p95:+.4f} | "
            f"active-vs-random "
            f"p={active_vs_random_p:.4f}"
        )


    tab = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Adjust across tested activity thresholds
    # --------------------------------------------------------

    tab[
        "active_vs_random_q_BH"
    ] = bh_adjust(
        tab[
            "active_vs_random_p"
        ].to_numpy()
    )


    tab.to_csv(
        "direction_by_activity.csv",
        index=False
    )


    print("\n" + "=" * 80)
    print("ACTIVITY-SUBSET SUMMARY")
    print("=" * 80)

    print(
        tab.to_string(
            index=False
        )
    )


    # ========================================================
    # PER-DRUG DIRECTIONAL CONCORDANCE
    # ========================================================

    per = per_drug_direction(
        Um,
        Ug,
        names
    )


    activity_lookup = (
        activity_df
        .set_index("drug")
    )


    per[
        "morphology_activity_percentile"
    ] = (
        per["drug"]
        .map(
            activity_lookup[
                "morphology_activity_percentile"
            ]
        )
    )


    per[
        "expression_activity_percentile"
    ] = (
        per["drug"]
        .map(
            activity_lookup[
                "expression_activity_percentile"
            ]
        )
    )


    per[
        "joint_activity_score"
    ] = (
        per["drug"]
        .map(
            activity_lookup[
                "joint_activity_score"
            ]
        )
    )


    rho = spearmanr(
        per[
            "joint_activity_score"
        ],
        per[
            "direction_concordance"
        ]
    ).statistic


    print("\n" + "=" * 80)
    print(
        "PER-DRUG DIRECTIONAL CONCORDANCE"
    )
    print("=" * 80)

    print(
        "\nDirection concordance vs "
        "joint activity:"
    )

    print(
        f"Spearman rho = {rho:+.4f}"
    )


    per = (
        per
        .sort_values(
            "direction_concordance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    per.to_csv(
        "per_drug_direction_concordance.csv",
        index=False
    )


    print(
        "\nTop 15 drugs by "
        "direction-only concordance:"
    )

    print(
        per.head(15)
        .to_string(index=False)
    )


    print(
        "\nLowest 15 drugs by "
        "direction-only concordance:"
    )

    print(
        per.tail(15)
        .sort_values(
            "direction_concordance"
        )
        .to_string(index=False)
    )


    # ========================================================
    # OPTIONAL MoA ANALYSIS
    # ========================================================

    if moa_path:

        moa = load_moa(
            moa_path
        )

        lab = moa.reindex(
            names
        )

        valid = (
            lab.notna()
        ).to_numpy()

        labels = (
            lab.to_numpy()
        )

        Dm = squareform(
            pdist(
                Um,
                metric="euclidean"
            )
        )

        Dg = squareform(
            pdist(
                Ug,
                metric="euclidean"
            )
        )

        iu = np.triu_indices(
            n,
            k=1
        )

        valid_pair = (
            valid[iu[0]]
            & valid[iu[1]]
        )

        same = (
            labels[iu[0]]
            == labels[iu[1]]
        ) & valid_pair


        print("\n" + "=" * 80)
        print(
            "OPTIONAL MoA DIRECTION-SPACE QC"
        )
        print("=" * 80)

        print(
            "Annotated drugs:",
            valid.sum()
        )

        print(
            "Same-MoA pairs:",
            same.sum()
        )

        print(
            "Valid different-MoA pairs:",
            (
                valid_pair & ~same
            ).sum()
        )


        other = (
            valid_pair
            & ~same
        )


        print(
            "\nMorphology direction distance:"
        )

        print(
            "same-MoA:",
            Dm[iu][same].mean()
        )

        print(
            "different-MoA:",
            Dm[iu][other].mean()
        )


        print(
            "\nExpression direction distance:"
        )

        print(
            "same-MoA:",
            Dg[iu][same].mean()
        )

        print(
            "different-MoA:",
            Dg[iu][other].mean()
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n" + "=" * 80)
    print("READING")
    print("=" * 80)


    tested = tab[
        tab[
            "top_k_active"
        ] < n
    ].copy()


    significant = tested[
        (
            tested[
                "active_vs_random_q_BH"
            ] < 0.05
        )
        &
        (
            tested[
                "direction_r"
            ]
            >
            tested[
                "random_subset_p95"
            ]
        )
    ]


    if len(significant) > 0:

        best = significant.loc[
            significant[
                "direction_r"
            ].idxmax()
        ]

        print(
            "\nEvidence that directional "
            "cross-modal correspondence is "
            "concentrated among strongly "
            "perturbing compounds."
        )

        print(
            f"\nBest supported subset: "
            f"top {int(best['top_k_active'])} "
            f"active-in-both drugs"
        )

        print(
            f"direction r = "
            f"{best['direction_r']:+.4f}"
        )

        print(
            f"random subset mean = "
            f"{best['random_subset_mean_r']:+.4f}"
        )

        print(
            f"random subset p95 = "
            f"{best['random_subset_p95']:+.4f}"
        )

        print(
            f"active-vs-random p = "
            f"{best['active_vs_random_p']:.4f}"
        )

        print(
            f"BH q = "
            f"{best['active_vs_random_q_BH']:.4f}"
        )

        print(
            "\nInterpretation: the global "
            "direction-only association is small, "
            "but compounds producing sufficiently "
            "strong perturbations in BOTH modalities "
            "show greater directional correspondence "
            "than random subsets of the same size."
        )

    else:

        print(
            "\nNo activity-selected subset "
            "showed direction concordance "
            "significantly above size-matched "
            "random subsets after correction."
        )

        print(
            "\nInterpretation: directional "
            "cross-modal agreement remains "
            "uniformly weak. The strong Euclidean "
            "association is therefore primarily "
            "an agreement in perturbation magnitude."
        )


    # ========================================================
    # SAVED FILES
    # ========================================================

    print("\n" + "=" * 80)
    print("SAVED FILES")
    print("=" * 80)

    print(
        "direction_by_activity.csv"
    )

    print(
        "direction_activity_ranking.csv"
    )

    print(
        "per_drug_direction_concordance.csv"
    )


    return (
        tab,
        per,
        activity_df,
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise SystemExit(__doc__)

    main(
        *sys.argv[1:4]
    )

