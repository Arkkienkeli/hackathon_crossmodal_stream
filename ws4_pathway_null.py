#!/usr/bin/env python3

"""
Full-chain drug-label permutation calibration for exploratory
OpenScreen morphology <-> TAHOE pathway analysis.

The real and shuffled passes use exactly the same pipeline:

    direction-normalize morphology and expression
        ->
    636 x 2000 Spearman association matrix
        ->
    select TOP 25 morphology features
        ->
    rank all 2000 genes
        ->
    preranked GSEA

Under each shuffle the top 25 features are RE-SELECTED.

This preserves:
    gene-gene covariance
    HVG universe
    morphology covariance
    pathway redundancy
    feature-selection bias

and destroys only:
    drug-level morphology <-> expression correspondence
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import rankdata

import gseapy as gp


# ============================================================
# MUST BE IDENTICAL FOR OBSERVED AND NULL
# ============================================================

N_PERM = 1000

MIN_SIZE = 15
MAX_SIZE = 500

THREADS = 4
SEED = 0

N_FEATURES = 25
N_SHUFFLES = 20

FDR_THRESHOLDS = (
    0.05,
    0.10,
    0.25
)


# ============================================================
# PREPROCESSING
# ============================================================

def zscore_cols(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    mu = X.mean(
        axis=0,
        keepdims=True
    )

    sd = X.std(
        axis=0,
        keepdims=True
    )

    sd[
        sd == 0
    ] = 1.0

    return (
        X - mu
    ) / sd


def unit_rows(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    norm = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    norm[
        norm == 0
    ] = 1.0

    return X / norm


def normalized_rank_columns(X):
    """
    Convert each column to centered ranks and L2-normalize.

    Then:
        A.T @ B
    is the full Spearman correlation matrix.
    """

    R = rankdata(
        X,
        axis=0,
        method="average"
    ).astype(
        np.float64
    )

    R -= R.mean(
        axis=0,
        keepdims=True
    )

    norm = np.linalg.norm(
        R,
        axis=0,
        keepdims=True
    )

    norm[
        norm == 0
    ] = np.nan

    return R / norm


def canonical_leading_edge(x):

    if pd.isna(x):
        return ""

    genes = sorted({
        g.strip()
        for g in str(x).split(";")
        if g.strip()
    })

    return ";".join(
        genes
    )


# ============================================================
# GSEA
# ============================================================

def run_gsea(
    gene_names,
    rho,
    gene_sets
):

    rnk = pd.DataFrame({
        "gene": gene_names,
        "rho": rho
    })

    rnk = (
        rnk
        .dropna()
        .sort_values(
            "rho",
            ascending=False
        )
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        pre = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            permutation_num=N_PERM,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            seed=SEED,
            threads=THREADS,
            outdir=None,
            verbose=False
        )

    return pre.res2d.copy()


# ============================================================
# ONE COMPLETE PASS
# ============================================================

def one_pass(
    MR,
    GR,
    morph_names,
    gene_names,
    gene_sets,
    expression_order,
    pass_label
):

    # Drug-label permutation is applied ONLY
    # between the two modality matrices.

    GRp = GR[
        expression_order,
        :
    ]

    # Full 636 x 2000 direction-normalized
    # Spearman association matrix.
    R = (
        MR.T
        @
        GRp
    )

    strength = np.nanmax(
        np.abs(R),
        axis=1
    )

    top_feature_idx = np.argsort(
        strength
    )[
        ::-1
    ][
        :N_FEATURES
    ]

    selected = [
        morph_names[i]
        for i in top_feature_idx
    ]

    all_results = []

    for rank_i, fi in enumerate(
        top_feature_idx,
        start=1
    ):

        feature = morph_names[
            fi
        ]

        rho = R[
            fi,
            :
        ]

        res = run_gsea(
            gene_names,
            rho,
            gene_sets
        )

        if len(res) == 0:
            continue

        res[
            "morphology_feature"
        ] = feature

        res[
            "feature_selection_rank"
        ] = rank_i

        res[
            "feature_max_abs_gene_rho"
        ] = strength[
            fi
        ]

        all_results.append(
            res
        )

    if not all_results:

        raise RuntimeError(
            f"No GSEA output for {pass_label}"
        )

    allres = pd.concat(
        all_results,
        ignore_index=True
    )

    q = pd.to_numeric(
        allres[
            "FDR q-val"
        ],
        errors="coerce"
    )

    summary = {
        "label":
            pass_label,

        "hits_fdr_005":
            int(
                (
                    q < 0.05
                ).sum()
            ),

        "hits_fdr_010":
            int(
                (
                    q < 0.10
                ).sum()
            ),

        "hits_fdr_025":
            int(
                (
                    q < 0.25
                ).sum()
            ),
    }

    nes = pd.to_numeric(
        allres[
            "NES"
        ],
        errors="coerce"
    )

    summary[
        "max_abs_NES"
    ] = float(
        np.nanmax(
            np.abs(nes)
        )
    )

    hit025 = allres[
        q < 0.25
    ].copy()

    if len(hit025):

        hit025[
            "canonical_le"
        ] = hit025[
            "Lead_genes"
        ].apply(
            canonical_leading_edge
        )

        summary[
            "unique_leading_edges_fdr025"
        ] = int(
            hit025[
                "canonical_le"
            ]
            .replace(
                "",
                np.nan
            )
            .nunique()
        )

    else:

        summary[
            "unique_leading_edges_fdr025"
        ] = 0

    print(
        f"{pass_label}: "
        f"FDR<.05={summary['hits_fdr_005']}  "
        f"FDR<.10={summary['hits_fdr_010']}  "
        f"FDR<.25={summary['hits_fdr_025']}  "
        f"uniqueLE={summary['unique_leading_edges_fdr025']}  "
        f"max|NES|={summary['max_abs_NES']:.3f}",
        flush=True
    )

    return (
        summary,
        selected,
        allres
    )


# ============================================================
# EMPIRICAL CALIBRATION
# ============================================================

def empirical_p(
    observed,
    null_values
):

    null_values = np.asarray(
        null_values,
        dtype=float
    )

    return (
        1
        +
        np.sum(
            null_values
            >=
            observed
        )
    ) / (
        len(
            null_values
        )
        +
        1
    )


def main(
    morph_path,
    gex_path,
    gene_sets_json,
    manifest_csv
):

    M = pd.read_parquet(
        morph_path
    )

    G = pd.read_parquet(
        gex_path
    )

    M.index = M.index.astype(str)
    G.index = G.index.astype(str)

    shared = sorted(
        set(
            M.index
        )
        &
        set(
            G.index
        )
    )

    M = M.loc[
        shared
    ]

    G = G.loc[
        shared
    ]

    morph_names = np.asarray(
        M.columns.astype(str)
    )

    gene_names = np.asarray(
        G.columns.astype(str)
    )

    print(
        f"{len(shared)} drugs | "
        f"{len(morph_names)} morphology features | "
        f"{len(gene_names)} genes"
    )

    print(
        f"N_FEATURES={N_FEATURES} | "
        f"N_SHUFFLES={N_SHUFFLES} | "
        f"MIN_SIZE={MIN_SIZE} | "
        f"N_PERM={N_PERM}"
    )

    # ========================================================
    # Direction-normalization exactly as earlier analysis
    # ========================================================

    Mdir = unit_rows(
        zscore_cols(
            M.to_numpy(
                np.float64
            )
        )
    )

    Gdir = unit_rows(
        zscore_cols(
            G.to_numpy(
                np.float64
            )
        )
    )

    MR = normalized_rank_columns(
        Mdir
    )

    GR = normalized_rank_columns(
        Gdir
    )

    with open(
        gene_sets_json
    ) as fh:

        gene_sets = json.load(
            fh
        )

    print(
        "cached gene sets:",
        len(gene_sets)
    )

    # Existing observed feature list only for
    # sanity-checking our reconstruction.
    manifest = pd.read_csv(
        manifest_csv
    )

    old_features = (
        manifest[
            "morphology_feature"
        ]
        .astype(str)
        .tolist()
    )

    # ========================================================
    # OBSERVED
    # ========================================================

    print(
        "\nOBSERVED"
    )

    obs_summary, obs_features, obs_results = (
        one_pass(
            MR,
            GR,
            morph_names,
            gene_names,
            gene_sets,
            np.arange(
                len(shared)
            ),
            "observed"
        )
    )

    overlap = len(
        set(
            obs_features
        )
        &
        set(
            old_features
        )
    )

    print(
        "Observed top-25 overlap with "
        "previous manifest:",
        f"{overlap}/25"
    )

    obs_results.to_csv(
        "pathway_observed_min15.csv",
        index=False
    )

    # ========================================================
    # SHUFFLES
    # ========================================================

    print(
        "\nDRUG-LABEL SHUFFLES"
    )

    rng = np.random.default_rng(
        SEED
    )

    summaries = [
        obs_summary
    ]

    feature_rows = []

    for rank_i, f in enumerate(
        obs_features,
        start=1
    ):

        feature_rows.append({
            "pass":
                "observed",
            "rank":
                rank_i,
            "feature":
                f
        })

    for i in range(
        N_SHUFFLES
    ):

        order = rng.permutation(
            len(shared)
        )

        summary, selected, _ = one_pass(
            MR,
            GR,
            morph_names,
            gene_names,
            gene_sets,
            order,
            f"shuffle_{i+1:02d}"
        )

        summaries.append(
            summary
        )

        for rank_i, f in enumerate(
            selected,
            start=1
        ):

            feature_rows.append({
                "pass":
                    f"shuffle_{i+1:02d}",
                "rank":
                    rank_i,
                "feature":
                    f
            })

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        "pathway_null_passes.csv",
        index=False
    )

    pd.DataFrame(
        feature_rows
    ).to_csv(
        "pathway_null_selected_features.csv",
        index=False
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "EMPIRICAL CALIBRATION"
    )

    print(
        "=" * 72
    )

    null_df = summary_df[
        summary_df[
            "label"
        ]
        !=
        "observed"
    ]

    metrics = [
        "hits_fdr_005",
        "hits_fdr_010",
        "hits_fdr_025",
        "unique_leading_edges_fdr025",
        "max_abs_NES",
    ]

    rows = []

    for metric in metrics:

        observed = float(
            obs_summary[
                metric
            ]
        )

        vals = null_df[
            metric
        ].to_numpy(
            dtype=float
        )

        p = empirical_p(
            observed,
            vals
        )

        row = {
            "metric":
                metric,
            "observed":
                observed,
            "null_mean":
                float(
                    vals.mean()
                ),
            "null_sd":
                float(
                    vals.std(
                        ddof=1
                    )
                ),
            "null_min":
                float(
                    vals.min()
                ),
            "null_max":
                float(
                    vals.max()
                ),
            "empirical_p":
                float(
                    p
                )
        }

        rows.append(
            row
        )

        print(
            f"{metric:32s} "
            f"obs={observed:8.3f}  "
            f"null={vals.mean():8.3f}"
            f" ± {vals.std(ddof=1):7.3f}  "
            f"max={vals.max():8.3f}  "
            f"emp.p={p:.4f}"
        )

    pd.DataFrame(
        rows
    ).to_csv(
        "pathway_null_calibration.csv",
        index=False
    )

    print(
        "\nNOTE:"
    )

    print(
        "With 20 shuffles the smallest attainable "
        "empirical p-value is 1/21 = 0.0476. "
        "Treat this as calibration, not a precise p-value."
    )

    print(
        "\nWROTE:"
    )

    print(
        "  pathway_observed_min15.csv"
    )

    print(
        "  pathway_null_passes.csv"
    )

    print(
        "  pathway_null_selected_features.csv"
    )

    print(
        "  pathway_null_calibration.csv"
    )


if __name__ == "__main__":

    if len(sys.argv) != 5:

        raise SystemExit(
            "usage: ws4_pathway_null.py "
            "M.parquet G.parquet "
            "gene_sets.json manifest.csv"
        )

    main(
        *sys.argv[1:5]
    )
