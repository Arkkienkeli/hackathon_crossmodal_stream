#!/usr/bin/env python3

"""
Exploratory pathway-level mapping between OpenScreen morphology
and TAHOE transcriptomics.

IMPORTANT
---------
No cells/wells are paired between modalities.

The unit connecting morphology and expression is DRUG IDENTITY.

For each morphology feature:
    119 matched drugs
        ↓
    rank 2,000 genes by direction-normalized
    morphology↔expression Spearman rho
        ↓
    preranked GSEA
        ↓
    morphology feature ↔ transcriptional program

The output is exploratory association, NOT causality.

Uses:
    MSigDB_Hallmark_2020
    Reactome_2022

Also identifies drugs that most strongly support or oppose
selected morphology↔pathway relationships.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import gseapy as gp


# ============================================================
# PATHS
# ============================================================

BASE = Path("OpenScreen/feature_gene_exploratory")

MANIFEST = BASE / "pathway_ranking_manifest.csv"

MORPH = Path(
    "OpenScreen/data/hepg2_morphology_final.parquet"
)

GEX = Path(
    "hepg2_platecorrected_drug_2000hvg.parquet"
)

OUT = Path(
    "OpenScreen/pathway_exploratory"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

ORGANISM = "Human"

REQUESTED_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "Reactome_2022",
]

N_PERM = 1000

MIN_SIZE = 5
MAX_SIZE = 500

THREADS = 4
SEED = 0

TOP_PATHWAYS_PER_FEATURE = 10
TOP_GLOBAL_PAIRS = 50
TOP_DRIVERS = 8


# ============================================================
# HELPERS
# ============================================================

def bh_fdr(p):
    p = np.asarray(
        p,
        dtype=float
    )

    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    q = (
        ranked
        *
        n
        /
        np.arange(
            1,
            n + 1
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(
        q,
        0,
        1
    )

    out = np.empty(
        n,
        dtype=float
    )

    out[order] = q

    return out


def zscore(x):
    x = np.asarray(
        x,
        dtype=float
    )

    sd = np.std(
        x,
        ddof=0
    )

    if sd == 0:
        return (
            x - np.mean(x)
        )

    return (
        x - np.mean(x)
    ) / sd


def safe_gene_list(x):
    if pd.isna(x):
        return []

    return [
        g.strip()
        for g in str(x).split(";")
        if g.strip()
    ]


# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 80)
print("PATHWAY-LEVEL EXPLORATORY ANALYSIS")
print("=" * 80)

for path in [
    MANIFEST,
    MORPH,
    GEX,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


manifest = pd.read_csv(
    MANIFEST
)


print(
    "Morphology rankings:",
    len(manifest)
)

print(
    manifest[
        [
            "rank",
            "morphology_feature",
            "ranking_file"
        ]
    ]
    .head()
    .to_string(index=False)
)


# ============================================================
# FETCH GENE SETS
# ============================================================

print("\n" + "=" * 80)
print("FETCH GENE-SET LIBRARIES")
print("=" * 80)


cache_file = (
    OUT
    /
    "combined_gene_sets.json"
)


if cache_file.exists():

    print(
        "Using cached gene sets:",
        cache_file
    )

    with open(
        cache_file,
        "r"
    ) as handle:

        combined_gene_sets = (
            json.load(handle)
        )

else:

    print(
        "Checking available Enrichr libraries..."
    )

    available = (
        gp.get_library_name(
            organism=ORGANISM
        )
    )


    print(
        "Number of available libraries:",
        len(available)
    )


    combined_gene_sets = {}


    for requested in REQUESTED_LIBRARIES:

        library = requested


        if library not in available:

            print(
                f"{requested} not found exactly."
            )

            if "Reactome" in requested:

                candidates = [
                    x
                    for x in available
                    if "Reactome" in x
                ]

            elif "Hallmark" in requested:

                candidates = [
                    x
                    for x in available
                    if "Hallmark" in x
                ]

            else:

                candidates = []


            if not candidates:

                raise RuntimeError(
                    f"No suitable replacement for "
                    f"{requested}"
                )


            library = sorted(
                candidates
            )[-1]


            print(
                "Using:",
                library
            )


        print(
            "Downloading:",
            library
        )


        gene_sets = gp.get_library(
            name=library,
            organism=ORGANISM
        )


        print(
            library,
            "pathways:",
            len(gene_sets)
        )


        for term, genes in gene_sets.items():

            key = (
                f"{library} :: {term}"
            )

            combined_gene_sets[
                key
            ] = list(
                genes
            )


    with open(
        cache_file,
        "w"
    ) as handle:

        json.dump(
            combined_gene_sets,
            handle
        )


    print(
        "Cached:",
        cache_file
    )


print(
    "Total combined pathways:",
    len(combined_gene_sets)
)


# ============================================================
# RUN PRERANKED GSEA
# ============================================================

print("\n" + "=" * 80)
print("RUN PRERANKED GSEA")
print("=" * 80)


all_results = []


for row in manifest.itertuples(
    index=False
):

    feature_rank = int(
        row.rank
    )

    feature = str(
        row.morphology_feature
    )

    ranking_path = (
        BASE
        /
        str(
            row.ranking_file
        )
    )


    print(
        f"\n[{feature_rank:02d}/"
        f"{len(manifest):02d}]"
    )

    print(
        feature
    )


    ranking = pd.read_csv(
        ranking_path,
        sep="\t"
    )


    ranking = ranking[
        [
            "gene",
            "rho"
        ]
    ].copy()


    ranking = ranking.dropna()


    ranking["gene"] = (
        ranking["gene"]
        .astype(str)
        .str.strip()
    )


    # Remove obvious non-symbol empty entries.
    ranking = ranking[
        ranking["gene"]
        !=
        ""
    ]


    # If duplicated gene IDs somehow exist,
    # retain the strongest absolute association.
    ranking["abs_rho"] = np.abs(
        ranking["rho"]
    )


    ranking = (
        ranking
        .sort_values(
            "abs_rho",
            ascending=False
        )
        .drop_duplicates(
            "gene"
        )
        .sort_values(
            "rho",
            ascending=False
        )
    )


    print(
        "  ranked genes:",
        len(ranking)
    )


    rnk = ranking[
        [
            "gene",
            "rho"
        ]
    ]


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        pre = gp.prerank(
            rnk=rnk,
            gene_sets=combined_gene_sets,
            permutation_num=N_PERM,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            seed=SEED,
            threads=THREADS,
            outdir=None,
            verbose=False
        )


    result = pre.res2d.copy()


    if len(result) == 0:

        print(
            "  WARNING: no pathways tested"
        )

        continue


    result[
        "morphology_feature"
    ] = feature


    result[
        "morphology_feature_rank"
    ] = feature_rank


    result[
        "n_ranked_genes"
    ] = len(ranking)


    # Library name is already prefixed to Term.
    result[
        "library"
    ] = (
        result["Term"]
        .astype(str)
        .str.split(
            " :: ",
            n=1
        )
        .str[0]
    )


    result[
        "pathway"
    ] = (
        result["Term"]
        .astype(str)
        .str.split(
            " :: ",
            n=1
        )
        .str[-1]
    )


    all_results.append(
        result
    )


    top = (
        result
        .sort_values(
            "FDR q-val"
        )
        .head(5)
    )


    print(
        top[
            [
                "pathway",
                "NES",
                "NOM p-val",
                "FDR q-val"
            ]
        ]
        .to_string(
            index=False
        )
    )


if not all_results:

    raise RuntimeError(
        "No GSEA results were produced."
    )


results = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# GLOBAL MULTIPLE-TESTING VIEW
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL MULTIPLE-TESTING SUMMARY")
print("=" * 80)


# permutation p=0 means smaller than resolution;
# do not literally treat it as zero for BH.
p_floor = (
    1.0
    /
    (
        N_PERM + 1
    )
)


nominal_for_bh = (
    pd.to_numeric(
        results[
            "NOM p-val"
        ],
        errors="coerce"
    )
    .fillna(1.0)
    .to_numpy()
)


nominal_for_bh[
    nominal_for_bh
    <
    p_floor
] = p_floor


results[
    "global_bh_q"
] = bh_fdr(
    nominal_for_bh
)


results[
    "abs_NES"
] = np.abs(
    pd.to_numeric(
        results["NES"],
        errors="coerce"
    )
)


results.to_parquet(
    OUT
    /
    "all_morphology_pathway_gsea.parquet",
    index=False
)


results.to_csv(
    OUT
    /
    "all_morphology_pathway_gsea.csv",
    index=False
)


print(
    "Total morphology-feature × pathway tests:",
    len(results)
)


print(
    "Within-feature GSEA FDR <0.25:",
    int(
        (
            pd.to_numeric(
                results[
                    "FDR q-val"
                ],
                errors="coerce"
            )
            <
            0.25
        ).sum()
    )
)


print(
    "Within-feature GSEA FDR <0.05:",
    int(
        (
            pd.to_numeric(
                results[
                    "FDR q-val"
                ],
                errors="coerce"
            )
            <
            0.05
        ).sum()
    )
)


print(
    "Global BH q <0.05:",
    int(
        (
            results[
                "global_bh_q"
            ]
            <
            0.05
        ).sum()
    )
)


# ============================================================
# TOP RESULTS
# ============================================================

top_results = (
    results
    .sort_values(
        [
            "global_bh_q",
            "FDR q-val",
            "abs_NES"
        ],
        ascending=[
            True,
            True,
            False
        ]
    )
    .head(
        TOP_GLOBAL_PAIRS
    )
)


top_results.to_csv(
    OUT
    /
    "top50_morphology_pathway_associations.csv",
    index=False
)


print("\nTop 30 morphology ↔ pathway results:")


display_cols = [
    "morphology_feature",
    "library",
    "pathway",
    "NES",
    "NOM p-val",
    "FDR q-val",
    "global_bh_q",
    "Lead_genes",
]


print(
    top_results[
        display_cols
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# REPEATED PATHWAYS
# ============================================================

print("\n" + "=" * 80)
print("PATHWAYS REPEATED ACROSS MORPHOLOGY FEATURES")
print("=" * 80)


repeated = (
    results
    .assign(
        exploratory_hit=
        (
            pd.to_numeric(
                results[
                    "FDR q-val"
                ],
                errors="coerce"
            )
            <
            0.25
        )
    )
    .groupby(
        [
            "library",
            "pathway"
        ],
        observed=True
    )
    .agg(
        n_morphology_features=(
            "morphology_feature",
            "nunique"
        ),

        n_exploratory_hits=(
            "exploratory_hit",
            "sum"
        ),

        max_abs_NES=(
            "abs_NES",
            "max"
        ),

        min_global_bh_q=(
            "global_bh_q",
            "min"
        ),

        min_within_feature_FDR=(
            "FDR q-val",
            "min"
        ),
    )
    .reset_index()
)


repeated = repeated.sort_values(
    [
        "n_exploratory_hits",
        "max_abs_NES"
    ],
    ascending=[
        False,
        False
    ]
)


repeated.to_csv(
    OUT
    /
    "pathway_recurrence_across_morphology_features.csv",
    index=False
)


print(
    repeated
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# LOAD DRUG-LEVEL MATRICES
# ============================================================

print("\n" + "=" * 80)
print("MAP TOP PATHWAY ASSOCIATIONS BACK TO DRUGS")
print("=" * 80)


M = pd.read_parquet(
    MORPH
)

G = pd.read_parquet(
    GEX
)


M.index = M.index.astype(str)
G.index = G.index.astype(str)


shared = sorted(
    set(M.index)
    &
    set(G.index)
)


M = M.loc[
    shared
]

G = G.loc[
    shared
]


print(
    "Matched drugs:",
    len(shared)
)


# z-score each expression gene across drugs
Gz = (
    G
    -
    G.mean(axis=0)
) / (
    G.std(
        axis=0,
        ddof=0
    )
    .replace(
        0,
        1
    )
)


drug_rows = []


# use top 30 pairs for driver analysis
for row in (
    top_results
    .head(30)
    .itertuples(
        index=False
    )
):

    feature = str(
        row.morphology_feature
    )


    if feature not in M.columns:
        continue


    lead_genes = safe_gene_list(
        row.Lead_genes
    )


    lead_genes = [
        g
        for g in lead_genes
        if g in Gz.columns
    ]


    if len(
        lead_genes
    ) < 3:

        continue


    pathway_score = (
        Gz[
            lead_genes
        ]
        .mean(
            axis=1
        )
        .to_numpy()
    )


    NES = float(
        row.NES
    )


    # Orient pathway score so positive means
    # movement in the morphology-associated direction.
    if NES < 0:

        pathway_score = (
            -pathway_score
        )


    morph_value = (
        M[
            feature
        ]
        .to_numpy(
            dtype=float
        )
    )


    morph_z = zscore(
        morph_value
    )

    path_z = zscore(
        pathway_score
    )


    rho = spearmanr(
        morph_value,
        pathway_score
    ).statistic


    contribution = (
        morph_z
        *
        path_z
    )


    order_support = np.argsort(
        contribution
    )[::-1]


    order_oppose = np.argsort(
        contribution
    )


    for i in order_support[
        :TOP_DRIVERS
    ]:

        drug_rows.append({
            "morphology_feature":
                feature,

            "pathway":
                row.pathway,

            "library":
                row.library,

            "NES":
                NES,

            "pathway_feature_rho":
                rho,

            "role":
                "supports",

            "drug":
                shared[i],

            "morphology_z":
                morph_z[i],

            "pathway_z":
                path_z[i],

            "contribution":
                contribution[i],

            "n_leading_genes":
                len(
                    lead_genes
                ),
        })


    for i in order_oppose[
        :TOP_DRIVERS
    ]:

        drug_rows.append({
            "morphology_feature":
                feature,

            "pathway":
                row.pathway,

            "library":
                row.library,

            "NES":
                NES,

            "pathway_feature_rho":
                rho,

            "role":
                "opposes",

            "drug":
                shared[i],

            "morphology_z":
                morph_z[i],

            "pathway_z":
                path_z[i],

            "contribution":
                contribution[i],

            "n_leading_genes":
                len(
                    lead_genes
                ),
        })


drivers = pd.DataFrame(
    drug_rows
)


drivers.to_csv(
    OUT
    /
    "top_pathway_drug_drivers.csv",
    index=False
)


print(
    "Drug-driver rows:",
    len(drivers)
)


# ============================================================
# FIGURE 1:
# MORPHOLOGY FEATURE × PATHWAY NES HEATMAP
# ============================================================

print("\nCreating heatmap...")


# Pick pathways showing strongest evidence
selected_pathways = (
    top_results[
        [
            "library",
            "pathway"
        ]
    ]
    .drop_duplicates()
    .head(20)
)


selected_features = (
    top_results[
        "morphology_feature"
    ]
    .drop_duplicates()
    .head(20)
    .tolist()
)


pathway_labels = [
    f"{x.library} :: {x.pathway}"
    for x in selected_pathways.itertuples(
        index=False
    )
]


heat = np.full(
    (
        len(
            selected_features
        ),
        len(
            pathway_labels
        )
    ),
    np.nan
)


for i, feature in enumerate(
    selected_features
):

    sub = results[
        results[
            "morphology_feature"
        ]
        ==
        feature
    ]


    lookup = {
        (
            str(r.library)
            +
            " :: "
            +
            str(r.pathway)
        ):
        float(
            r.NES
        )

        for r in sub.itertuples(
            index=False
        )
    }


    for j, pathway in enumerate(
        pathway_labels
    ):

        if pathway in lookup:

            heat[
                i,
                j
            ] = lookup[
                pathway
            ]


plt.figure(
    figsize=(
        16,
        11
    )
)


im = plt.imshow(
    heat,
    aspect="auto",
    interpolation="nearest"
)


plt.colorbar(
    im,
    label="GSEA normalized enrichment score (NES)"
)


plt.xticks(
    np.arange(
        len(
            pathway_labels
        )
    ),
    [
        x.replace(
            "MSigDB_Hallmark_2020 :: ",
            "Hallmark: "
        ).replace(
            "Reactome_2022 :: ",
            "Reactome: "
        )
        for x
        in pathway_labels
    ],
    rotation=90,
    fontsize=7
)


plt.yticks(
    np.arange(
        len(
            selected_features
        )
    ),
    selected_features,
    fontsize=7
)


plt.xlabel(
    "Transcriptional pathways"
)

plt.ylabel(
    "Morphology features"
)

plt.title(
    "Exploratory morphology ↔ transcriptional-program map"
)


plt.tight_layout()


plt.savefig(
    OUT
    /
    "01_morphology_pathway_NES_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2:
# TOP PATHWAY ASSOCIATIONS
# ============================================================

plot_df = (
    top_results
    .head(25)
    .copy()
)


plot_df[
    "label"
] = (
    plot_df[
        "pathway"
    ].astype(str)
    .str.slice(
        0,
        55
    )
    +
    "\n"
    +
    plot_df[
        "morphology_feature"
    ].astype(str)
    .str.slice(
        0,
        40
    )
)


plt.figure(
    figsize=(
        11,
        11
    )
)


y = np.arange(
    len(
        plot_df
    )
)


plt.barh(
    y,
    pd.to_numeric(
        plot_df[
            "NES"
        ],
        errors="coerce"
    )
)


plt.yticks(
    y,
    plot_df[
        "label"
    ],
    fontsize=7
)


plt.gca().invert_yaxis()


plt.axvline(
    0,
    linewidth=1
)


plt.xlabel(
    "Normalized enrichment score (NES)"
)


plt.title(
    "Top exploratory morphology ↔ pathway associations"
)


plt.tight_layout()


plt.savefig(
    OUT
    /
    "02_top_morphology_pathway_associations.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "n_morphology_features_tested":
            results[
                "morphology_feature"
            ].nunique(),

        "n_gene_sets":
            results[
                [
                    "library",
                    "pathway"
                ]
            ]
            .drop_duplicates()
            .shape[0],

        "n_total_feature_pathway_tests":
            len(
                results
            ),

        "n_within_feature_fdr_lt_025":
            int(
                (
                    pd.to_numeric(
                        results[
                            "FDR q-val"
                        ],
                        errors="coerce"
                    )
                    <
                    0.25
                ).sum()
            ),

        "n_within_feature_fdr_lt_005":
            int(
                (
                    pd.to_numeric(
                        results[
                            "FDR q-val"
                        ],
                        errors="coerce"
                    )
                    <
                    0.05
                ).sum()
            ),

        "n_global_bh_lt_005":
            int(
                (
                    results[
                        "global_bh_q"
                    ]
                    <
                    0.05
                ).sum()
            ),

        "max_abs_NES":
            results[
                "abs_NES"
            ].max(),
    }
])


summary.to_csv(
    OUT
    /
    "pathway_analysis_summary.csv",
    index=False
)


print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)


print(
    summary.to_string(
        index=False
    )
)


print("\nCreated files:")


for f in sorted(
    OUT.iterdir()
):

    print(
        " ",
        f
    )


print("\nINTERPRETATION:")
print(
    "Pathway associations are exploratory relationships across matched "
    "drug perturbations. They do not imply paired-cell measurements "
    "or causal gene-to-morphology relationships."
)

