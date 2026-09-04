#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata, t
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

MORPH_FILE = Path(
    "OpenScreen/data/hepg2_morphology_final.parquet"
)

DATA_DIR = Path("OpenScreen/data")

OUT = Path(
    "OpenScreen/data/hepg2_rankings"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

N_PCS = 30

# Frozen final HepG2 Task-2 result
TARGET_R = 0.369703
MAX_R_DIFFERENCE = 0.02


# ============================================================
# HELPERS
# ============================================================

def clean_name(x):
    return str(x).strip().lower()


def zscore_cols(X):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    mu = X.mean(
        axis=0,
        keepdims=True,
    )

    sd = X.std(
        axis=0,
        ddof=0,
        keepdims=True,
    )

    sd[sd == 0] = 1.0

    return (
        X - mu
    ) / sd


def bh_fdr(p):
    p = np.asarray(
        p,
        dtype=float,
    )

    out = np.full(
        len(p),
        np.nan,
    )

    valid = np.isfinite(p)

    pv = p[valid]

    if len(pv) == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]

    q = (
        ranked
        *
        len(ranked)
        /
        np.arange(
            1,
            len(ranked) + 1,
        )
    )

    q = np.minimum.accumulate(
        q[::-1]
    )[::-1]

    q = np.clip(
        q,
        0,
        1,
    )

    tmp = np.empty(
        len(pv),
    )

    tmp[order] = q

    out[valid] = tmp

    return out


def prepare_df(df):
    """
    Try to create:
        index = drug
        columns = numeric features/genes
    """

    df = df.copy()

    # possible drug column
    for c in [
        "drug",
        "Metadata_Drug",
        "compound",
        "compound_name",
    ]:
        if c in df.columns:
            df = df.set_index(c)
            break

    df.index = (
        df.index
        .astype(str)
    )

    # Remove obvious metadata
    metadata = [
        c
        for c in df.columns
        if (
            str(c).startswith("Metadata_")
            or str(c).lower()
            in {
                "drug",
                "plate",
                "sample",
                "n_cells",
                "moa",
                "moa-fine",
            }
        )
    ]

    if metadata:
        df = df.drop(
            columns=metadata,
        )

    # retain numeric
    df = df.select_dtypes(
        include=[np.number]
    )

    return df


def task2_r(M, G):
    """
    Frozen HepG2 Task-2 geometry:
    separate PCA30
    -> z-score PC axes
    -> Euclidean pair distances
    -> Spearman
    """

    Mpca = PCA(
        n_components=N_PCS,
        svd_solver="full",
        random_state=0,
    ).fit_transform(
        M
    )

    Gpca = PCA(
        n_components=N_PCS,
        svd_solver="full",
        random_state=0,
    ).fit_transform(
        G
    )

    Mz = zscore_cols(Mpca)
    Gz = zscore_cols(Gpca)

    Dm = squareform(
        pdist(Mz)
    )

    Dg = squareform(
        pdist(Gz)
    )

    iu = np.triu_indices(
        len(M),
        1,
    )

    r = spearmanr(
        Dm[iu],
        Dg[iu],
    ).statistic

    return (
        float(r),
        Mz,
        Gz,
        Dm,
        Dg,
    )


# ============================================================
# MORPHOLOGY
# ============================================================

print("=" * 100)
print("HEPG2 — DRUG + GENE CROSS-MODAL RANKINGS")
print("=" * 100)

if not MORPH_FILE.exists():
    raise FileNotFoundError(
        MORPH_FILE
    )

M0 = prepare_df(
    pd.read_parquet(
        MORPH_FILE
    )
)

print(
    "\nMorphology:",
    MORPH_FILE,
)

print(
    "shape:",
    M0.shape,
)

if M0.shape != (
    119,
    636,
):
    print(
        "WARNING: expected approximately 119 x 636"
    )


# normalized drug lookup
morph_lookup = {
    clean_name(x): x
    for x in M0.index
}


# ============================================================
# FIND FINAL 2000-HVG DRUG MATRIX
# ============================================================

print(
    "\nSearching for HepG2 drug-level expression parquets..."
)

candidates = []

for path in DATA_DIR.rglob(
    "*.parquet"
):

    if path == MORPH_FILE:
        continue

    try:
        df = pd.read_parquet(
            path
        )

        d = prepare_df(df)

    except Exception:
        continue

    # drug-level expression should be around
    # 119/120 rows and ~2000 genes
    if not (
        110 <= d.shape[0] <= 130
        and
        1500 <= d.shape[1] <= 2500
    ):
        continue

    expr_lookup = {
        clean_name(x): x
        for x in d.index
    }

    shared_norm = sorted(
        set(morph_lookup)
        &
        set(expr_lookup)
    )

    if len(shared_norm) < 110:
        continue

    M = M0.loc[
        [
            morph_lookup[x]
            for x in shared_norm
        ]
    ].to_numpy(
        dtype=np.float64
    )

    G = d.loc[
        [
            expr_lookup[x]
            for x in shared_norm
        ]
    ].to_numpy(
        dtype=np.float64
    )

    try:
        r, _, _, _, _ = (
            task2_r(
                M,
                G,
            )
        )
    except Exception:
        continue

    candidates.append(
        {
            "path": path,
            "df": d,
            "shared": shared_norm,
            "r": r,
            "difference": abs(
                r - TARGET_R
            ),
        }
    )

    print(
        f"  {path}"
    )
    print(
        f"      shape={d.shape}, "
        f"shared={len(shared_norm)}, "
        f"Task2 r={r:.6f}"
    )


if not candidates:
    raise RuntimeError(
        "\nCould not locate a ~120 x ~2000 "
        "HepG2 expression parquet.\n"
        "Run:\n"
        "find OpenScreen -type f -name '*.parquet' | sort"
    )


candidates.sort(
    key=lambda x: x[
        "difference"
    ]
)

best = candidates[0]


if best[
    "difference"
] > MAX_R_DIFFERENCE:

    print(
        "\nClosest candidate did NOT reproduce "
        "the frozen final r≈0.370."
    )

    print(
        "Closest:",
        best["path"],
    )

    print(
        "r:",
        best["r"],
    )

    raise RuntimeError(
        "Stopping rather than ranking the wrong "
        "HepG2 expression representation."
    )


G0 = best["df"]

print(
    "\nSELECTED FINAL EXPRESSION MATRIX:"
)

print(
    best["path"]
)

print(
    "shape:",
    G0.shape,
)

print(
    f"fingerprint Task2 r={best['r']:.6f}"
)


# ============================================================
# ALIGN 119 DRUGS
# ============================================================

expr_lookup = {
    clean_name(x): x
    for x in G0.index
}

shared_norm = sorted(
    set(morph_lookup)
    &
    set(expr_lookup)
)

drug_names = [
    str(
        morph_lookup[x]
    )
    for x in shared_norm
]

Mdf = M0.loc[
    [
        morph_lookup[x]
        for x in shared_norm
    ]
].copy()

Gdf = G0.loc[
    [
        expr_lookup[x]
        for x in shared_norm
    ]
].copy()

Mdf.index = drug_names
Gdf.index = drug_names


print(
    "\nShared drugs:",
    len(drug_names),
)

print(
    "Morphology:",
    Mdf.shape,
)

print(
    "Expression:",
    Gdf.shape,
)


M = Mdf.to_numpy(
    dtype=np.float64
)

G = Gdf.to_numpy(
    dtype=np.float64
)


# ============================================================
# REBUILD FINAL TASK2 SPACE
# ============================================================

overall_r, Mz, Gz, Dm, Dg = (
    task2_r(
        M,
        G,
    )
)

print(
    "\nReconstructed overall Task2 r:",
    f"{overall_r:.6f}"
)


# ============================================================
# 1. PER-DRUG NEIGHBORHOOD CONCORDANCE
# ============================================================

print(
    "\nCalculating per-drug neighborhood concordance..."
)

rows = []

n = len(
    drug_names
)

for i, drug in enumerate(
    drug_names
):

    other = (
        np.arange(n)
        != i
    )

    rho, p = spearmanr(
        Dm[i, other],
        Dg[i, other],
    )

    rows.append(
        {
            "drug": drug,
            "crossmodal_neighborhood_r": rho,
            "p_value": p,
            "morphology_strength":
                np.linalg.norm(
                    Mz[i]
                ),
            "expression_strength":
                np.linalg.norm(
                    Gz[i]
                ),
        }
    )


drug_rank = pd.DataFrame(
    rows
)

drug_rank[
    "q_value"
] = bh_fdr(
    drug_rank[
        "p_value"
    ]
)

drug_rank = (
    drug_rank
    .sort_values(
        "crossmodal_neighborhood_r",
        ascending=False,
    )
)


drug_rank.to_csv(
    OUT /
    "HepG2_drug_crossmodal_concordance.csv",
    index=False,
)


print(
    "\nTOP 15 CROSS-MODALLY CONCORDANT HEPG2 DRUGS"
)

print(
    drug_rank[
        [
            "drug",
            "crossmodal_neighborhood_r",
            "p_value",
            "q_value",
            "morphology_strength",
            "expression_strength",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


print(
    "\nBOTTOM 10 HEPG2 DRUGS"
)

print(
    drug_rank[
        [
            "drug",
            "crossmodal_neighborhood_r",
        ]
    ]
    .tail(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 2. GENES ASSOCIATED WITH MORPHOLOGY STRENGTH
# ============================================================

print(
    "\nCalculating genes associated with "
    "morphology perturbation strength..."
)

morph_strength = (
    np.linalg.norm(
        Mz,
        axis=1,
    )
)


gene_names = (
    Gdf.columns
    .astype(str)
    .to_numpy()
)


gene_rows = []

for j, gene in enumerate(
    gene_names
):

    rho, p = spearmanr(
        G[:, j],
        morph_strength,
    )

    gene_rows.append(
        {
            "gene": gene,
            "rho_vs_morphology_strength": rho,
            "p_value": p,
        }
    )


gene_strength = pd.DataFrame(
    gene_rows
)

gene_strength[
    "q_value"
] = bh_fdr(
    gene_strength[
        "p_value"
    ]
)

gene_strength[
    "abs_rho"
] = np.abs(
    gene_strength[
        "rho_vs_morphology_strength"
    ]
)

gene_strength = (
    gene_strength
    .sort_values(
        "abs_rho",
        ascending=False,
    )
)


gene_strength.to_csv(
    OUT /
    "HepG2_genes_vs_morphology_strength.csv",
    index=False,
)


print(
    "\nTOP 20 GENES ASSOCIATED WITH "
    "MORPHOLOGY STRENGTH"
)

print(
    gene_strength[
        [
            "gene",
            "rho_vs_morphology_strength",
            "p_value",
            "q_value",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nGenes q<0.05:",
    int(
        (
            gene_strength[
                "q_value"
            ]
            < 0.05
        ).sum()
    )
)


# ============================================================
# 3. MORPHOLOGY FEATURE ↔ GENE
# ============================================================

print(
    "\nCalculating "
    f"{M.shape[1]} x {G.shape[1]} "
    "feature-gene associations..."
)


# Rank each column across drugs.
# Pearson correlation of ranks = Spearman correlation.

MR = np.apply_along_axis(
    rankdata,
    0,
    M,
)

GR = np.apply_along_axis(
    rankdata,
    0,
    G,
)


MR -= MR.mean(
    axis=0,
    keepdims=True,
)

GR -= GR.mean(
    axis=0,
    keepdims=True,
)


Mnorm = np.sqrt(
    np.sum(
        MR ** 2,
        axis=0,
    )
)

Gnorm = np.sqrt(
    np.sum(
        GR ** 2,
        axis=0,
    )
)


Mnorm[
    Mnorm == 0
] = np.nan

Gnorm[
    Gnorm == 0
] = np.nan


R = (
    MR.T
    @
    GR
) / (
    Mnorm[:, None]
    *
    Gnorm[None, :]
)


# approximate two-sided p-values
# using correlation t statistic
df = len(
    drug_names
) - 2

Rc = np.clip(
    R,
    -0.999999999,
    0.999999999,
)


T = (
    Rc
    *
    np.sqrt(
        df
        /
        (
            1
            -
            Rc ** 2
        )
    )
)


P = (
    2
    *
    t.sf(
        np.abs(T),
        df=df,
    )
)


flat_r = R.ravel()
flat_p = P.ravel()

flat_q = bh_fdr(
    flat_p
)


n_sig = int(
    np.sum(
        flat_q < 0.05
    )
)

print(
    "\nFDR q<0.05 feature-gene pairs:",
    n_sig
)

print(
    "Maximum |rho|:",
    float(
        np.nanmax(
            np.abs(flat_r)
        )
    )
)


# Only materialize top 1000 rows,
# not the entire 1.27M-row string dataframe.

order = np.argsort(
    np.nan_to_num(
        np.abs(flat_r),
        nan=-np.inf,
    )
)[::-1]

top = order[
    :1000
]


feature_idx, gene_idx = (
    np.unravel_index(
        top,
        R.shape,
    )
)


morph_features = (
    Mdf.columns
    .astype(str)
    .to_numpy()
)


pairs = pd.DataFrame(
    {
        "morphology_feature":
            morph_features[
                feature_idx
            ],
        "gene":
            gene_names[
                gene_idx
            ],
        "rho":
            flat_r[
                top
            ],
        "p_value":
            flat_p[
                top
            ],
        "q_value":
            flat_q[
                top
            ],
    }
)


pairs.to_csv(
    OUT /
    "HepG2_top1000_feature_gene_pairs.csv",
    index=False,
)


print(
    "\nTOP 20 HEPG2 FEATURE ↔ GENE PAIRS"
)

print(
    pairs.head(
        20
    ).to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "n_drugs":
                len(drug_names),
            "n_morph_features":
                M.shape[1],
            "n_genes":
                G.shape[1],
            "overall_task2_r":
                overall_r,
            "top_drug":
                drug_rank.iloc[0]["drug"],
            "top_drug_r":
                drug_rank.iloc[0][
                    "crossmodal_neighborhood_r"
                ],
            "top_strength_gene":
                gene_strength.iloc[0]["gene"],
            "top_strength_gene_rho":
                gene_strength.iloc[0][
                    "rho_vs_morphology_strength"
                ],
            "feature_gene_FDR05":
                n_sig,
            "max_feature_gene_abs_rho":
                float(
                    np.nanmax(
                        np.abs(flat_r)
                    )
                ),
            "expression_file":
                str(
                    best["path"]
                ),
        }
    ]
)


summary.to_csv(
    OUT /
    "HepG2_drug_gene_ranking_summary.csv",
    index=False,
)


print(
    "\nSaved to:",
    OUT,
)

print(
    "\nDONE"
)
