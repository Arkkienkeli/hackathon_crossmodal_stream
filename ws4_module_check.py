#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0
N_PCS = 10


def zscore_cols(X):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def unit_rows(X):
    X = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm


def entropy_effective_rank(ev):
    p = ev / ev.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-300))))


def participation_ratio(ev):
    p = ev / ev.sum()
    return float(1.0 / np.sum(p ** 2))


def canonical_leading_edge(x):
    if pd.isna(x):
        return ""
    genes = sorted({
        g.strip()
        for g in str(x).split(";")
        if g.strip()
    })
    return ";".join(genes)


def main(morph_path, gex_path, gsea_csv):

    M = pd.read_parquet(morph_path)
    G = pd.read_parquet(gex_path)

    M.index = M.index.astype(str)
    G.index = G.index.astype(str)

    shared = sorted(set(M.index) & set(G.index))

    M = M.loc[shared]
    G = G.loc[shared]

    print(
        f"{len(shared)} drugs | "
        f"{M.shape[1]} morphology | "
        f"{G.shape[1]} genes"
    )

    # ========================================================
    # 1. EXPRESSION DIMENSIONALITY
    # ========================================================

    Z = StandardScaler().fit_transform(
        G.to_numpy(np.float64)
    )

    k = min(
        len(shared) - 1,
        G.shape[1]
    )

    pca = PCA(
        n_components=k,
        random_state=RANDOM_STATE
    ).fit(Z)

    ev = pca.explained_variance_

    er_entropy = entropy_effective_rank(ev)
    er_pr = participation_ratio(ev)

    cum = np.cumsum(
        pca.explained_variance_ratio_
    )

    print("\nEXPRESSION DIMENSIONALITY")
    print(
        f"  entropy effective rank: "
        f"{er_entropy:.1f} of {k}"
    )
    print(
        f"  participation ratio:   "
        f"{er_pr:.1f} of {k}"
    )

    for j in (1, 2, 3, 5, 10, 20, 30):

        if j <= k:
            print(
                f"  first {j:>2} PCs explain "
                f"{cum[j-1]:.1%}"
            )

    # ========================================================
    # 2. SAME DIRECTION-NORMALIZED REPRESENTATION USED
    #    FOR FEATURE <-> GENE ANALYSIS
    # ========================================================

    Mdir = unit_rows(
        zscore_cols(
            M.to_numpy(np.float64)
        )
    )

    Gdir = unit_rows(
        zscore_cols(
            G.to_numpy(np.float64)
        )
    )

    pca_dir = PCA(
        n_components=min(
            len(shared) - 1,
            Gdir.shape[1]
        ),
        random_state=RANDOM_STATE
    )

    scores = pca_dir.fit_transform(
        Gdir
    )

    # ========================================================
    # Exact morphology features used by GSEA
    # ========================================================

    gsea = pd.read_csv(
        gsea_csv
    )

    selected = (
        gsea["morphology_feature"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    selected = [
        f
        for f in selected
        if f in M.columns
    ]

    print(
        "\nGSEA-selected morphology features:",
        len(selected)
    )

    morph_pos = {
        f: i
        for i, f in enumerate(M.columns)
    }

    # ========================================================
    # 3. MORPHOLOGY FEATURES VS EXPRESSION PCs
    # ========================================================

    rows = []

    for feature in selected:

        x = Mdir[
            :,
            morph_pos[feature]
        ]

        rhos = []

        for j in range(N_PCS):

            r = spearmanr(
                x,
                scores[:, j]
            ).statistic

            rhos.append(r)

        row = {
            "feature": feature,
            **{
                f"PC{j+1}": rhos[j]
                for j in range(N_PCS)
            },
            "max_abs_first10":
                float(
                    np.nanmax(
                        np.abs(rhos)
                    )
                )
        }

        rows.append(row)

    tab = pd.DataFrame(rows)

    tab = tab.sort_values(
        "max_abs_first10",
        ascending=False
    )

    tab.to_csv(
        "module_check_feature_pc_correlations.csv",
        index=False
    )

    print(
        "\nDIRECTION-NORMALIZED MORPHOLOGY "
        "VS EXPRESSION PCs"
    )

    print(
        tab.head(25).to_string(
            index=False,
            float_format=lambda v:
                f"{v:+.3f}"
        )
    )

    print(
        "\nmedian |max rho| to first 10 PCs:",
        f"{tab.max_abs_first10.median():.3f}"
    )

    print(
        "features with |rho| >0.30 "
        "to any first-10 PC:",
        f"{int((tab.max_abs_first10 > 0.30).sum())}"
        f"/{len(tab)}"
    )

    # PC1 sign pattern among THE SAME 25 features
    pc1 = tab["PC1"].to_numpy()

    print(
        "\nPC1 SIGN PATTERN"
    )

    print(
        "  positive:",
        int((pc1 > 0).sum())
    )

    print(
        "  negative:",
        int((pc1 < 0).sum())
    )

    # ========================================================
    # 4. LEADING-EDGE REDUNDANCY
    # ========================================================

    q = pd.to_numeric(
        gsea["FDR q-val"],
        errors="coerce"
    )

    hits = gsea[
        q < 0.25
    ].copy()

    print(
        "\nLEADING-EDGE REDUNDANCY"
    )

    print(
        "  pathway hits FDR<0.25:",
        len(hits)
    )

    if len(hits):

        hits[
            "canonical_le"
        ] = hits[
            "Lead_genes"
        ].apply(
            canonical_leading_edge
        )

        n_unique = (
            hits[
                "canonical_le"
            ]
            .replace(
                "",
                np.nan
            )
            .nunique()
        )

        print(
            "  exact distinct leading-edge "
            "gene sets:",
            n_unique
        )

        print(
            "  exact redundancy factor:",
            f"{len(hits) / max(n_unique,1):.2f}x"
        )

        duplicated = (
            hits
            .groupby(
                "canonical_le",
                observed=True
            )
            .agg(
                n_hits=(
                    "pathway",
                    "size"
                ),
                n_distinct_pathways=(
                    "pathway",
                    "nunique"
                ),
                n_features=(
                    "morphology_feature",
                    "nunique"
                ),
                example_pathway=(
                    "pathway",
                    "first"
                ),
            )
            .sort_values(
                "n_hits",
                ascending=False
            )
            .head(15)
        )

        print(
            "\nMost duplicated leading edges:"
        )

        print(
            duplicated.to_string()
        )

    print(
        "\nINTERPRETATION"
    )

    print(
        "A low-dimensional expression representation, "
        "sign-flipping morphology loadings on the same PCs, "
        "and repeated leading-edge gene sets would support "
        "a small number of shared transcriptomic modules rather "
        "than hundreds of independent morphology-pathway links."
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: ws4_module_check.py "
            "M.parquet G.parquet gsea.csv"
        )

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3]
    )
