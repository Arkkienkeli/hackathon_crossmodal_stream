#!/usr/bin/env python3
"""
Build the four half-matrices for ws4_noise_ceiling.py.

Expression halves come from the 409 sample-level pseudobulks: samples of the
same drug are split into two disjoint groups and averaged. Drugs with fewer
than 2 surviving samples cannot be split and are dropped.

Morphology halves come from two acquisition sites, which are already
independent measurements of the same compounds.

Note on what "reliability" means here. The document reports same-drug replicate
median r = 0.355 against a different-drug null p95 of 0.431 -- i.e. the median
same-drug pair does NOT clear the null, and only ~33% of pairs do. That is
profile-level reliability. Direction-level reliability, which is what the
cross-modal direction statistic is attenuated by, is normally lower still, so
expect a low ceiling.

Usage:
    python ws4_make_halves.py \\
        OpenScreen/data/hepg2_sample_log1p_2000hvg.parquet \\
        OpenScreen/data/hepg2_sample_pseudobulk_counts.h5ad \\
        OpenScreen/sphering/FMP_HepG2_baseline_consensus.parquet \\
        OpenScreen/sphering/IMTM_HepG2_baseline_consensus.parquet \\
        [expression_drug_to_pdid.csv]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

RANDOM_STATE = 0


def sample_to_drug(h5ad_path, sample_key="sample", drug_key="drug"):
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs.copy()
    a.file.close()
    if sample_key in obs.columns:
        return obs.set_index(sample_key)[drug_key].astype(str)
    # sample pseudobulks are usually indexed BY sample
    return obs[drug_key].astype(str)


def split_expression(sample_matrix, s2d, seed=RANDOM_STATE):
    """Two disjoint sample-halves per drug, each averaged."""
    rng = np.random.default_rng(seed)
    X = sample_matrix.copy()
    X.index = X.index.astype(str)
    drugs = s2d.reindex(X.index)
    missing = drugs.isna().sum()
    if missing:
        print(f"  {missing} samples have no drug label, dropped")
        X = X[drugs.notna()]
        drugs = drugs.dropna()

    a_rows, b_rows, dropped = {}, {}, []
    for drug, idx in drugs.groupby(drugs).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            dropped.append(drug)
            continue
        order = list(rng.permutation(idx))
        half = len(order) // 2
        a_rows[drug] = X.loc[order[:half] or order[:1]].mean()
        b_rows[drug] = X.loc[order[half:]].mean()

    if dropped:
        print(f"  {len(dropped)} drugs had <2 samples and cannot be split: "
              f"{dropped[:8]}")
    A = pd.DataFrame(a_rows).T
    B = pd.DataFrame(b_rows).T
    print(f"  expression halves: {A.shape} and {B.shape}")
    return A, B


def _read_site(path):
    """Accept either a parquet (indexed by pdid) or a consensus h5ad
    (indexed via obs['Metadata_Drug'], which needs no pdid mapping)."""
    if str(path).endswith(".h5ad"):
        import anndata as ad
        a = ad.read_h5ad(path)
        key = "Metadata_Drug" if "Metadata_Drug" in a.obs.columns else a.obs.columns[0]
        df = pd.DataFrame(np.asarray(a.X, dtype=np.float64),
                          index=a.obs[key].astype(str).to_numpy(),
                          columns=list(a.var_names))
        return df.groupby(level=0).mean()
    return pd.read_parquet(path)


def morphology_halves(p_a, p_b, pdid_map=None):
    A = _read_site(p_a)
    B = _read_site(p_b)
    if pdid_map is not None:
        A.index = A.index.astype(str).map(pdid_map)
        B.index = B.index.astype(str).map(pdid_map)
        A = A[A.index.notna()]
        B = B[B.index.notna()]
    A.index = A.index.astype(str)
    B.index = B.index.astype(str)
    print(f"  morphology halves: {A.shape} and {B.shape}")
    return A, B


def main(sample_parquet, sample_h5ad, morph_a, morph_b, pdid_csv=None):
    print("expression:")
    X = pd.read_parquet(sample_parquet)
    s2d = sample_to_drug(sample_h5ad)
    Ga, Gb = split_expression(X, s2d)

    pdid_map = None
    if pdid_csv:
        m = pd.read_csv(pdid_csv)
        m = m.dropna(subset=["pdid"])
        pdid_map = dict(zip(m["pdid"].astype(str), m["drug"].astype(str)))
        print(f"\nmorphology (mapping {len(pdid_map)} pdids to drug names):")
    else:
        print("\nmorphology:")
    Ma, Mb = morphology_halves(morph_a, morph_b, pdid_map)

    shared = sorted(set(Ma.index) & set(Mb.index) & set(Ga.index) & set(Gb.index))
    print(f"\ndrugs present in all four halves: {len(shared)}")
    if len(shared) < 60:
        print("  -> too few for a stable Mantel; check the index conventions "
              "match (pdid vs drug name, case, salt suffixes)")

    for d, name in ((Ma.loc[shared], "M_half_a"), (Mb.loc[shared], "M_half_b"),
                    (Ga.loc[shared], "G_half_a"), (Gb.loc[shared], "G_half_b")):
        d.to_parquet(f"{name}.parquet")
        print(f"  wrote {name}.parquet {d.shape}")

    print("\nnext:\n  python ws4_noise_ceiling.py M_half_a.parquet "
          "M_half_b.parquet G_half_a.parquet G_half_b.parquet")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:6])