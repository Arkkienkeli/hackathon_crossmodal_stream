#!/usr/bin/env python3
"""
Is the two-cluster split in the TAHOE drug PCA biology, or plate structure?

What prompted this
------------------
  - the drug-level PCA (fig 4) splits cleanly into two groups along PC1,
    which carries 35.9% of the variance
  - the different-drug null (fig 3) is BIMODAL, with modes near -0.2 and +0.2
  - perturbation magnitude is scattered across both clusters, so potency is
    not what separates them

Two groups of drugs that correlate positively within group and negatively
across produce exactly that double-humped null. And the methods note that each
sample maps to one drug and one plate across 14 plates -- so if a drug's
samples sit on one plate, plate and drug are confounded and PC1 may be a batch
axis.

Why it matters
--------------
An inflated different-drug null (p95 = 0.431) is what leaves only ~33% of
same-drug pairs clearing it. If the inflation is plate structure, the
"transcriptomic reproducibility is limited" conclusion is partly an artifact,
and the expression activity magnitude that carries the r = 0.328 result partly
encodes cluster membership.

Checks performed
----------------
1. how many distinct plates each drug appears on (is drug nested in plate?)
2. variance in each leading PC explained by plate (eta-squared)
3. whether the PC1 sign split maps onto plates
4. the same-drug vs different-drug comparison recomputed using only
   WITHIN-PLATE pairs, which removes the plate contribution from the null

Usage:
    python ws4_batch_check.py \\
        OpenScreen/data/hepg2_sample_log1p_2000hvg.parquet \\
        OpenScreen/data/hepg2_sample_pseudobulk_counts.h5ad
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0


def load_meta(h5ad_path, keys=("drug", "plate", "sample")):
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs.copy()
    a.file.close()
    have = [k for k in keys if k in obs.columns]
    print(f"metadata columns found: {have}")
    missing = [k for k in keys if k not in obs.columns]
    if missing:
        print(f"  missing: {missing} (available: {list(obs.columns)})")
    return obs


def eta_squared(values, groups):
    """Fraction of variance in `values` explained by `groups`."""
    df = pd.DataFrame({"v": values, "g": np.asarray(groups)})
    grand = df.v.mean()
    ss_tot = ((df.v - grand) ** 2).sum()
    ss_between = sum(len(g) * (g.v.mean() - grand) ** 2 for _, g in df.groupby("g"))
    return float(ss_between / ss_tot) if ss_tot > 0 else np.nan


def main(sample_parquet, sample_h5ad, drug_key="drug", plate_key="plate"):
    X = pd.read_parquet(sample_parquet)
    X.index = X.index.astype(str)
    obs = load_meta(sample_h5ad)
    obs.index = obs.index.astype(str)
    meta = obs.reindex(X.index)

    if plate_key not in meta.columns:
        print(f"\nno '{plate_key}' column -- cannot run the batch check. "
              f"Rebuild the sample pseudobulk carrying plate in obs.")
        return

    drugs = meta[drug_key].astype(str)
    plates = meta[plate_key].astype(str)
    print(f"\n{len(X)} samples | {drugs.nunique()} drugs | {plates.nunique()} plates")

    # ---- 1. is drug nested inside plate? --------------------------------
    per_drug = drugs.groupby(drugs).apply(lambda s: plates[s.index].nunique())
    print("\nplates per drug:")
    print(per_drug.value_counts().sort_index().to_string())
    if (per_drug == 1).mean() > 0.5:
        print("  -> MOST DRUGS SIT ON A SINGLE PLATE. Drug and plate are "
              "confounded: any plate effect is indistinguishable from a drug "
              "effect, and replicate agreement partly measures plate.")
    else:
        print("  -> drugs span multiple plates; the confound is partial")

    # ---- 2. how much of each PC is plate? --------------------------------
    Z = StandardScaler().fit_transform(X.to_numpy(np.float64))
    P = PCA(n_components=10, random_state=RANDOM_STATE)
    S = P.fit_transform(Z)
    print("\nvariance in each PC explained by plate:")
    rows = []
    for i in range(5):
        e_plate = eta_squared(S[:, i], plates)
        e_drug = eta_squared(S[:, i], drugs)
        rows.append({"PC": i + 1, "var_explained": P.explained_variance_ratio_[i],
                     "eta2_plate": e_plate, "eta2_drug": e_drug})
        print(f"  PC{i+1}  {P.explained_variance_ratio_[i]:.3f} of variance | "
              f"plate eta2 = {e_plate:.3f} | drug eta2 = {e_drug:.3f}")
    if rows[0]["eta2_plate"] > 0.5:
        print("  -> PC1 is largely a PLATE axis, not a biological one")

    # ---- 3. does the PC1 split map onto plates? --------------------------
    side = pd.Series(np.where(S[:, 0] > np.median(S[:, 0]), "hi", "lo"),
                     index=X.index)
    ct = pd.crosstab(plates, side)
    purity = (ct.max(axis=1) / ct.sum(axis=1))
    print(f"\nplate purity w.r.t. the PC1 split: median {purity.median():.2f}, "
          f"{int((purity > 0.9).sum())}/{len(purity)} plates >90% on one side")
    if purity.median() > 0.9:
        print("  -> the PC1 split is essentially a plate grouping")

    # ---- 4. null recomputed within plate ---------------------------------
    G = pd.DataFrame(StandardScaler().fit_transform(X.to_numpy(np.float64)),
                     index=X.index)
    A = G.to_numpy()
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
    n = len(A)
    rng = np.random.default_rng(RANDOM_STATE)

    def corr(i, j):
        return float((A[i] * A[j]).mean())

    same_all, diff_all, diff_within = [], [], []
    idx = np.arange(n)
    dr = drugs.to_numpy()
    pl = plates.to_numpy()
    for i, j in combinations(idx, 2):
        if dr[i] == dr[j]:
            same_all.append(corr(i, j))
    pairs = rng.integers(0, n, (40000, 2))
    for i, j in pairs:
        if i == j or dr[i] == dr[j]:
            continue
        c = corr(i, j)
        diff_all.append(c)
        if pl[i] == pl[j]:
            diff_within.append(c)

    def summarise(name, v):
        v = np.asarray(v)
        return (f"  {name:28s} n={len(v):>6}  median {np.median(v):+.3f}  "
                f"p95 {np.quantile(v, 0.95):+.3f}")

    print("\nreplicate comparison:")
    print(summarise("same-drug pairs", same_all))
    print(summarise("different-drug, any plate", diff_all))
    print(summarise("different-drug, SAME plate", diff_within))

    p95_all = np.quantile(diff_all, 0.95)
    p95_within = np.quantile(diff_within, 0.95) if diff_within else np.nan
    frac_all = float(np.mean(np.asarray(same_all) > p95_all))
    frac_within = float(np.mean(np.asarray(same_all) > p95_within))
    print(f"\nfraction of same-drug pairs above the null p95:")
    print(f"  against the any-plate null   {frac_all:.1%}")
    print(f"  against the same-plate null  {frac_within:.1%}")

    med_all = float(np.median(diff_all))
    med_within = float(np.median(diff_within)) if diff_within else np.nan
    print("\nreading:")
    if not np.isnan(med_within) and abs(med_within - med_all) > 0.15:
        print(f"  The different-drug median moves from {med_all:+.3f} (any "
              f"plate) to {med_within:+.3f} (same plate). That gap IS the "
              f"batch structure: unrelated drugs on the same plate look alike, "
              f"and drugs on different plates look anti-correlated. The "
              f"bimodal null and the two-cluster PCA are both this effect.")
        print(f"  Consequence: the pooled null p95 ({p95_all:.3f}) is not a "
              f"valid reference. Recompute reproducibility within plate, and "
              f"treat expression PC1 as a batch axis until shown otherwise.")
    elif rows[0]["eta2_plate"] > 0.5:
        print("  PC1 tracks plate strongly even though the null medians are "
              "similar. Treat PC1 as suspect and check a second grouping "
              "variable (sub-library, sequencing run, harvest day).")
    else:
        print("  No plate signature in the leading PCs and no shift in the "
              "null when plate is held constant. The two clusters are not a "
              "plate effect -- look for another grouping variable, or accept "
              "them as biological.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:5])
