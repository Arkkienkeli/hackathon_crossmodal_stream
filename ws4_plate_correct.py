#!/usr/bin/env python3
"""
Correct stratified analysis of the TAHOE plate effect, and a plate-corrected
drug-level matrix.

Why this supersedes the pooled numbers
--------------------------------------
ws4_batch_check reported two figures, and both compare the wrong things:

  32.8% of same-drug pairs beat the ANY-PLATE null. That null pools same-plate
  pairs (median +0.340) with cross-plate pairs (median well below zero), so its
  p95 of 0.432 is inflated by the batch effect.

  3.0% beat the SAME-PLATE null. Worse: 77 of 120 drugs sit on 3 plates and 32
  on 4, so most same-drug pairs are CROSS-plate. Scoring cross-plate pairs
  against a same-plate null is apples to oranges and badly understates the
  drug signal.

The comparison has to be stratified on both sides:

    same-drug  same-plate   vs   different-drug  same-plate
    same-drug  cross-plate  vs   different-drug  cross-plate

The second row is the one that matters here, and it is also the honest one:
it asks whether a drug reproduces across plates, with plate held out of the
comparison rather than smuggled into the null.

A note on the eta-squared table: drug has 120 levels and plate has 14, so drug
eta2 is inflated by degrees of freedom and the two columns are not directly
comparable. Plate eta2 of 0.36-0.61 on PC1, PC2 and PC4 is the real signal
there; PC4 in particular (plate 0.610, drug 0.122) is close to a pure plate
axis.

Usage:
    python ws4_plate_correct.py \\
        OpenScreen/data/hepg2_sample_log1p_2000hvg.parquet \\
        OpenScreen/data/hepg2_sample_pseudobulk_counts.h5ad
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0


def load_meta(h5ad_path):
    import anndata as ad
    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs.copy()
    a.file.close()
    return obs


def zrows(A):
    A = A - A.mean(1, keepdims=True)
    return A / (A.std(1, keepdims=True) + 1e-12)


def stratified_pairs(A, drugs, plates, max_pairs=60000, seed=RANDOM_STATE):
    """Split every pair into the four drug x plate cells."""
    rng = np.random.default_rng(seed)
    n = len(A)
    out = {("same", "same"): [], ("same", "cross"): [],
           ("diff", "same"): [], ("diff", "cross"): []}

    # all same-drug pairs (few enough to enumerate)
    for i, j in combinations(range(n), 2):
        if drugs[i] == drugs[j]:
            k = ("same", "same" if plates[i] == plates[j] else "cross")
            out[k].append(float((A[i] * A[j]).mean()))

    # sample different-drug pairs
    ii = rng.integers(0, n, max_pairs)
    jj = rng.integers(0, n, max_pairs)
    for i, j in zip(ii, jj):
        if i == j or drugs[i] == drugs[j]:
            continue
        k = ("diff", "same" if plates[i] == plates[j] else "cross")
        out[k].append(float((A[i] * A[j]).mean()))
    return {k: np.asarray(v) for k, v in out.items()}


def report_strata(cells):
    print(f"{'stratum':34s} {'n':>7} {'median':>9} {'p95':>9}")
    for (d, p), v in cells.items():
        if len(v) == 0:
            continue
        print(f"{d+'-drug, '+p+'-plate':34s} {len(v):>7} "
              f"{np.median(v):>+9.3f} {np.quantile(v, 0.95):>+9.3f}")

    out = {}
    for p in ("same", "cross"):
        s, d = cells[("same", p)], cells[("diff", p)]
        if len(s) == 0 or len(d) == 0:
            continue
        thr = float(np.quantile(d, 0.95))
        frac = float(np.mean(s > thr))
        out[p] = {"n_same": len(s), "median_same": float(np.median(s)),
                  "median_diff": float(np.median(d)), "null_p95": thr,
                  "frac_above": frac}
        print(f"\n  {p}-plate comparison: same-drug median "
              f"{np.median(s):+.3f} vs different-drug {np.median(d):+.3f}; "
              f"{frac:.1%} of same-drug pairs clear the matched null p95 "
              f"({thr:+.3f})")
    return out


def plate_center(X, plates):
    """Subtract each plate's mean profile.

    Legitimate here because drugs span plates (77 drugs on 3, 32 on 4), so the
    plate mean is an average over several different drugs rather than a proxy
    for one. Some drug signal is lost -- report that the correction was applied.
    """
    Xc = X.copy()
    for p in pd.unique(plates):
        m = plates == p
        Xc[m] = Xc[m] - Xc[m].mean(0, keepdims=True)
    return Xc


def main(sample_parquet, sample_h5ad, out_prefix="hepg2_platecorrected"):
    X = pd.read_parquet(sample_parquet)
    X.index = X.index.astype(str)
    obs = load_meta(sample_h5ad)
    obs.index = obs.index.astype(str)
    meta = obs.reindex(X.index)
    drugs = meta["drug"].astype(str).to_numpy()
    plates = meta["plate"].astype(str).to_numpy()
    raw = X.to_numpy(np.float64)
    print(f"{len(X)} samples | {len(set(drugs))} drugs | {len(set(plates))} plates")

    print("\n=== BEFORE plate correction (stratified) ===")
    before = report_strata(stratified_pairs(
        zrows(StandardScaler().fit_transform(raw)), drugs, plates))

    print("\n=== AFTER plate centring ===")
    corrected = plate_center(StandardScaler().fit_transform(raw), plates)
    after = report_strata(stratified_pairs(zrows(corrected), drugs, plates))

    # drug-level matrix from plate-corrected samples, equal weight per sample
    dfc = pd.DataFrame(corrected, index=X.index, columns=X.columns)
    # sample-level corrected matrix, needed to build halves for the noise ceiling
    dfc.to_parquet(f"{out_prefix}_sample_2000hvg.parquet")
    print(f"\nwrote {out_prefix}_sample_2000hvg.parquet {dfc.shape}")
    dfc["__drug__"] = drugs
    drug_level = dfc.groupby("__drug__", observed=True).mean()
    drug_level.to_parquet(f"{out_prefix}_drug_2000hvg.parquet")
    print(f"wrote {out_prefix}_drug_2000hvg.parquet {drug_level.shape}")

    print("\nreading:")
    b, a = before.get("cross"), after.get("cross")
    if b:
        print(f"  Cross-plate, before correction: same-drug {b['median_same']:+.3f} "
              f"vs different-drug {b['median_diff']:+.3f}, "
              f"{b['frac_above']:.0%} above the matched null.")
        print("  This is the honest measure of transcriptomic reproducibility: "
              "a drug reproducing across plates cannot be explained by plate.")
    if a:
        print(f"  Cross-plate, after plate centring: same-drug "
              f"{a['median_same']:+.3f} vs different-drug "
              f"{a['median_diff']:+.3f}, {a['frac_above']:.0%} above null.")
    if b and a and a["frac_above"] >= b["frac_above"] - 0.05:
        print("  Plate centring does not cost drug signal -- use the corrected "
              "matrix for Task 2 and rerun the decomposition on it.")
    else:
        print("  Plate centring costs drug signal; report both, and treat the "
              "corrected result as the conservative bound.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:4])