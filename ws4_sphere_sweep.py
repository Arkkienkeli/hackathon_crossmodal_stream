#!/usr/bin/env python3
"""
Why DMSO-referenced sphering hurt, and whether regularisation fixes it.

The result being explained
--------------------------
ZCA-cor sphering at epsilon=1e-6 lowered cross-site percent_replicating
(0.56/0.47/0.46 -> 0.36/0.38/0.35), lowered within-site reproducibility at two
of three sites, and RAISED same-site neighbour enrichment (0.784 -> 0.898)
while lowering same-compound (0.291 -> 0.163).

The hypothesis
--------------
784 DMSO wells estimate a 636x636 covariance: n/p ~ 1.2. The smallest
eigenvalues are near zero and badly determined. Whitening divides by their
square roots, so those directions get amplified into noise -- and since each
site has its own ill-conditioned estimate, the amplified noise is
site-specific. That explains all three observations at once, including the
counter-intuitive rise in same-site clustering.

This script measures the conditioning directly, then sweeps regularisation:
  - ZCA-cor at a range of epsilon values
  - PCA whitening truncated to k components (regularised by construction)
and reports the null-referenced metric for each. Works from
hepg2_mad_selected.parquet, so no CSV reloading or feature selection.

Usage:
    python ws4_sphere_sweep.py OpenScreen/sphering/hepg2_mad_selected.parquet
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd

RANDOM_STATE = 0
REPORT_SITES = ["FMP_HepG2", "IMTM_HepG2", "MEDINA_HepG2"]
EPSILONS = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
PCA_KS = [50, 100, 200, 400]
N_NULL = 10000


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------

def conditioning_report(X_dmso, label=""):
    """Eigenspectrum of the DMSO correlation matrix -- what whitening inverts."""
    Z = X_dmso - X_dmso.mean(0, keepdims=True)
    sd = Z.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Z / sd
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = np.clip(ev, 0, None)

    n, p = X_dmso.shape
    tot = ev.sum()
    eff_rank = float(np.exp(-(ev / tot * np.log(ev / tot + 1e-300)).sum()))
    out = {
        "site": label,
        "n_dmso": n,
        "p_features": p,
        "n_over_p": n / p,
        "largest_ev": float(ev[0]),
        "smallest_ev": float(ev[-1]),
        "condition_number": float(ev[0] / max(ev[-1], 1e-12)),
        "ev_below_1e-3": int((ev < 1e-3).sum()),
        "ev_below_1e-2": int((ev < 1e-2).sum()),
        "effective_rank": eff_rank,
    }
    return out


# --------------------------------------------------------------------------
# whitening
# --------------------------------------------------------------------------

def fit_zca_cor(X_dmso, epsilon):
    """ZCA-cor whitening fitted on control wells, as pycytominer does it."""
    mu = X_dmso.mean(0)
    sd = X_dmso.std(0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X_dmso - mu) / sd
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    ev, V = np.linalg.eigh(C)
    ev = np.clip(ev, 0, None)
    W = V @ np.diag(1.0 / np.sqrt(ev + epsilon)) @ V.T
    return lambda X: ((X - mu) / sd) @ W


def fit_pca_whiten(X_dmso, k):
    """PCA whitening truncated to k components -- regularised by construction."""
    from sklearn.decomposition import PCA

    mu = X_dmso.mean(0)
    sd = X_dmso.std(0, ddof=1)
    sd[sd == 0] = 1.0
    p = PCA(n_components=min(k, X_dmso.shape[0] - 1, X_dmso.shape[1]),
            whiten=True, random_state=RANDOM_STATE)
    p.fit((X_dmso - mu) / sd)
    return lambda X: p.transform((X - mu) / sd)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _row_corr(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    d = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    d[d == 0] = np.nan
    return (A * B).sum(1) / d


def cross_site_pr(consensus, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rows = []
    for s1, s2 in combinations(consensus, 2):
        idx = consensus[s1].index.intersection(consensus[s2].index)
        A = consensus[s1].loc[idx].to_numpy(np.float64)
        B = consensus[s2].loc[idx].to_numpy(np.float64)
        matched = _row_corr(A, B)
        null = []
        while len(null) < N_NULL:
            perm = rng.permutation(len(idx))
            v = _row_corr(A, B[perm])
            null.extend(v[(perm != np.arange(len(idx))) & np.isfinite(v)].tolist())
        null = np.asarray(null[:N_NULL])
        p95 = np.nanquantile(null, 0.95)
        rows.append({"pair": f"{s1[:4]}/{s2[:4]}",
                     "percent_replicating": float(np.nanmean(matched > p95))})
    return pd.DataFrame(rows)


def site_nn(X, sites, pdids, k=5):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    return (float((sites[idx] == sites[:, None]).mean()),
            float((pdids[idx] == pdids[:, None]).mean()))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def evaluate(df, features, transform_per_site, label):
    parts, cons = [], {}
    for site in REPORT_SITES:
        s = df[df["Metadata_Site"] == site]
        dmso = s[s["Metadata_negcon"]][features].to_numpy(np.float64)
        trt = s[~s["Metadata_negcon"] & s["Metadata_pdid"].notna()].copy()
        f = transform_per_site(dmso)
        Y = f(trt[features].to_numpy(np.float64))
        cols = [f"c{i}" for i in range(Y.shape[1])]
        Yd = pd.DataFrame(Y, columns=cols)
        Yd["Metadata_pdid"] = trt["Metadata_pdid"].astype(str).to_numpy()
        Yd["Metadata_Site"] = site
        parts.append(Yd)
        cons[site] = Yd.groupby("Metadata_pdid", observed=True)[cols].mean().sort_index()

    stacked = pd.concat(parts, ignore_index=True)
    cols = [c for c in stacked.columns if not c.startswith("Metadata_")]
    ss, sc = site_nn(stacked[cols].to_numpy(np.float64),
                     stacked["Metadata_Site"].to_numpy(),
                     stacked["Metadata_pdid"].to_numpy())
    pr = cross_site_pr(cons)
    return {"transform": label, "dims": len(cols),
            "cross_site_PR_mean": float(pr.percent_replicating.mean()),
            "cross_site_PR_min": float(pr.percent_replicating.min()),
            "same_site_nn": ss, "same_compound_nn": sc}


def main(parquet):
    df = pd.read_parquet(parquet)
    features = [c for c in df.columns if not c.startswith("Metadata_")]
    df = df[df["Metadata_Site"].isin(REPORT_SITES)]
    print(f"{len(df)} wells, {len(features)} features\n")

    print("CONDITIONING OF THE DMSO COVARIANCE")
    cond = pd.DataFrame([
        conditioning_report(
            df[(df.Metadata_Site == s) & df.Metadata_negcon][features]
            .to_numpy(np.float64), s)
        for s in REPORT_SITES])
    print(cond.to_string(index=False))
    if (cond["n_over_p"] < 3).any():
        print("  -> n/p below 3: the smallest eigenvalues are not reliably "
              "estimated, and whitening divides by their square roots")

    rows = [evaluate(df, features, lambda d: (lambda X: X), "none (baseline)")]
    print(f"\n{rows[0]}")

    for eps in EPSILONS:
        r = evaluate(df, features,
                     lambda d, e=eps: fit_zca_cor(d, e), f"ZCA-cor eps={eps:g}")
        rows.append(r)
        print(r)

    for k in PCA_KS:
        r = evaluate(df, features,
                     lambda d, kk=k: fit_pca_whiten(d, kk), f"PCA-whiten k={k}")
        rows.append(r)
        print(r)

    out = pd.DataFrame(rows)
    out.to_csv("sphere_sweep.csv", index=False)
    print("\n" + out.to_string(index=False))
    best = out.loc[out["cross_site_PR_mean"].idxmax()]
    print(f"\nbest cross-site PR: {best['transform']} "
          f"({best['cross_site_PR_mean']:.3f})")
    if str(best["transform"]).startswith("none"):
        print("no whitening variant beats the unsphered baseline -- report "
              "that sphering does not help this dataset at this control count")
    print("saved sphere_sweep.csv")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
