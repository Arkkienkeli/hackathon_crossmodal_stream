"""
QC for the three OpenScreen HepG2 morphology sites (FMP, IMTM, MEDINA).

Answers, in order:
  1. what is actually in each site file
  2. do the three sites share a feature space
  3. do the sites agree on a compound -- i.e. is a compound's profile more like
     its own replicate at another site than like other compounds
  4. does acquisition site dominate the geometry (batch effect)
  5. which compounds survive to the TAHOE match

Step 3 is the one that decides whether the downstream modelling is worth doing.
The standard field metric is "percent replicating": compare matched-compound
cross-site correlations against a null built from non-matching pairs, and count
how many matched pairs exceed the null's 95th percentile. A consensus profile
averaged over sites that disagree is an average of noise.

All of this runs on the ~1.3 MB per-site files. Nothing here opens TAHOE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 0
SITES = ("FMP", "IMTM", "MEDINA")


# --------------------------------------------------------------------------
# 1. what is in the files
# --------------------------------------------------------------------------

def load_sites(data_dir, sites=SITES, consensus=True, cell_line="HepG2"):
    """Read the per-site h5ad files into a dict of AnnData."""
    import anndata as ad
    import os

    out = {}
    for s in sites:
        suffix = "_consensus" if consensus else ""
        path = os.path.join(data_dir, f"morphology_{s}_{cell_line}{suffix}.h5ad")
        out[s] = ad.read_h5ad(path)
        print(f"{s:8s} {out[s].shape[0]:>6} rows x {out[s].shape[1]:>5} features"
              f"  ({os.path.getsize(path)/1e6:.1f} MB)")
    return out


def guess_compound_key(adata, candidates=("drug", "compound", "pert_iname",
                                          "Metadata_compound", "broad_id",
                                          "Metadata_pert_iname", "InChIKey")):
    """Find the column holding compound identity, or show what is available."""
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError(
        "no obvious compound column; obs has: " + ", ".join(adata.obs.columns)
    )


def describe_sites(sites, compound_key=None):
    """Compounds, features, missingness and non-finite values per site."""
    rows = []
    for name, a in sites.items():
        key = compound_key or guess_compound_key(a)
        X = np.asarray(a.X, dtype=np.float64)
        rows.append({
            "site": name,
            "rows": a.n_obs,
            "compounds": a.obs[key].nunique(),
            "features": a.n_vars,
            "nan_frac": float(np.isnan(X).mean()),
            "inf_frac": float(np.isinf(X).mean()),
            "zero_var_features": int((np.nanstd(X, axis=0) == 0).sum()),
            "median_abs": float(np.nanmedian(np.abs(X))),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if (df.nan_frac > 0).any():
        print("  -> NaNs present; decide impute vs drop-feature BEFORE consensus")
    if (df.zero_var_features > 0).any():
        print("  -> constant features present; drop them, they break correlation")
    return df


# --------------------------------------------------------------------------
# 2. shared feature space
# --------------------------------------------------------------------------

def feature_consistency(sites):
    """Are the three sites even on the same features, in the same order?"""
    names = {s: list(a.var_names) for s, a in sites.items()}
    sets = {s: set(v) for s, v in names.items()}
    common = set.intersection(*sets.values())

    print(f"shared features across all sites: {len(common)}")
    for s, v in sets.items():
        only = v - common
        print(f"  {s:8s} {len(v):>5} total, {len(only):>4} not shared"
              + (f"  e.g. {sorted(only)[:3]}" if only else ""))

    order_same = len({tuple(v) for v in names.values()}) == 1
    print(f"identical feature order across sites: {order_same}")
    if not order_same:
        print("  -> reindex every site to a common sorted feature list before "
              "any cross-site comparison; do not rely on positional alignment")
    return sorted(common)


def to_matrix(adata, features, compound_key, aggregate="median"):
    """Compound x feature matrix on a fixed feature list."""
    key = compound_key or guess_compound_key(adata)
    X = pd.DataFrame(np.asarray(adata.X, dtype=np.float64),
                     index=adata.obs[key].astype(str).to_numpy(),
                     columns=list(adata.var_names))
    X = X.reindex(columns=features)
    if X.index.has_duplicates:
        X = X.groupby(level=0).agg(aggregate)
    return X.sort_index()


# --------------------------------------------------------------------------
# 3. do the sites agree on a compound
# --------------------------------------------------------------------------

def cross_site_reproducibility(mats, n_null=20000, percentile=95):
    """Matched-compound cross-site correlation vs a non-matching null.

    mats: dict of site -> compound x feature DataFrame (same feature list).

    Returns per-site-pair statistics and a per-compound table so you can see
    WHICH compounds reproduce, not just how many.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    site_names = list(mats)
    pair_rows, per_compound = [], {}

    for i in range(len(site_names)):
        for j in range(i + 1, len(site_names)):
            a, b = site_names[i], site_names[j]
            idx = mats[a].index.intersection(mats[b].index)
            A = mats[a].loc[idx].to_numpy()
            B = mats[b].loc[idx].to_numpy()

            keep = np.isfinite(A).all(0) & np.isfinite(B).all(0)
            A, B = A[:, keep], B[:, keep]
            A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
            B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)

            matched = (A * B).mean(1)

            n = len(idx)
            ii = rng.integers(0, n, n_null)
            jj = rng.integers(0, n, n_null)
            ok = ii != jj
            null = (A[ii[ok]] * B[jj[ok]]).mean(1)
            thr = float(np.percentile(null, percentile))
            frac = float((matched > thr).mean())

            pair_rows.append({
                "pair": f"{a} vs {b}",
                "n_compounds": int(n),
                "median_matched_r": float(np.median(matched)),
                "median_null_r": float(np.median(null)),
                f"null_p{percentile}": thr,
                "percent_replicating": frac,
            })
            per_compound[f"{a}|{b}"] = pd.Series(matched, index=idx)

    df = pd.DataFrame(pair_rows)
    print(df.to_string(index=False))
    worst = df.percent_replicating.min()
    if worst < 0.5:
        print(f"  -> weakest pair replicates only {worst:.0%} of compounds. "
              f"The consensus is averaging over real disagreement; report this "
              f"and consider restricting to reproducible compounds.")
    else:
        print(f"  -> all pairs replicate >= {worst:.0%}; consensus is defensible")
    return df, pd.DataFrame(per_compound)


def reproducible_compounds(per_compound, min_r=None, quantile=0.5):
    """Compounds whose cross-site agreement is decent in EVERY site pair.

    Use as an optional filter for a sensitivity analysis: rerun the MoA
    experiment on this subset and check the conclusion does not flip.
    """
    worst = per_compound.min(axis=1)
    if min_r is None:
        min_r = float(worst.quantile(quantile))
    keep = worst >= min_r
    print(f"{int(keep.sum())}/{len(keep)} compounds agree across all site pairs "
          f"at r >= {min_r:.3f}")
    return per_compound.index[keep].to_numpy(), min_r


# --------------------------------------------------------------------------
# 4. does site dominate the geometry
# --------------------------------------------------------------------------

def site_effect_check(mats, k=5):
    """Is a profile's nearest neighbour its own compound, or just its own site?

    Stacks all site x compound profiles and asks, for each one, how often the k
    nearest neighbours share its SITE versus share its COMPOUND. If site wins
    decisively, acquisition batch is the dominant axis and the consensus (or a
    site-aware correction) has to come before any modelling.
    """
    from sklearn.preprocessing import StandardScaler

    frames, site_lbl, cmp_lbl = [], [], []
    for s, m in mats.items():
        frames.append(m.to_numpy())
        site_lbl += [s] * len(m)
        cmp_lbl += list(m.index)
    X = np.vstack(frames)
    site_lbl = np.array(site_lbl)
    cmp_lbl = np.array(cmp_lbl)

    good = np.isfinite(X).all(0)
    X = StandardScaler().fit_transform(X[:, good])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]

    same_site = float((site_lbl[nn] == site_lbl[:, None]).mean())
    same_cmp = float((cmp_lbl[nn] == cmp_lbl[:, None]).mean())
    n_sites = len(mats)
    chance_site = float(np.mean([np.mean(site_lbl == s) for s in site_lbl]))

    print(f"neighbour composition (k={k}):")
    print(f"  same SITE     {same_site:.3f}   (chance ~{chance_site:.3f})")
    print(f"  same COMPOUND {same_cmp:.3f}   (chance ~{(n_sites-1)/(len(X)-1):.4f})")
    if same_site > chance_site + 0.25 and same_cmp < 0.3:
        print("  -> site dominates compound identity; treat site as a batch "
              "variable and hold sites out in CV, not just compounds")
    return {"same_site": same_site, "same_compound": same_cmp,
            "chance_site": chance_site}


# --------------------------------------------------------------------------
# 5. match to the expression side
# --------------------------------------------------------------------------

def match_to_expression(morph_index, expr_drugs, normalise=True):
    """Intersect morphology compounds with TAHOE drugs and show what is lost."""
    def norm(x):
        return str(x).strip().lower() if normalise else str(x)

    m = pd.Index([norm(x) for x in morph_index])
    e = pd.Index([norm(x) for x in expr_drugs])
    shared = m.intersection(e)

    print(f"morphology {len(m)} | expression {len(e)} | shared {len(shared)}")
    only_e = e.difference(m)
    if len(only_e):
        print(f"  in expression but not morphology ({len(only_e)}): "
              f"{sorted(only_e)[:10]}")
    only_m = m.difference(e)
    if len(only_m):
        print(f"  in morphology but not expression ({len(only_m)}): "
              f"{sorted(only_m)[:10]}")
    return sorted(shared)