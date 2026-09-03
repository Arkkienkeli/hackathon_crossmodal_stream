"""
Drug-level pseudobulk expression matrix for WS4 (OpenScreen / HepG2).

Produces the 119 x 2000 counterpart to `consensus_all_sites` so that PCA can be
fit inside each CV fold instead of once on all compounds.

Design decisions worth defending at the readout:

1. Reads the h5ad in backed mode and aggregates in chunks. Nothing here loads
   the full expression matrix, so it is safe on a login node.
2. Normalisation is per cell over ALL genes, then log1p, then mean per drug.
   Subsetting to HVGs before computing library size would distort the
   normalisation, so the totals are computed on the full row.
3. HVG selection: hepg2_cells.h5ad has NO vehicle/DMSO group (controls exist on
   the morphology side only), so hvg_on="all" is the only option here and the
   feature space is chosen with the treated cells in view. That is an
   unsupervised, label-free step, but it is still global -- state it plainly in
   the writeup rather than implying fold-restricted selection.
4. Genes are filtered to those detected in a minimum number of cells before HVG
   ranking. With 62,710 genes, a large fraction are all-zero and will otherwise
   distort the mean-variance fit.

Run `inspect_h5ad(path)` first and read the output before running the builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

RANDOM_STATE = 0


# --------------------------------------------------------------------------
# step 0: look before you leap
# --------------------------------------------------------------------------

def inspect_h5ad(path, n_preview=5):
    """Print what is actually in the file: obs columns, layers, and whether X
    looks like raw counts or something already normalised."""
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    print(f"shape: {a.shape[0]} cells x {a.shape[1]} genes")
    print(f"layers: {list(a.layers.keys()) or 'none'}")
    print(f"raw present: {a.raw is not None}")
    print("\nobs columns:")
    for c in a.obs.columns:
        nu = a.obs[c].nunique(dropna=True)
        sample = list(pd.Series(a.obs[c].dropna().unique()[:n_preview]).astype(str))
        print(f"  {c:30s} n_unique={nu:<6} e.g. {sample}")

    sub = a.X[:2000] if a.isbacked else a.X[:2000]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    frac_int = float(np.mean(np.isclose(sub, np.round(sub))))
    print(f"\nX preview: min={sub.min():.3f} max={sub.max():.3f} "
          f"mean={sub.mean():.3f} fraction_integer={frac_int:.3f}")
    if frac_int > 0.99 and sub.max() > 50:
        print("  -> looks like RAW COUNTS. Use already_lognorm=False.")
    elif sub.max() < 20:
        print("  -> looks ALREADY log-normalised. Use already_lognorm=True.")
    else:
        print("  -> ambiguous. Check how the notebook loaded this before deciding.")
    a.file.close()


# --------------------------------------------------------------------------
# step 1: pick genes
# --------------------------------------------------------------------------

def select_hvgs(path, n_hvg=2000, hvg_on="all", drug_key=None,
                control_label=None, max_cells=40000, already_lognorm=False,
                flavor="seurat", min_gene_cells=10, target_sum=1e4):
    """Choose HVGs from a subsample, so nothing large is held in memory.

    hvg_on="control" restricts selection to untreated wells; this keeps the
    perturbation signal out of the feature-selection step.
    """
    import anndata as ad
    import scanpy as sc

    a = ad.read_h5ad(path, backed="r")
    obs = a.obs

    if hvg_on == "control":
        if drug_key is None or control_label is None:
            raise ValueError("hvg_on='control' needs drug_key and control_label")
        mask = (obs[drug_key].astype(str) == str(control_label)).to_numpy()
        if mask.sum() == 0:
            raise ValueError(
                f"no cells with {drug_key} == {control_label!r}; "
                f"check inspect_h5ad output for the real control label"
            )
        pool = np.flatnonzero(mask)
        print(f"HVG selection on {len(pool)} control cells")
    elif hvg_on == "all":
        pool = np.arange(a.n_obs)
        print(f"HVG selection on all {len(pool)} cells -- declare this as a "
              f"global step in the writeup")
    else:
        raise ValueError("hvg_on must be 'control' or 'all'")

    rng = np.random.default_rng(RANDOM_STATE)
    if len(pool) > max_cells:
        pool = np.sort(rng.choice(pool, max_cells, replace=False))

    sub = a[pool].to_memory()
    a.file.close()

    # 62,710 genes includes a long tail never detected in this subsample;
    # leaving them in skews the mean-variance fit that HVG ranking relies on
    n_before = sub.n_vars
    sc.pp.filter_genes(sub, min_cells=min_gene_cells)
    print(f"gene filter: {sub.n_vars}/{n_before} genes detected in "
          f">= {min_gene_cells} cells")

    if not already_lognorm:
        sc.pp.normalize_total(sub, target_sum=target_sum)
        sc.pp.log1p(sub)
    sc.pp.highly_variable_genes(sub, n_top_genes=min(n_hvg, sub.n_vars),
                                flavor=flavor)

    genes = sub.var_names[sub.var["highly_variable"]].to_numpy()
    print(f"selected {len(genes)} HVGs")
    return genes


# --------------------------------------------------------------------------
# step 2: aggregate
# --------------------------------------------------------------------------

def build_drug_pseudobulk(path, drug_key, genes=None, already_lognorm=False,
                          min_cells=20, chunk_size=5000, target_sum=1e4,
                          layer=None):
    """Mean log-normalised expression per drug, chunked and backed.

    The library size comes from sparse row sums, so each chunk is only
    densified across the retained HVG columns. Peak memory is roughly
    chunk_size * len(genes) * 4 bytes -- about 40 MB at the defaults, rather
    than the ~1.25 GB a full-width densification would need. That is what makes
    this viable on a login node when compute nodes are unavailable.

    Returns (matrix, n_cells) where matrix is a DataFrame indexed by drug.
    """
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    var_names = a.var_names.to_numpy()
    if genes is None:
        gene_idx = np.arange(len(var_names))
    else:
        pos = pd.Index(var_names).get_indexer(pd.Index(genes))
        missing = int((pos < 0).sum())
        if missing:
            print(f"warning: {missing} requested genes absent from var_names, dropped")
        gene_idx = pos[pos >= 0]
    kept_genes = var_names[gene_idx]

    labels = a.obs[drug_key].astype(str).to_numpy()
    uniq = pd.Index(pd.unique(labels)).sort_values()
    code = pd.Index(uniq).get_indexer(labels)

    sums = np.zeros((len(uniq), len(gene_idx)), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.int64)

    n = a.n_obs
    n_keep = len(gene_idx)
    est_gb = chunk_size * n_keep * 4 / 1e9
    print(f"streaming {n} cells in chunks of {chunk_size} "
          f"(~{est_gb:.3f} GB peak per chunk, {n_keep} of {a.n_vars} genes kept)")
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        blk = a.X[start:stop] if layer is None else a.layers[layer][start:stop]

        if sp.issparse(blk):
            # library size from sparse row sums -- no need to densify all genes
            tot = np.asarray(blk.sum(axis=1)).ravel()
            dense = np.asarray(blk[:, gene_idx].todense(), dtype=np.float32)
        else:
            blk = np.asarray(blk, dtype=np.float32)
            tot = blk.sum(axis=1)
            dense = blk[:, gene_idx]

        if not already_lognorm:
            tot[tot == 0] = 1.0
            dense = np.log1p(dense / tot[:, None] * target_sum)

        c = code[start:stop]
        np.add.at(sums, c, dense.astype(np.float64, copy=False))
        np.add.at(counts, c, 1)
        del dense, blk
        if (start // chunk_size) % 20 == 0:
            print(f"  {stop}/{n} cells", flush=True)

    a.file.close()

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]

    df = pd.DataFrame(means, index=uniq, columns=kept_genes)
    n_cells = pd.Series(counts, index=uniq, name="n_cells")

    thin = n_cells[n_cells < min_cells]
    if len(thin):
        print(f"dropping {len(thin)} drugs with < {min_cells} cells: "
              f"{list(thin.index[:10])}{' ...' if len(thin) > 10 else ''}")
        df = df.drop(index=thin.index)
        n_cells = n_cells.drop(index=thin.index)

    print(f"pseudobulk: {df.shape[0]} drugs x {df.shape[1]} genes")
    print(f"cells per drug: median {int(n_cells.median())}, "
          f"min {int(n_cells.min())}, max {int(n_cells.max())}")
    return df, n_cells


# --------------------------------------------------------------------------
# step 2b: is sequencing depth per drug driving the geometry?
# --------------------------------------------------------------------------

def cell_count_diagnostics(pseudobulk, n_cells, n_pcs=5):
    """Check whether cells-per-drug predicts where a drug lands in expression space.

    Drugs with 900 cells and drugs with 8,000 cells have pseudobulk means with
    very different noise levels. If cell count correlates with the leading PCs,
    part of any apparent structure is sampling depth, not biology, and you need
    a sensitivity analysis (downsample every drug to a common cell count and
    confirm the result survives) before presenting it.
    """
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    idx = pseudobulk.index
    nc = pd.Series(n_cells).reindex(idx).to_numpy(dtype=float)
    X = StandardScaler().fit_transform(pseudobulk.to_numpy(np.float64))
    k = min(n_pcs, X.shape[0] - 1, X.shape[1])
    Z = PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(X)

    if np.unique(nc).size < 2:
        print("cells per drug is constant; depth confounding check not applicable")
        return pd.DataFrame(columns=["feature", "spearman_r", "p"])

    rows = []
    for i in range(k):
        r, p = spearmanr(nc, Z[:, i])
        rows.append({"feature": f"PC{i+1}", "spearman_r": float(r), "p": float(p)})
    r, p = spearmanr(nc, np.linalg.norm(X, axis=1))
    rows.append({"feature": "profile_norm", "spearman_r": float(r), "p": float(p)})

    out = pd.DataFrame(rows)
    print(f"\ncells per drug: median {np.median(nc):.0f}, "
          f"min {nc.min():.0f}, max {nc.max():.0f}, "
          f"ratio {nc.max()/max(nc.min(), 1):.1f}x")
    print(out.to_string(index=False))
    flagged = out[(out.p < 0.05) & (out.spearman_r.abs() > 0.3)]
    if len(flagged):
        print("  -> cell count tracks expression geometry; run the downsampling "
              "sensitivity check before reporting")
    else:
        print("  -> no strong depth confounding detected")
    return out


# --------------------------------------------------------------------------
# step 3: hand to the alignment code
# --------------------------------------------------------------------------

def align_blocks(pseudobulk, morph_raw, moa_by_drug, shared_compounds=None):
    """Intersect the three tables and return arrays in one consistent order."""
    idx = pseudobulk.index.intersection(morph_raw.index)
    if shared_compounds is not None:
        idx = idx.intersection(pd.Index(shared_compounds))
    moa = moa_by_drug.reindex(idx)
    idx = idx[moa.notna().to_numpy()]

    X_expr = pseudobulk.loc[idx].to_numpy(np.float64)
    X_morph = morph_raw.loc[idx].to_numpy(np.float64)
    y = moa_by_drug.reindex(idx).to_numpy()

    bad = ~np.isfinite(X_morph)
    if bad.any():
        print(f"warning: {bad.sum()} non-finite morphology values -> column medians")
        med = np.nanmedian(np.where(bad, np.nan, X_morph), axis=0)
        X_morph[bad] = np.take(med, np.where(bad)[1])

    print(f"aligned: {len(idx)} compounds | morph {X_morph.shape[1]}d "
          f"| expr {X_expr.shape[1]}d | {pd.Series(y).nunique()} MoA classes")
    return X_morph, X_expr, y, np.asarray(idx)
