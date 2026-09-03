#!/usr/bin/env python3
"""
WS4 Task 2 pipeline: OpenScreen HepG2 morphology <-> TAHOE HepG2 transcriptomics

Purpose
-------
Build aligned drug-level morphology (M) and gene-expression (G) matrices and
measure whether drug-drug structure in morphology agrees with drug-drug
structure in transcriptomics.

Expected companion modules in the same repo:
    ws4_morph_qc.py
    ws4_pseudobulk.py

Run from the repository root, for example:
    python ws4_task2_pipeline.py

IMPORTANT
---------
Steps touching hepg2_cells.h5ad (inspection, HVG selection, pseudobulk) should
be run on a compute node, not a shared login node.

This script assumes raw counts ONLY after the inspection step confirms that.
If the TAHOE X matrix is already normalized/log-transformed, set
ALREADY_LOGNORM = True below before continuing.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from ws4_morph_qc import (
    load_sites,
    feature_consistency,
    to_matrix,
    cross_site_reproducibility,
    site_effect_check,
    within_site_reproducibility,
)

from ws4_pseudobulk import (
    inspect_h5ad,
    select_hvgs,
    build_drug_pseudobulk,
)


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("OpenScreen/data")
TAHOE_FILE = DATA_DIR / "hepg2_cells.h5ad"

HVG_FILE = DATA_DIR / "hepg2_hvg_2000.txt"
PB_FILE = DATA_DIR / "hepg2_pseudobulk_2000hvg.parquet"
CELL_COUNT_FILE = DATA_DIR / "hepg2_cells_per_drug.parquet"

MORPH_FILE = DATA_DIR / "hepg2_morphology_all_sites.parquet"
ALIGNED_M_FILE = DATA_DIR / "hepg2_task2_M.parquet"
ALIGNED_G_FILE = DATA_DIR / "hepg2_task2_G.parquet"

CROSS_SITE_FILE = DATA_DIR / "hepg2_cross_site_reproducibility.csv"
CROSS_SITE_PER_COMPOUND_FILE = DATA_DIR / "hepg2_cross_site_per_compound.csv"
WITHIN_SITE_FILE = DATA_DIR / "hepg2_within_site_reproducibility.csv"

COMPOUND_KEY_MORPH = "Metadata_Drug"
COMPOUND_KEY_EXPR = "drug"

N_HVG = 2000
HVG_MAX_CELLS = 40000
CHUNK_SIZE = 5000
MIN_CELLS_PER_DRUG = 20
N_PERM = 999
RANDOM_STATE = 0

# CHANGE THIS ONLY AFTER inspecting the TAHOE object.
# False = X contains raw counts and should be normalize_total + log1p.
# True  = X is already normalized/log-transformed.
ALREADY_LOGNORM = False


def heading(text: str) -> None:
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def norm_name(x) -> str:
    return str(x).strip().lower()


def require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


# ============================================================
# STEP 0 — INPUT CHECK
# ============================================================

heading("STEP 0 — CHECK INPUTS")

require(DATA_DIR, "OpenScreen data directory")
require(TAHOE_FILE, "TAHOE HepG2 expression file")

for site in ("FMP", "IMTM", "MEDINA"):
    require(
        DATA_DIR / f"morphology_{site}_HepG2_consensus.h5ad",
        f"{site} morphology consensus",
    )
    require(
        DATA_DIR / f"morphology_{site}_HepG2.h5ad",
        f"{site} morphology well-level file",
    )

print("Input directory:", DATA_DIR)
print("TAHOE:", TAHOE_FILE)
print("All expected OpenScreen morphology files found.")


# ============================================================
# STEP 1 — OPENSCREEN CONSENSUS MORPHOLOGY
# ============================================================

heading("STEP 1 — LOAD OPENSCREEN HEPG2 MORPHOLOGY")

print("""
INPUT
-----
OpenScreen/data/morphology_FMP_HepG2_consensus.h5ad
OpenScreen/data/morphology_IMTM_HepG2_consensus.h5ad
OpenScreen/data/morphology_MEDINA_HepG2_consensus.h5ad

Goal
----
Confirm all sites use the same morphology feature space and build
compound x morphology matrices.
""")

consensus = load_sites(DATA_DIR, consensus=True)
features = feature_consistency(consensus)

mats = {}
for site, adata in consensus.items():
    mats[site] = to_matrix(
        adata,
        features=features,
        compound_key=COMPOUND_KEY_MORPH,
    )
    print(
        f"{site:8s}: {mats[site].shape[0]} drugs x "
        f"{mats[site].shape[1]} morphology features"
    )


# ============================================================
# STEP 2 — CROSS-SITE QC
# ============================================================

heading("STEP 2 — CROSS-SITE MORPHOLOGY REPRODUCIBILITY")

pairs, per_compound = cross_site_reproducibility(mats)
site_stats = site_effect_check(mats)

pairs.to_csv(CROSS_SITE_FILE, index=False)
per_compound.to_csv(CROSS_SITE_PER_COMPOUND_FILE)

print("\nSaved:")
print(" ", CROSS_SITE_FILE)
print(" ", CROSS_SITE_PER_COMPOUND_FILE)


# ============================================================
# STEP 3 — WITHIN-SITE REPLICATE CEILING
# ============================================================

heading("STEP 3 — WITHIN-SITE REPLICATE REPRODUCIBILITY")

raw = load_sites(DATA_DIR, consensus=False)
within_results = {}

for site, adata in raw.items():
    print(f"\n{site}")
    within_results[site] = within_site_reproducibility(
        adata,
        compound_key=COMPOUND_KEY_MORPH,
        features=features,
    )

pd.DataFrame(within_results).T.to_csv(WITHIN_SITE_FILE)

print("\nSaved:", WITHIN_SITE_FILE)


# ============================================================
# STEP 4 — BUILD ONE MORPHOLOGY PROFILE PER DRUG
# ============================================================

heading("STEP 4 — BUILD ALL-SITE MORPHOLOGY CONSENSUS")

long_parts = []

for site, matrix in mats.items():
    tmp = matrix.copy()
    tmp["__site__"] = site
    tmp["__drug__"] = tmp.index.astype(str)
    long_parts.append(tmp.reset_index(drop=True))

stacked = pd.concat(long_parts, ignore_index=True)

feature_cols = [c for c in stacked.columns if c not in {"__site__", "__drug__"}]

morph_all = (
    stacked
    .groupby("__drug__", observed=True)[feature_cols]
    .mean()
    .sort_index()
)

n_sites = (
    stacked
    .groupby("__drug__", observed=True)["__site__"]
    .nunique()
    .sort_index()
)

print("Morphology consensus shape:", morph_all.shape)
print("\nNumber of sites contributing per drug:")
print(n_sites.value_counts().sort_index())

morph_all.to_parquet(MORPH_FILE)
print("\nSaved:", MORPH_FILE)


# ============================================================
# STEP 5 — INSPECT TAHOE EXPRESSION
# ============================================================

heading("STEP 5 — INSPECT TAHOE HEPG2 TRANSCRIPTOMICS")

print("""
INPUT
-----
OpenScreen/data/hepg2_cells.h5ad

Expected scale from the current notebook
----------------------------------------
~389,085 cells x ~62,710 genes

IMPORTANT
---------
One row in TAHOE is a CELL.
One row in morphology is a DRUG.

We therefore need to:
  1. confirm normalization status,
  2. select HVGs,
  3. aggregate cells by drug.
""")

inspect_h5ad(str(TAHOE_FILE))

print(f"""
Current setting:
    ALREADY_LOGNORM = {ALREADY_LOGNORM}

Before allowing Steps 6-7 to run, verify that this setting matches the
inspection above.

If X is raw counts:
    ALREADY_LOGNORM = False

If X is already normalized/log-transformed:
    ALREADY_LOGNORM = True
""")


# ============================================================
# STEP 6 — SELECT HVGs
# ============================================================

heading("STEP 6 — SELECT HIGHLY VARIABLE GENES")

if HVG_FILE.exists():
    print("Using existing HVG file:", HVG_FILE)
    genes = np.loadtxt(HVG_FILE, dtype=str)
    genes = np.atleast_1d(genes)
else:
    print(
        f"Selecting {N_HVG} HVGs using up to "
        f"{HVG_MAX_CELLS:,} cells..."
    )

    genes = select_hvgs(
        str(TAHOE_FILE),
        n_hvg=N_HVG,
        hvg_on="all",
        max_cells=HVG_MAX_CELLS,
        already_lognorm=ALREADY_LOGNORM,
        min_gene_cells=10,
    )

    np.savetxt(HVG_FILE, genes, fmt="%s")
    print("Saved:", HVG_FILE)

print("HVG count:", len(genes))


# ============================================================
# STEP 7 — CELLS -> DRUG-LEVEL PSEUDOBULK
# ============================================================

heading("STEP 7 — BUILD DRUG-LEVEL EXPRESSION PSEUDOBULK")

if PB_FILE.exists() and CELL_COUNT_FILE.exists():
    print("Using existing pseudobulk:", PB_FILE)
    pb = pd.read_parquet(PB_FILE)
    cell_df = pd.read_parquet(CELL_COUNT_FILE)
    if "n_cells" in cell_df.columns:
        n_cells = cell_df["n_cells"]
    else:
        n_cells = cell_df.iloc[:, 0]
else:
    pb, n_cells = build_drug_pseudobulk(
        str(TAHOE_FILE),
        drug_key=COMPOUND_KEY_EXPR,
        genes=genes,
        already_lognorm=ALREADY_LOGNORM,
        chunk_size=CHUNK_SIZE,
        min_cells=MIN_CELLS_PER_DRUG,
    )

    pb.to_parquet(PB_FILE)
    n_cells.to_frame("n_cells").to_parquet(CELL_COUNT_FILE)

    print("Saved:", PB_FILE)
    print("Saved:", CELL_COUNT_FILE)

print("\nExpression pseudobulk shape:", pb.shape)
print("\nCells per drug:")
print(n_cells.describe())


# ============================================================
# STEP 8 — MATCH MORPHOLOGY AND EXPRESSION
# ============================================================

heading("STEP 8 — MATCH M AND G BY DRUG IDENTITY")

morph_lookup = {norm_name(x): x for x in morph_all.index}
expr_lookup = {norm_name(x): x for x in pb.index}

shared_norm = sorted(set(morph_lookup) & set(expr_lookup))

morph_names = [morph_lookup[x] for x in shared_norm]
expr_names = [expr_lookup[x] for x in shared_norm]

M = morph_all.loc[morph_names].copy()
G = pb.loc[expr_names].copy()

M.index = shared_norm
G.index = shared_norm

assert M.shape[0] == G.shape[0]
assert list(M.index) == list(G.index)

print("Shared drugs:", len(shared_norm))
print("M morphology:", M.shape)
print("G expression:", G.shape)

only_m = sorted(set(morph_lookup) - set(expr_lookup))
only_g = sorted(set(expr_lookup) - set(morph_lookup))

if only_m:
    print("\nMorphology-only drugs:", only_m[:20])
if only_g:
    print("\nExpression-only drugs:", only_g[:20])

M.to_parquet(ALIGNED_M_FILE)
G.to_parquet(ALIGNED_G_FILE)

print("\nSaved:")
print(" ", ALIGNED_M_FILE)
print(" ", ALIGNED_G_FILE)


# ============================================================
# STEP 9 — TASK 2: OVERALL M <-> G AGREEMENT
# ============================================================

heading("STEP 9 — TASK 2: MORPHOLOGY <-> GENE-EXPRESSION AGREEMENT")

print("""
Question
--------
If two drugs create similar morphology profiles, do they also create similar
gene-expression profiles?

Method
------
1. Standardize each feature block.
2. Compute drug x drug correlation-distance matrices for M and G.
3. Compare their upper triangles using Spearman correlation.
4. Use a Mantel-style drug-label permutation test.
""")

M_arr = np.asarray(M, dtype=np.float64)
G_arr = np.asarray(G, dtype=np.float64)

if not np.isfinite(M_arr).all():
    raise ValueError("M contains NaN/Inf values.")
if not np.isfinite(G_arr).all():
    raise ValueError("G contains NaN/Inf values.")

Mz = StandardScaler().fit_transform(M_arr)
Gz = StandardScaler().fit_transform(G_arr)

D_morph = squareform(pdist(Mz, metric="correlation"))
D_expr = squareform(pdist(Gz, metric="correlation"))

upper = np.triu_indices(len(shared_norm), k=1)

observed_r, _ = spearmanr(
    D_morph[upper],
    D_expr[upper],
)

print(f"Observed M-G distance correlation: {observed_r:.4f}")


# ============================================================
# STEP 10 — PERMUTATION TEST
# ============================================================

heading("STEP 10 — MANTEL-STYLE PERMUTATION TEST")

rng = np.random.default_rng(RANDOM_STATE)
null_r = np.empty(N_PERM, dtype=np.float64)

for i in range(N_PERM):
    perm = rng.permutation(len(shared_norm))
    D_expr_perm = D_expr[np.ix_(perm, perm)]

    null_r[i], _ = spearmanr(
        D_morph[upper],
        D_expr_perm[upper],
    )

p_perm = (
    1 + np.sum(np.abs(null_r) >= abs(observed_r))
) / (N_PERM + 1)

print(f"Observed correlation : {observed_r:.4f}")
print(f"Permutation p-value  : {p_perm:.4f}")
print(f"Null median r        : {np.median(null_r):.4f}")


# ============================================================
# FINAL SUMMARY
# ============================================================

heading("FINAL SUMMARY")

print(f"""
OPENSCREEN MORPHOLOGY
---------------------
3 acquisition sites
    -> site/replicate QC
    -> all-site compound consensus
    -> M = {M.shape[0]} matched drugs x {M.shape[1]} morphology features

TAHOE TRANSCRIPTOMICS
---------------------
single-cell expression
    -> normalization check
    -> {len(genes)} HVGs
    -> pseudobulk by drug
    -> G = {G.shape[0]} matched drugs x {G.shape[1]} genes

MATCH
-----
Shared compounds = {len(shared_norm)}

TASK 2 RESULT
-------------
M-G distance correlation = {observed_r:.4f}
Permutation p-value      = {p_perm:.4f}

Saved aligned matrices:
    {ALIGNED_M_FILE}
    {ALIGNED_G_FILE}
""")
