"""
Build a MuData object for HepG2 OpenScreen (morphology) + TAHOE (gex) data
combining morphology and gene-expression
"""

from pathlib import Path

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_FILE = DATA_DIR / "hepg2_mdata.h5mu"


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# --- morphology: average the three HepG2 replicates (FMP/IMTM/MEDINA) -------

fmp = ad.read_h5ad(DATA_DIR / "morphology_FMP_HepG2_consensus.h5ad")
imtm = ad.read_h5ad(DATA_DIR / "morphology_IMTM_HepG2_consensus.h5ad")
medina = ad.read_h5ad(DATA_DIR / "morphology_MEDINA_HepG2_consensus.h5ad")

# MEDINA has one extra drug ("Idarubicin (hydrochloride)") missing from
# FMP/IMTM; add it to both so all three sites cover the same drug set.
missing = medina.obs.Metadata_Drug[~medina.obs.Metadata_Drug.isin(fmp.obs.Metadata_Drug)].unique()
for drug in missing:
    extra = medina[medina.obs.Metadata_Drug == drug]
    fmp = ad.concat([fmp, extra], axis=0)
    imtm = ad.concat([imtm, extra], axis=0)

# Align by shared observations (drugs) and features, then average across sites.
common_obs = fmp.obs_names.intersection(imtm.obs_names).intersection(medina.obs_names)
common_var = fmp.var_names.intersection(imtm.var_names).intersection(medina.var_names)

fmp_c = fmp[common_obs, common_var]
imtm_c = imtm[common_obs, common_var]
medina_c = medina[common_obs, common_var]

mean_X = np.mean(
    np.stack([to_dense(fmp_c.X), to_dense(imtm_c.X), to_dense(medina_c.X)], axis=0),
    axis=0,
)

morph = ad.AnnData(X=mean_X, obs=fmp_c.obs.copy(), var=fmp_c.var.copy())
morph.obs.drop(columns=["Metadata_Site"], inplace=True)


# --- expression: keep gene-symbol features, DMSO-normalize ------------------

exp = ad.read_h5ad(DATA_DIR / "hepg2_cons.h5ad")

# Keep only gene-symbol-like variables: drop Ensembl IDs ("ENSG...")
# and purely numeric variable names.
var_mask = ~exp.var_names.str.startswith("ENSG", na=False) & ~exp.var_names.str.fullmatch(
    r"\d+(\.\d+)?", na=False
)
exp = exp[:, var_mask].copy()

dmso_mask = exp.obs["drug"] == "DMSO_TF"
if dmso_mask.sum() == 0:
    raise ValueError("No samples found with obs['drug'] == 'DMSO_TF'.")

X = to_dense(exp.X)
dmso_ref = X[dmso_mask.values, :].mean(axis=0)

exp.layers["raw_counts"] = exp.X.copy()

eps = 1e-8
exp.X = np.log2((X + eps) / (dmso_ref + eps))
exp.layers["log_counts"] = exp.X.copy()

exp.obs["pubchem_cid"] = pd.to_numeric(exp.obs["pubchem_cid"], errors="coerce").astype("Int64")


# --- combine into MuData, aligned by drug name -------------------------------

morph.obs_names = morph.obs["Metadata_Drug"].astype(str)
exp.obs_names = exp.obs["drug"].astype(str)

common = morph.obs_names.intersection(exp.obs_names)
morph = morph[common].copy()
exp = exp[common].copy()

# Use the (richer) expression obs for both modalities.
morph.obs = exp.obs.copy()

mdata = mu.MuData({"morph": morph, "gex": exp})
mdata.obs = exp.obs.copy()

print(mdata)


# --- save ---------------------------------------------------------------------

prev_allow = ad.settings.allow_write_nullable_strings
ad.settings.allow_write_nullable_strings = True
try:
    mdata.write(OUT_FILE)
finally:
    ad.settings.allow_write_nullable_strings = prev_allow
print(f"Saved MuData to: {OUT_FILE}")
