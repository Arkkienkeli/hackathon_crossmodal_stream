#!/usr/bin/env python3

from pathlib import Path
import glob
import os
import re
import gc
from itertools import combinations

import numpy as np
import pandas as pd
import anndata as ad

from pycytominer import normalize, feature_select
from pycytominer.cyto_utils import infer_cp_features
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path("OpenScreen/raw_source/extracted")
DATA_DIR = ROOT / "aggregated_data"
ANNOTATION_DIR = ROOT / "annotations"

OUT_DIR = Path("OpenScreen/sphering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SITES = [
    "FMP_HepG2",
    "IMTM_HepG2",
    "MEDINA_HepG2",
    "USC_HepG2",
]

# These are the three sites in our existing QC/H5AD comparison.
REPORT_SITES = [
    "FMP_HepG2",
    "IMTM_HepG2",
    "MEDINA_HepG2",
]

SITE_ANNOTATION_FILES = {
    "FMP_HepG2": "2022-07-08_Annotation_Bioactives_HepG2.csv",
    "IMTM_HepG2": "2023-08-14_Annotation2_IMTM_HepG2.csv",
    "MEDINA_HepG2": "2023-11-28_Annotation_MEDINA_HepG2.csv",
    "USC_HepG2": "2023-11-28_Annotation_USC_HepG2.csv",
}

EXISTING_H5AD = {
    "FMP_HepG2": "OpenScreen/data/morphology_FMP_HepG2.h5ad",
    "IMTM_HepG2": "OpenScreen/data/morphology_IMTM_HepG2.h5ad",
    "MEDINA_HepG2": "OpenScreen/data/morphology_MEDINA_HepG2.h5ad",
}

COMPARTMENTS = ["Cells", "Nuc", "Cyto"]

FEATURE_SELECT_OPS = [
    "variance_threshold",
    "frequency_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
]

RANDOM_STATE = 42
N_NULL = 10000
K_NEIGHBOURS = 5

PLATE_RE = re.compile(r"B\d{3,}")
REPLICATE_RE = re.compile(r"(?<![A-Za-z])R\d+")


# ============================================================
# HELPERS
# ============================================================

def parse_plate_replicate(filename):
    plate = PLATE_RE.search(filename)
    rep = REPLICATE_RE.search(filename)

    if plate is None or rep is None:
        raise ValueError(f"Could not parse plate/replicate: {filename}")

    return plate.group(), rep.group()


def load_site_profiles(site):
    site_dir = DATA_DIR / site
    files = sorted(glob.glob(str(site_dir / "*.csv")))

    if not files:
        raise FileNotFoundError(f"No CSV files for {site}")

    frames = []

    for f in files:
        d = pd.read_csv(f)

        feature_cols = [
            c for c in d.columns
            if not c.startswith("Metadata_")
        ]

        d[feature_cols] = d[feature_cols].astype("float32")

        plate, replicate = parse_plate_replicate(os.path.basename(f))

        d["Metadata_Site"] = site
        d["Metadata_PlateID"] = plate
        d["Metadata_Replicate"] = replicate
        d["Metadata_SourceFile"] = f"{site}_{plate}_{replicate}"

        frames.append(d)

    df = pd.concat(frames, ignore_index=True)

    print(
        f"{site}: {len(files)} files, "
        f"{df.shape[0]} wells, {df.shape[1]} columns"
    )

    return df


def normalize_per_plate(df, features):
    parts = []

    groups = list(df.groupby("Metadata_SourceFile", sort=False))

    for i, (_, group) in enumerate(groups, start=1):

        normalized = normalize(
            profiles=group,
            features=features,
            meta_features="infer",
            samples="all",
            method="mad_robustize",
        )

        parts.append(normalized)

        if i % 10 == 0 or i == len(groups):
            print(f"    MAD normalized {i}/{len(groups)} plates")

    return pd.concat(parts, ignore_index=True)


def load_well_annotation(site, eos_to_pdid_map):
    ann = pd.read_csv(
        ANNOTATION_DIR / SITE_ANNOTATION_FILES[site]
    )

    ann = ann.rename(
        columns={
            "Metadata_Plate": "Metadata_PlateID",
            "Metadata_Batch": "Metadata_Replicate",
        }
    )

    ann["Metadata_pdid"] = (
        ann["Metadata_EOS"]
        .map(eos_to_pdid_map)
        .astype("string")
    )

    ann["Metadata_negcon"] = (
        ann["Metadata_EOS"] == "DMSO"
    )

    return ann[
        [
            "Metadata_PlateID",
            "Metadata_Well",
            "Metadata_Replicate",
            "Metadata_pdid",
            "Metadata_negcon",
        ]
    ]


def load_existing_shared_pdids():
    result = {}

    for site, path in EXISTING_H5AD.items():
        a = ad.read_h5ad(path)

        vals = (
            a.obs["Metadata_pdid"]
            .dropna()
            .astype(str)
        )

        result[site] = set(vals)

        print(
            f"{site}: {len(result[site])} pdids "
            f"in existing H5AD"
        )

        del a

    return result


def apply_sphering(df, features, site):
    site_df = df[df["Metadata_Site"] == site].copy()

    n_dmso = int(site_df["Metadata_negcon"].sum())

    print(
        f"\nSphering {site}: "
        f"{len(site_df)} wells, {n_dmso} DMSO controls"
    )

    if n_dmso < 2:
        raise RuntimeError(
            f"{site}: insufficient DMSO controls"
        )

    meta_cols = [
        c for c in site_df.columns
        if c.startswith("Metadata_")
    ]

    result = normalize(
        profiles=site_df,
        features=features,
        meta_features=meta_cols,
        samples="Metadata_negcon == True",
        method="spherize",
        spherize_center=True,
        spherize_method="ZCA-cor",
        spherize_epsilon=1e-6,
    )

    return result


def shared_treatment_rows(df, site, allowed_pdids):
    x = df[
        (df["Metadata_Site"] == site)
        & (~df["Metadata_negcon"])
        & (df["Metadata_pdid"].notna())
    ].copy()

    x["Metadata_pdid"] = x["Metadata_pdid"].astype(str)

    x = x[
        x["Metadata_pdid"].isin(allowed_pdids)
    ].copy()

    return x


def make_consensus(df, site, allowed_pdids, features):
    x = shared_treatment_rows(
        df,
        site,
        allowed_pdids,
    )

    return (
        x.groupby("Metadata_pdid", observed=True)[features]
        .mean()
        .sort_index()
    )


def row_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)

    denom = np.sqrt(
        np.sum(a * a, axis=1)
        * np.sum(b * b, axis=1)
    )

    denom[denom == 0] = np.nan

    return np.sum(a * b, axis=1) / denom


def cross_site_qc(consensus, stage):
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []

    for s1, s2 in combinations(REPORT_SITES, 2):

        a = consensus[s1]
        b = consensus[s2]

        shared = sorted(
            set(a.index) & set(b.index)
        )

        A = a.loc[shared].to_numpy()
        B = b.loc[shared].to_numpy()

        matched = row_corr(A, B)

        null = []

        while len(null) < N_NULL:
            perm = rng.permutation(len(shared))
            vals = row_corr(A, B[perm])

            keep = perm != np.arange(len(shared))

            null.extend(
                vals[keep & np.isfinite(vals)].tolist()
            )

        null = np.asarray(null[:N_NULL])

        p95 = np.nanquantile(null, 0.95)

        rows.append(
            {
                "stage": stage,
                "pair": f"{s1} vs {s2}",
                "n_compounds": len(shared),
                "median_matched_r": np.nanmedian(matched),
                "median_null_r": np.nanmedian(null),
                "null_p95": p95,
                "percent_replicating":
                    np.mean(matched > p95),
            }
        )

    return pd.DataFrame(rows)


def within_site_qc(
    df,
    features,
    shared_pdids,
    stage,
):
    rng = np.random.default_rng(RANDOM_STATE)

    output = []

    for site in REPORT_SITES:

        x = shared_treatment_rows(
            df,
            site,
            shared_pdids[site],
        )

        within = []

        for pdid, group in x.groupby(
            "Metadata_pdid",
            observed=True,
        ):
            if len(group) < 2:
                continue

            values = group[features].to_numpy(
                dtype=np.float64
            )

            corr = np.corrcoef(values)

            iu = np.triu_indices(
                len(group),
                k=1,
            )

            within.append(
                np.nanmedian(corr[iu])
            )

        within = np.asarray(within)

        X = x[features].to_numpy(
            dtype=np.float64
        )

        labels = x["Metadata_pdid"].astype(str).to_numpy()

        null = []

        while len(null) < N_NULL:
            i = rng.integers(0, len(X), size=N_NULL)
            j = rng.integers(0, len(X), size=N_NULL)

            valid = (
                (i != j)
                & (labels[i] != labels[j])
            )

            if not np.any(valid):
                continue

            vals = row_corr(
                X[i[valid]],
                X[j[valid]],
            )

            null.extend(
                vals[np.isfinite(vals)].tolist()
            )

        null = np.asarray(null[:N_NULL])

        p95 = np.nanquantile(null, 0.95)

        output.append(
            {
                "stage": stage,
                "site": site,
                "n_compounds_with_replicates":
                    len(within),
                "median_within_site_r":
                    np.nanmedian(within),
                "median_null_r":
                    np.nanmedian(null),
                "null_p95": p95,
                "percent_replicating":
                    np.mean(within > p95),
            }
        )

    return pd.DataFrame(output)


def site_effect_qc(
    df,
    features,
    shared_pdids,
    stage,
):
    frames = []

    for site in REPORT_SITES:
        frames.append(
            shared_treatment_rows(
                df,
                site,
                shared_pdids[site],
            )
        )

    x = pd.concat(
        frames,
        ignore_index=True,
    )

    X = x[features].to_numpy(
        dtype=np.float64
    )

    X = StandardScaler().fit_transform(X)

    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBOURS + 1,
        metric="euclidean",
        n_jobs=-1,
    )

    nn.fit(X)

    idx = nn.kneighbors(
        X,
        return_distance=False,
    )[:, 1:]

    sites = x["Metadata_Site"].to_numpy()
    pdids = x["Metadata_pdid"].astype(str).to_numpy()

    same_site = (
        sites[idx] == sites[:, None]
    ).mean()

    same_compound = (
        pdids[idx] == pdids[:, None]
    ).mean()

    return {
        "stage": stage,
        "n_wells": len(x),
        "same_site_nn": same_site,
        "same_compound_nn": same_compound,
    }


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOADING RAW OPENSCREEN MORPHOLOGY")
print("=" * 80)

raw = {
    site: load_site_profiles(site)
    for site in ALL_SITES
}


# ============================================================
# 2. INFER CELL PAINTING FEATURES
# ============================================================

FEATURES = infer_cp_features(
    next(iter(raw.values())),
    compartments=COMPARTMENTS,
)

print(f"\nInferred Cell Painting features: {len(FEATURES)}")

if len(FEATURES) != 2977:
    print(
        "WARNING: original notebook reported "
        "2977 inferred features."
    )


# ============================================================
# 3. ORIGINAL PER-PLATE MAD ROBUSTIZATION
# ============================================================

print("\n" + "=" * 80)
print("2. PER-PLATE MAD ROBUSTIZATION")
print("=" * 80)

normalized = {}

for site in ALL_SITES:
    print(f"\n{site}")
    normalized[site] = normalize_per_plate(
        raw[site],
        FEATURES,
    )

del raw
gc.collect()

pooled = pd.concat(
    normalized.values(),
    ignore_index=True,
)

del normalized
gc.collect()

print(
    f"\nPooled morphology: "
    f"{pooled.shape[0]} wells"
)


# ============================================================
# 4. ORIGINAL POOLED FEATURE SELECTION
# ============================================================

print("\n" + "=" * 80)
print("3. FEATURE SELECTION")
print("=" * 80)

selected = feature_select(
    pooled,
    features=FEATURES,
    operation=FEATURE_SELECT_OPS,
)

del pooled
gc.collect()

morph_features = [
    c for c in selected.columns
    if not c.startswith("Metadata_")
]

print(
    f"Selected morphology features: "
    f"{len(morph_features)}"
)

if len(morph_features) != 636:
    print(
        "WARNING: original notebook reported "
        "636 selected features."
    )


# ============================================================
# 5. ATTACH DMSO / PDID ANNOTATIONS
# ============================================================

print("\n" + "=" * 80)
print("4. ATTACHING ANNOTATIONS")
print("=" * 80)

eos = pd.read_csv(
    ANNOTATION_DIR / "2024-08-02_EOS_pdid.csv"
)

eos_to_pdid_map = dict(
    zip(
        eos["Metadata_EOS"],
        eos["Metadata_pdid"],
    )
)

annotated_parts = []

for site, group in selected.groupby(
    "Metadata_Site",
    observed=True,
):
    ann = load_well_annotation(
        site,
        eos_to_pdid_map,
    )

    merged = group.merge(
        ann,
        on=[
            "Metadata_PlateID",
            "Metadata_Well",
            "Metadata_Replicate",
        ],
        how="left",
    )

    merged["Metadata_negcon"] = (
        merged["Metadata_negcon"]
        .fillna(False)
        .astype(bool)
    )

    merged["Metadata_pdid"] = (
        merged["Metadata_pdid"]
        .astype("string")
    )

    print(
        f"{site}: "
        f"{len(merged)} wells, "
        f"{merged['Metadata_negcon'].sum()} DMSO"
    )

    annotated_parts.append(merged)

baseline = pd.concat(
    annotated_parts,
    ignore_index=True,
)

del selected, annotated_parts
gc.collect()

baseline.to_parquet(
    OUT_DIR / "hepg2_mad_selected.parquet",
    index=False,
)

print(
    "\nSaved:",
    OUT_DIR / "hepg2_mad_selected.parquet",
)


# ============================================================
# 6. SPHERING USING DMSO REFERENCE — PER SITE
# ============================================================

print("\n" + "=" * 80)
print("5. DMSO-REFERENCED SPHERING")
print("=" * 80)

sphered_parts = []

for site in ALL_SITES:
    sphered_parts.append(
        apply_sphering(
            baseline,
            morph_features,
            site,
        )
    )

sphered = pd.concat(
    sphered_parts,
    ignore_index=True,
)

del sphered_parts
gc.collect()

sphered.to_parquet(
    OUT_DIR / "hepg2_mad_sphered_dmso_zcacor.parquet",
    index=False,
)

print(
    "\nSaved:",
    OUT_DIR / "hepg2_mad_sphered_dmso_zcacor.parquet",
)


# ============================================================
# 7. GET EXACT COMPOUND SUBSET USED IN EXISTING H5ADS
# ============================================================

print("\n" + "=" * 80)
print("6. MATCHING EXISTING 119-COMPOUND SUBSET")
print("=" * 80)

shared_pdids = load_existing_shared_pdids()

for site in REPORT_SITES:
    n_base = len(
        shared_treatment_rows(
            baseline,
            site,
            shared_pdids[site],
        )
    )

    n_sph = len(
        shared_treatment_rows(
            sphered,
            site,
            shared_pdids[site],
        )
    )

    print(
        f"{site}: baseline={n_base}, "
        f"sphered={n_sph}"
    )


# ============================================================
# 8. DRUG CONSENSUS
# ============================================================

baseline_consensus = {}
sphered_consensus = {}

for site in REPORT_SITES:

    baseline_consensus[site] = make_consensus(
        baseline,
        site,
        shared_pdids[site],
        morph_features,
    )

    sphered_consensus[site] = make_consensus(
        sphered,
        site,
        shared_pdids[site],
        morph_features,
    )

    baseline_consensus[site].to_parquet(
        OUT_DIR /
        f"{site}_baseline_consensus.parquet"
    )

    sphered_consensus[site].to_parquet(
        OUT_DIR /
        f"{site}_sphered_consensus.parquet"
    )


# ============================================================
# 9. WITHIN-SITE REPRODUCIBILITY
# ============================================================

print("\n" + "=" * 80)
print("7. WITHIN-SITE QC")
print("=" * 80)

within_before = within_site_qc(
    baseline,
    morph_features,
    shared_pdids,
    "baseline",
)

within_after = within_site_qc(
    sphered,
    morph_features,
    shared_pdids,
    "sphered",
)

within = pd.concat(
    [within_before, within_after],
    ignore_index=True,
)

print(within.to_string(index=False))

within.to_csv(
    OUT_DIR / "within_site_before_after.csv",
    index=False,
)


# ============================================================
# 10. CROSS-SITE REPRODUCIBILITY
# ============================================================

print("\n" + "=" * 80)
print("8. CROSS-SITE QC")
print("=" * 80)

cross_before = cross_site_qc(
    baseline_consensus,
    "baseline",
)

cross_after = cross_site_qc(
    sphered_consensus,
    "sphered",
)

cross = pd.concat(
    [cross_before, cross_after],
    ignore_index=True,
)

print(cross.to_string(index=False))

cross.to_csv(
    OUT_DIR / "cross_site_before_after.csv",
    index=False,
)


# ============================================================
# 11. SITE-EFFECT QC
# ============================================================

print("\n" + "=" * 80)
print("9. SITE-EFFECT QC")
print("=" * 80)

site_effect = pd.DataFrame(
    [
        site_effect_qc(
            baseline,
            morph_features,
            shared_pdids,
            "baseline",
        ),
        site_effect_qc(
            sphered,
            morph_features,
            shared_pdids,
            "sphered",
        ),
    ]
)

print(site_effect.to_string(index=False))

site_effect.to_csv(
    OUT_DIR / "site_effect_before_after.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"""
Outputs:

{OUT_DIR}/hepg2_mad_selected.parquet
{OUT_DIR}/hepg2_mad_sphered_dmso_zcacor.parquet

{OUT_DIR}/within_site_before_after.csv
{OUT_DIR}/cross_site_before_after.csv
{OUT_DIR}/site_effect_before_after.csv

Plus baseline and sphered per-site drug consensus parquet files.

Main question:

Did sphering LOWER same-site neighbour enrichment
without destroying within-site or cross-site drug reproducibility?
"""
)
