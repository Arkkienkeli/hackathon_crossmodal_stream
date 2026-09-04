"""Gather Tox21 plate-level RData objects into a single DataFrame.

Reads every ``*.RData`` under ``OpenScreen/data/plate_level_data/``. Each file
holds one CEBS plate-level table (object name ``cebs``) for a Tox21 real-time
viability screen. Column definitions are documented in
``plate_level_readme_original.xlsx``.

Filename pattern
----------------
``tox21-rt-viability-{cell_line}-{assay}-p{plate}_{timepoint}.RData``

Examples: ``hepg2`` / ``hek293``, ``flor`` / ``glo``, ``p1``, ``24h``.

Usage
-----
From the repo root or ``OpenScreen/``::

    python OpenScreen/gather_plate_level_data.py
    python OpenScreen/gather_plate_level_data.py --hepg2-only
    python OpenScreen/gather_plate_level_data.py --format parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyreadr

DATA_DIR = Path(__file__).resolve().parent / "data" / "plate_level_data"
FILENAME_RE = re.compile(
    r"^tox21-rt-viability-"
    r"(?P<cell_line>[a-z0-9]+)-"
    r"(?P<assay>flor|glo)-"
    r"p(?P<plate>\d+)_"
    r"(?P<timepoint>\d+h)\.RData$",
    re.IGNORECASE,
)


def parse_filename(path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized plate-level RData filename: {path.name}")
    fields = match.groupdict()
    return {
        "source_file": path.name,
        "cell_line": fields["cell_line"].lower(),
        "assay": fields["assay"].lower(),
        "plate": int(fields["plate"]),
        "timepoint": fields["timepoint"].lower(),
        "timepoint_h": int(fields["timepoint"][:-1]),
    }


# Columns that should be numeric but are sometimes stored as strings in a
# subset of the Tox21 RData exports (e.g. curvep_slope = "0").
KEEP_AS_STRING = {
    "CAS",
    "Tox21.ID",
    "Tox21AgencyID",
    "uniqueID",
    "Chemical.Name",
    "Chemical.ID",
    "pathway",
    "readout",
    "Library",
    "Cmpd_Library",
    "input_mask",
    "curvep_remark",
    "curvep_mask",
    "Mask.Flags",
    "source_file",
    "cell_line",
    "assay",
    "timepoint",
}


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in KEEP_AS_STRING:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Only replace if conversion recovers a usable numeric column.
        if coerced.notna().any() or df[col].isna().all():
            df[col] = coerced
    return df


def load_rdata(path: Path) -> pd.DataFrame:
    objects = pyreadr.read_r(str(path))
    if "cebs" not in objects:
        raise KeyError(f"{path.name}: expected object 'cebs', found {list(objects)}")
    df = objects["cebs"].copy()
    meta = parse_filename(path)
    for key, value in meta.items():
        df[key] = value
    return coerce_numeric_columns(df)


def gather_plate_level_data(
    data_dir: Path = DATA_DIR,
    *,
    cell_line: str | None = None,
) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.RData"))
    if not files:
        raise FileNotFoundError(f"No *.RData files in {data_dir}")

    frames = [load_rdata(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = coerce_numeric_columns(df)

    # Put provenance / screen metadata first for easier browsing.
    front = [
        "source_file",
        "cell_line",
        "assay",
        "plate",
        "timepoint",
        "timepoint_h",
        "pathway",
        "readout",
        "CAS",
        "Tox21.ID",
        "Chemical.Name",
        "Chemical.ID",
    ]
    ordered = front + [c for c in df.columns if c not in front]
    df = df[ordered]

    if cell_line is not None:
        df = df.loc[df["cell_line"] == cell_line.lower()].reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing Tox21 *.RData files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <data-dir>/tox21_rt_viability_plate_level.<ext>)",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--hepg2-only",
        action="store_true",
        help="Keep only HepG2 rows (relevant to the OpenScreen notebooks)",
    )
    args = parser.parse_args()

    cell_line = "hepg2" if args.hepg2_only else None
    df = gather_plate_level_data(args.data_dir, cell_line=cell_line)

    out = args.output
    if out is None:
        stem = "tox21_rt_viability_plate_level"
        if cell_line:
            stem = f"{stem}_{cell_line}"
        out = args.data_dir / f"{stem}.{args.format}"

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(f"  rows={len(df):,}  cols={df.shape[1]}")
    print(f"  cell_line: {sorted(df['cell_line'].unique())}")
    print(f"  assay: {sorted(df['assay'].unique())}")
    print(f"  timepoint: {sorted(df['timepoint'].unique(), key=lambda t: int(t[:-1]))}")
    print(f"  unique chemicals (CAS): {df['CAS'].nunique():,}")


if __name__ == "__main__":
    main()
