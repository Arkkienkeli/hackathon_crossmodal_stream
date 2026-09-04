#!/usr/bin/env python3
"""WS4A — merge the parts of a sharded (array-job) ML run into the standard tables.

    apptainer run ws4a.sif python /work/scripts/ws4a_merge.py \
        --parts <outputs>/ml/parts --cell-line hepg2 [--out <outputs>/ml]

Each array task wrote its own <parts>/<NNN>_<target>_<block>/ml_<cl>.csv. This
concatenates them and writes ml_<cl>.csv + ml_summary_<cl>.csv exactly as a
single-job run would (same write_tables function), so compare / plots / xai read the
merged run without knowing it was sharded.

It refuses to produce a table that silently lacks units: every line of the work
list must have a part, or the merge aborts naming the missing ones. A partial merge
looks like a finished run and is the worst outcome for a deadline, not the best.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402
import ws4a_ml as ML                              # noqa: E402

LOG = C.LOG


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", required=True, help="directory holding one subdir per unit")
    ap.add_argument("--cell-line", default="hepg2")
    ap.add_argument("--out", default=None, help="default: the parent of --parts")
    ap.add_argument("--units", default=None,
                    help="the work list the array ran (units.tsv); every unit must "
                         "have a part or the merge aborts")
    ap.add_argument("--allow-missing", action="store_true",
                    help="merge what exists and WARN about the rest (not for reporting)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    C.setup_logging(args.log_level)

    parts = Path(args.parts)
    out = Path(args.out) if args.out else parts.parent
    out.mkdir(parents=True, exist_ok=True)

    frames, have = [], set()
    for d in sorted(parts.iterdir()):
        f = d / f"ml_{args.cell_line}.csv"
        if d.is_dir() and f.exists():
            df = pd.read_csv(f)
            frames.append(df)
            have |= set(zip(df.target, df.block))
            LOG.info("part       : %-40s %3d rows", d.name, len(df))
        elif d.is_dir():
            LOG.warning("part       : %-40s NO ml_%s.csv (task failed or still running)",
                        d.name, args.cell_line)

    if args.units:
        want = set()
        for line in Path(args.units).read_text().splitlines():
            if line.strip():
                col, blk = line.rstrip("\n").split("\t")
                want.add((col, blk))
        missing = sorted(want - have)
        if missing:
            level = LOG.warning if args.allow_missing else LOG.error
            level("%d of %d unit(s) have no result:", len(missing), len(want))
            for col, blk in missing:
                level("   %-28s %s", col, blk)
            if not args.allow_missing:
                LOG.error("refusing to write a table that looks complete and is not. "
                          "Re-run the failed tasks, or pass --allow-missing.")
                return 1

    if not frames:
        LOG.error("no parts under %s", parts)
        return 1

    df = pd.concat(frames, ignore_index=True)
    key = ["target", "block", "model", "permuted"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        LOG.warning("%d duplicated rows across parts (same unit run twice?) -- keeping "
                    "the LAST", int(dup.sum()))
        df = df.drop_duplicates(key, keep="last")

    merged = ML.write_tables(df, out, args.cell_line)

    # skipped_targets from any part are the same list; take the first that exists
    for d in sorted(parts.iterdir()):
        sk = d / f"skipped_targets_{args.cell_line}.csv"
        if sk.exists():
            C.save_table(pd.read_csv(sk), out / sk.name)
            break

    print(f"\nmerged {len(frames)} part(s) -> {len(df)} rows, "
          f"{merged.target.nunique()} target(s), {merged.block.nunique()} block(s), "
          f"{merged.model.nunique()} model(s)")
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
