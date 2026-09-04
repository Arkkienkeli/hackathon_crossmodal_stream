#!/usr/bin/env bash
# Build the WS4A container from container/ws4a.def.
#
#   bash scripts/build_ws4a_container.sh
#   SIF=/scratch/$USER/ws4a.sif bash scripts/build_ws4a_container.sh
#   LOCK=1 bash scripts/build_ws4a_container.sh    # recompile the pinned lock first
#
# Must be runnable from anywhere; cd's to the repo root itself, because the def
# file's %files paths are relative to the build working directory.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SIF="${SIF:-$REPO/container/ws4a.sif}"

if [[ "${LOCK:-0}" == "1" ]]; then
    command -v uv >/dev/null || { echo "uv not on PATH; cannot recompile the lock." >&2; exit 1; }
    echo "recompiling container/ws4a-requirements.txt from .in"
    uv pip compile container/ws4a-requirements.in --python-version 3.12 \
        -o container/ws4a-requirements.txt
fi

[[ -f container/ws4a-requirements.txt ]] || {
    echo "container/ws4a-requirements.txt missing. Run with LOCK=1 to generate it." >&2
    exit 1
}

echo "packages: $(grep -c '==' container/ws4a-requirements.txt) pinned"
apptainer build --force "$SIF" container/ws4a.def

echo
echo "Built -> $SIF  ($(du -h "$SIF" | cut -f1))"
echo "Verify:  bash scripts/ws4a.sh test"
echo "Run:     bash scripts/ws4a.sh stats --cell-line hepg2"
