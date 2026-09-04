#!/usr/bin/env bash
# One entry point for the WS4A container (cross-modal integration).
#
#   bash scripts/ws4a.sh stats --cell-line hepg2
#   bash scripts/ws4a.sh ml    --cell-line hepg2 --target toxicity
#   bash scripts/ws4a.sh xai   --cell-line hepg2 --target tox_cardiotoxicity
#   bash scripts/ws4a.sh plots --cell-line hepg2
#   bash scripts/ws4a.sh ml      --config /work/configs/ws4a_tuned.yaml   # Optuna
#   bash scripts/ws4a.sh compare --baseline <dir> --tuned <dir> --cell-line hepg2
#   bash scripts/ws4a.sh selftest                   # vendored API contracts (~12 s)
#   bash scripts/ws4a.sh test                       # apptainer test on the image
#   bash scripts/ws4a.sh python -c 'import xgboost; print(xgboost.__version__)'
#
# Knobs:
#   WS4A_SIF=/path.sif   use a specific image        (default container/ws4a.sif)
#   WS4A_DATA=/path      bind a data root outside the repo and pass --root
#   WS4A_GPU=0|1|auto    --nv passthrough            (default auto; only XGBoost uses it)
#   WS4A_TMPDIR=/scratch where matplotlib caches go  (default $TMPDIR)
#
# This is the WS4A sibling of run.sh and ws4.sh. Three separate images on purpose;
# see container/ws4a.def for why this one carries no deep-learning stack.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${WS4A_SIF:-$REPO/container/ws4a.sif}"

if [[ $# -eq 0 ]]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi

if [[ ! -f "$SIF" ]]; then
    cat >&2 <<MSG
No WS4A image at: $SIF

Build it:
    bash scripts/build_ws4a_container.sh

Or point at one you copied in:
    WS4A_SIF=/scratch/\$USER/ws4a.sif bash scripts/ws4a.sh "\$@"
MSG
    exit 1
fi

BINDS=(--bind "$REPO":/work)
EXTRA=()
if [[ -n "${WS4A_DATA:-}" ]]; then
    [[ -d "$WS4A_DATA" ]] || { echo "WS4A_DATA does not exist: $WS4A_DATA" >&2; exit 1; }
    BINDS+=(--bind "$WS4A_DATA":"$WS4A_DATA")
    EXTRA+=(--root "$WS4A_DATA")
fi

ENVS=()
if [[ -n "${WS4A_TMPDIR:-}" ]]; then
    mkdir -p "$WS4A_TMPDIR"
    BINDS+=(--bind "$WS4A_TMPDIR":"$WS4A_TMPDIR")
    ENVS+=(--env "TMPDIR=$WS4A_TMPDIR")
fi

# --nv exposes the host driver; without it the GPU is invisible whatever the image
# holds. Passing it on a node with no driver makes apptainer fail, so detect first.
NV=()
case "${WS4A_GPU:-auto}" in
    1) NV=(--nv) ;;
    0) : ;;
    *) if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
           NV=(--nv)
       fi ;;
esac

COMMON=(--cleanenv "${ENVS[@]+"${ENVS[@]}"}" "${BINDS[@]}" "${NV[@]+"${NV[@]}"}" --pwd /work)

case "$1" in
    test)
        # Bind the repo so %test can also exercise the VENDORED modules in
        # scripts/ws4a/, which live in the repo rather than in the image. Without
        # the bind those assertions are skipped and the test says so.
        exec apptainer test --bind "$REPO":/work "${NV[@]+"${NV[@]}"}" "$SIF" ;;
    shell) exec apptainer run "${COMMON[@]}" "$SIF" shell ;;
    selftest) exec apptainer exec "${COMMON[@]}" "$SIF" python /work/scripts/ws4a_selftest.py ;;
    stats|ml|xai|plots|compare|report|crossmodal|hvg)
        sub="$1"; shift
        exec apptainer run "${COMMON[@]}" "$SIF" "$sub" "${EXTRA[@]+"${EXTRA[@]}"}" "$@" ;;
    *) exec apptainer exec "${COMMON[@]}" "$SIF" "$@" ;;
esac
