#!/usr/bin/env bash
# =============================================================================
# WS4A — merge the parts of an array run into ml_<cl>.csv + ml_summary_<cl>.csv.
# STANDALONE. Submit after the array, with a dependency so it waits:
#
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> slurm/ws4a_ml_merge.sbatch
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> slurm/ws4a_ml_merge.sbatch --tuned
#
# `afterany`, not `afterok`: a failed task must not stop the merge from telling
# you WHICH units are missing. The merge refuses to write a table with any unit
# missing; re-run that index and resubmit this:
#   sbatch --array=17 slurm/ws4a_ml_array.sbatch --tuned --n-trials 20
#
# Pass --allow-missing (forwarded to ws4a_merge.py) to merge what exists for a
# look -- not for reporting.
# =============================================================================
#SBATCH --job-name=ws4a-merge
#SBATCH --output=logs/ws4a-merge-%j.out
#SBATCH --error=logs/ws4a-merge-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

## ---- EDIT THESE, or pass -A / -p / --qos on the sbatch command line ----------
##SBATCH --account=CHANGE_ME
##SBATCH --partition=CHANGE_ME
##SBATCH --qos=CHANGE_ME

set -euo pipefail

TUNED=0; CELL=hepg2; MERGE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tuned) TUNED=1; shift ;;
        --cell-line) CELL="$2"; shift 2 ;;
        *) MERGE_ARGS+=("$1"); shift ;;
    esac
done

REPO="${WS4A_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIF="${WS4A_SIF:-$REPO/container/ws4a.sif}"
CONFIG=/work/configs/ws4a.yaml
[[ $TUNED -eq 1 ]] && CONFIG=/work/configs/ws4a_tuned.yaml
[[ -f "$SIF" ]] || { echo "no image at $SIF" >&2; exit 1; }

DATA_ARGS=()
[[ -n "${WS4A_DATA:-}" ]] && DATA_ARGS=(--bind "$WS4A_DATA":"$WS4A_DATA")
APP=(apptainer exec --cleanenv --bind "$REPO":/work "${DATA_ARGS[@]+"${DATA_ARGS[@]}"}" --pwd /work "$SIF")

OUT_IN=$("${APP[@]}" python -c "import sys; sys.path.insert(0,'/work/scripts'); from ws4a import common as C
cfg=C.load_config('$CONFIG', ${WS4A_DATA:+'$WS4A_DATA'}${WS4A_DATA:-None}); print(C.outputs_dir(cfg,'ml'))" 2>/dev/null | tail -1)

echo "merging $OUT_IN/parts  (config $CONFIG, cell $CELL)"
"${APP[@]}" python /work/scripts/ws4a_merge.py --parts "$OUT_IN/parts" --units "$OUT_IN/units.tsv" \
    --cell-line "$CELL" --out "$OUT_IN" "${MERGE_ARGS[@]+"${MERGE_ARGS[@]}"}"
