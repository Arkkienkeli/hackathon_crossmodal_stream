#!/usr/bin/env bash
# =============================================================================
# WS4A — ML stage as a SLURM ARRAY: one task per (target column, feature block).
# STANDALONE. Plain sbatch, own header, reads no slurm.env.
#
#   sbatch slurm/ws4a_ml_array.sbatch                                # baseline
#   sbatch slurm/ws4a_ml_array.sbatch --tuned --n-trials 20          # tuned
#   sbatch -A myacct -p zen5_dense slurm/ws4a_ml_array.sbatch --tuned --n-trials 20
#   sbatch slurm/ws4a_ml_array.sbatch --cell-line a549
#
# Then, when the array has ended (any state -- the merge reports what is missing):
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> slurm/ws4a_ml_merge.sbatch [--tuned]
#
# Anything after the script name that this file does not consume is passed to the
# ml stage of EVERY task (--n-trials, --models, --quick ...).
#
# EACH TASK builds the same work list itself (`ml --list-units`, ~2 s, deterministic,
# guards applied) and takes line SLURM_ARRAY_TASK_ID+1 of it. No wrapper, no shared
# state between tasks. Each task writes to <outputs>/ml/parts/<NNN>_<target>_<block>/.
#
# --array below is 0-29 because HepG2 has 30 usable units (6 targets x 5 blocks).
# For another cell line count first and override on the command line:
#   bash scripts/ws4a.sh ml --cell-line a549 --target all --list-units /tmp/u.tsv; wc -l /tmp/u.tsv
#   sbatch --array=0-$((N-1)) slurm/ws4a_ml_array.sbatch --cell-line a549
# A task whose index is past the end of the list exits 0 and says so.
#
# Sizing: the parallel unit inside a task is the outer CV fold (25 of them), so a
# task uses ~25 cores. 32 asked; 12 tasks fit one 384-core zen5_dense node. The
# slowest unit type (elastic_net on the 41,780-feature blocks) takes ~5 min untuned
# and ~20 min at 20 trials. --time is generous for that.
# =============================================================================
#SBATCH --job-name=ws4a-ml
#SBATCH --output=logs/ws4a-ml-%A_%a.out
#SBATCH --error=logs/ws4a-ml-%A_%a.err
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
# NO --mem: rejected on Sofia. Memory follows cores (~1.9 GB/core).

## ---- EDIT THESE, or pass -A / -p / --qos on the sbatch command line ----------
##SBATCH --account=CHANGE_ME
##SBATCH --partition=CHANGE_ME
##SBATCH --qos=CHANGE_ME

set -euo pipefail

# --- arguments ---------------------------------------------------------------
TUNED=0; CELL=hepg2; ML_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tuned) TUNED=1; shift ;;
        --cell-line) CELL="$2"; shift 2 ;;
        *) ML_ARGS+=("$1"); shift ;;
    esac
done

REPO="${WS4A_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIF="${WS4A_SIF:-$REPO/container/ws4a.sif}"
CONFIG=/work/configs/ws4a.yaml
[[ $TUNED -eq 1 ]] && CONFIG=/work/configs/ws4a_tuned.yaml
IDX="${SLURM_ARRAY_TASK_ID:?this is an array job: submit with sbatch, or set SLURM_ARRAY_TASK_ID to test one task}"

[[ -d "$REPO/scripts/ws4a" ]] || { echo "not the repo: $REPO (submit from the repo root)" >&2; exit 1; }
[[ -f "$SIF" ]] || { echo "no image at $SIF" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "apptainer not on PATH (module load?)" >&2; exit 1; }
RUN=(); command -v srun >/dev/null && RUN=(srun)

WS4A_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ws4a-${SLURM_JOB_ID:-$$}-${IDX}"
mkdir -p "$WS4A_TMP"; trap 'rm -rf "$WS4A_TMP"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

DATA_ARGS=(); ROOT_ARGS=()
if [[ -n "${WS4A_DATA:-}" ]]; then
    DATA_ARGS=(--bind "$WS4A_DATA":"$WS4A_DATA"); ROOT_ARGS=(--root "$WS4A_DATA")
fi

APP=(apptainer exec --cleanenv
     --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
     --env SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
     --bind "$REPO":/work --bind "$WS4A_TMP":"$WS4A_TMP"
     "${DATA_ARGS[@]+"${DATA_ARGS[@]}"}"
     --env TMPDIR="$WS4A_TMP" --pwd /work "$SIF")

# --- the work list, built by this task, and its own line of it ---------------
"${APP[@]}" python /work/scripts/ws4a_ml.py --config "$CONFIG" "${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}" \
    --cell-line "$CELL" --target all --list-units "$WS4A_TMP/units.tsv" > "$WS4A_TMP/list.log" 2>&1 \
    || { cat "$WS4A_TMP/list.log"; echo "could not build the work list" >&2; exit 1; }
N=$(wc -l < "$WS4A_TMP/units.tsv")
if [[ "$IDX" -ge "$N" ]]; then
    echo "task $IDX: only $N unit(s) for $CELL -- nothing to do (index past the end)"; exit 0
fi
LINE=$(sed -n "$((IDX + 1))p" "$WS4A_TMP/units.tsv")
TARGET="${LINE%%$'\t'*}"; BLOCK="${LINE#*$'\t'}"

# --- where this task writes: <paths.outputs>/ml/parts/<NNN>_<target>_<block> --
OUT_IN=$("${APP[@]}" python -c "import sys; sys.path.insert(0,'/work/scripts'); from ws4a import common as C
cfg=C.load_config('$CONFIG', ${WS4A_DATA:+'$WS4A_DATA'}${WS4A_DATA:-None}); print(C.outputs_dir(cfg,'ml'))" 2>/dev/null | tail -1)
PART_IN="$OUT_IN/parts/$(printf '%03d' "$IDX")_${TARGET}_${BLOCK}"
# the merge needs the list too; every task writes identical content, atomically
"${APP[@]}" sh -c "mkdir -p '$OUT_IN/parts' && cp '$WS4A_TMP/units.tsv' '$OUT_IN/units.tsv.$$' && mv -f '$OUT_IN/units.tsv.$$' '$OUT_IN/units.tsv'"

echo "=============================================================="
echo "task $IDX/$N on $(hostname): target=$TARGET block=$BLOCK cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "config=$CONFIG cell=$CELL ml args=[${ML_ARGS[*]-}]"
echo "-> $PART_IN"
echo "=============================================================="

"${RUN[@]+"${RUN[@]}"}" "${APP[@]}" python /work/scripts/ws4a_ml.py \
    --config "$CONFIG" "${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}" --cell-line "$CELL" \
    --target "$TARGET" --blocks "$BLOCK" --out "$PART_IN" "${ML_ARGS[@]+"${ML_ARGS[@]}"}"

echo "task $IDX done: $TARGET x $BLOCK"
