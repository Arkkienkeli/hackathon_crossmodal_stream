#!/usr/bin/env bash
# =============================================================================
# WS4A — cross-modal integration, full pipeline.  STANDALONE SBATCH JOB.
#
# Submitted with plain sbatch. Reads no configs/slurm.env, sources nothing, and
# does not go through slurm/submit.sh — everything it needs is in this file, so
# what you submit is exactly what runs.
#
#   sbatch slurm/ws4a_pipeline.sbatch
#   sbatch slurm/ws4a_pipeline.sbatch --cell-line a549
#   sbatch --gres=NONE -p cpu slurm/ws4a_pipeline.sbatch --stage stats
#
# TUNED RUN (Optuna, inner CV loop only). Writes to a DIFFERENT output directory,
# so it never overwrites the untuned baseline it is meant to be compared against:
#   sbatch slurm/ws4a_pipeline.sbatch --tuned
#   sbatch slurm/ws4a_pipeline.sbatch --stage compare      # after both have run
#
# Anything after the script name is passed to EVERY stage that accepts it.
# Use --stage to run one part:  stats | ml | xai | plots | compare | all  (default all)
#
# EDIT THE #SBATCH BLOCK BELOW before the first submission — account and partition
# are cluster-specific and there is no sane default. Command-line flags override
# the directives in this file:
#
#   sbatch -A myacct -p gpu -t 08:00:00 slurm/ws4a_pipeline.sbatch
#
# NOTHING IN slurm/ HAS EVER BEEN RUN except ws4_batch_correction.sbatch. The
# container, the scripts and the data paths were verified on the workstation; the
# scheduler parts are untested.
# =============================================================================

#SBATCH --job-name=ws4a
#SBATCH --output=logs/ws4a-%j.out
#SBATCH --error=logs/ws4a-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00

# NO --mem. On this cluster (Sofia) memory is not requestable: it is preselected
# per node and allocated in proportion to the cores you ask for. Passing --mem is
# rejected. To get more RAM, ask for more CORES.

## ---- EDIT THESE: no default can be right for your cluster -------------------
##SBATCH --account=CHANGE_ME
##SBATCH --partition=CHANGE_ME
##SBATCH --qos=CHANGE_ME
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.org

# ---- NO GPU BY DEFAULT, and that is a measured decision ---------------------
# XGBoost is the only GPU consumer in this pipeline, and it was benchmarked at the
# real data sizes:
#     morphology 119 x 636     CPU 0.16s   GPU 0.14s
#     expression 119 x 41,780  CPU 9.31s   GPU 9.35s
#     50,000 x 636 (hypothetical cell-level)  CPU 4.07s  GPU 3.69s
# At ~100 rows there is nothing to parallelise on a GPU. Everything else -- Mantel,
# RV, PROTEST, AJIVE, permutation CCA, stability selection, SHAP -- has no GPU code
# path at all. Requesting an H200 would idle it through a CPU-bound run.
#
# Add one only if the single-cell tier is ever built:
#     sbatch --gres=gpu:1 -p zen4_h200 slurm/ws4a_pipeline.sbatch
#
# ---- SIZING on a zen5_dense node (2x 192-core EPYC 9965, 742 GB, 1.9 GB/core) --
# The dominant costs are embarrassingly parallel: 40 control replicates per modality
# pair (120 tasks) and the AJIVE permutation null (20 BLAS-heavy refits). The
# default above takes roughly a quarter of a node. To take a whole one:
#     sbatch --cpus-per-task=384 slurm/ws4a_pipeline.sbatch
#
# Cores are the ONLY sizing knob here -- memory follows them at ~1.9 GB/core, so 96
# cores implies ~180 GB and 384 implies ~740 GB. That is far more than this pipeline
# needs (a worker holds ~40 MB; even 384 of them is ~15 GB), so ask for cores to go
# faster, never to get memory.

set -euo pipefail

# --- arguments ---------------------------------------------------------------
STAGE=all
TUNED=0
PASS=()
ML_ONLY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --tuned) TUNED=1; shift ;;
        # Flags only ws4a_ml.py understands. PASS goes to EVERY stage, so putting
        # these there makes `stats` die on an unrecognised argument before the run
        # has produced anything.
        --n-trials|--models|--bias-sweep) ML_ONLY+=("$1" "$2"); shift 2 ;;
        --no-permuted-control|--quick) ML_ONLY+=("$1"); shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done

# --- where things are --------------------------------------------------------
# sbatch copies this file to a spool directory, so BASH_SOURCE cannot locate the
# repo. Submit from the repo root, or set WS4A_REPO explicitly.
REPO="${WS4A_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIF="${WS4A_SIF:-$REPO/container/ws4a.sif}"
CONFIG="${WS4A_CONFIG:-/work/configs/ws4a.yaml}"
# --tuned swaps in the overlay config: same data, same targets, same models, but
# Optuna inside the inner CV fold and a separate outputs directory. It is an overlay
# (`extends: ws4a.yaml`), so nothing has to be kept in sync by hand.
if [[ $TUNED -eq 1 ]]; then
    CONFIG="${WS4A_CONFIG:-/work/configs/ws4a_tuned.yaml}"
fi

# module load apptainer      # uncomment with the right name if your cluster needs it

echo "=============================================================="
echo "job        : ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo "repo       : $REPO"
echo "image      : $SIF"
echo "stage      : $STAGE"
echo "config     : $CONFIG$([[ $TUNED -eq 1 ]] && echo '   (TUNED — Optuna)')"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?}"
echo "args       : ${PASS[*]-<none>}"
echo "ml args    : ${ML_ONLY[*]-<none>}"
echo "=============================================================="

command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer not on PATH. Uncomment the 'module load' line above." >&2; exit 1; }
[[ -d "$REPO/scripts/ws4a" ]] || {
    echo "REPO does not look like the repo: $REPO (no scripts/ws4a/)." >&2
    echo "Submit from the repo root, or: sbatch --export=ALL,WS4A_REPO=/path ..." >&2; exit 1; }
[[ -f "$SIF" ]] || {
    echo "no WS4A image at $SIF" >&2
    echo "  build:  bash scripts/build_ws4a_container.sh" >&2
    echo "  stage:  rsync -avP --partial container/ws4a.sif <cluster>:$SIF" >&2
    echo "  or:     sbatch --export=ALL,WS4A_SIF=/scratch/\$USER/ws4a.sif ..." >&2; exit 1; }

# --- writable caches ---------------------------------------------------------
# The image is read-only and matplotlib hard-fails when it cannot write its cache.
# Prefer node-local SSD (1.4 TB on these nodes) over a shared filesystem.
WS4A_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ws4a-${SLURM_JOB_ID:-$$}"
mkdir -p "$WS4A_TMP"
trap 'rm -rf "$WS4A_TMP"' EXIT
echo "tmpdir     : $WS4A_TMP"

# --- data outside the repo ---------------------------------------------------
# A path outside $REPO is invisible inside the container unless bound. WS4A_DATA
# binds it at its own name and is passed to the scripts as --root.
DATA_ARGS=(); ROOT_ARGS=()
if [[ -n "${WS4A_DATA:-}" ]]; then
    [[ -d "$WS4A_DATA/LINCS" || -d "$WS4A_DATA/DATA_MAIN" || -d "$WS4A_DATA/OpenScreen" ]] || {
        echo "WS4A_DATA=$WS4A_DATA does not look like hackathon_crossmodal_stream" >&2
        echo "  expected one of LINCS/ DATA_MAIN/ OpenScreen/ beneath it" >&2; exit 1; }
    DATA_ARGS=(--bind "$WS4A_DATA":"$WS4A_DATA")
    ROOT_ARGS=(--root "$WS4A_DATA")
    echo "data       : $WS4A_DATA (bound)"
else
    echo "data       : from configs/ws4a.yaml, resolved under the repo"
fi

# --- GPU ---------------------------------------------------------------------
# Only XGBoost uses it. --nv on a driverless node makes apptainer fail outright,
# so add it only when this job actually got a GPU.
NV=()
if [[ -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}${SLURM_STEP_GPUS:-}" ]] \
   || { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
    NV=(--nv)
    echo "gpu        : allocated, passing --nv"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    echo "gpu        : none — XGBoost runs on CPU (correct, just slower)"
fi

# The scripts pin BLAS inside each worker themselves, but the PARENT process must
# not grab every core for its own BLAS pool before joblib starts -- that is the
# classic oversubscription that makes a parallel run slower than a serial one.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# --cleanenv keeps the host environment out, but it also strips the Slurm variables,
# so the allocation has to be handed in explicitly. Without this a 128-core job sees
# only the node's 384 cores and oversubscribes 3x.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

APP=(apptainer run --cleanenv "${NV[@]+"${NV[@]}"}"
     --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
     --env SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
     --bind "$REPO":/work --bind "$WS4A_TMP":"$WS4A_TMP"
     "${DATA_ARGS[@]+"${DATA_ARGS[@]}"}"
     --env TMPDIR="$WS4A_TMP" --pwd /work "$SIF")

COMMON=(--config "$CONFIG" "${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}" "${PASS[@]+"${PASS[@]}"}")

run_stage () {
    echo
    echo "----------------------------------------------------------------"
    echo ">>> $1"
    echo "----------------------------------------------------------------"
    srun "${APP[@]}" "$@"
}

# --- stage 0: pre-flight -----------------------------------------------------
# ~12 s, on synthetic data. It calls every vendored API exactly as the pipeline
# does and asserts the SHAPE of what comes back. A contract mismatch here would
# otherwise surface deep into a run, after the expensive part had completed --
# which is precisely how job 1349279 died at the AJIVE permutation null.
echo
echo "----------------------------------------------------------------"
echo ">>> pre-flight (vendored API contracts)"
echo "----------------------------------------------------------------"
if ! srun "${APP[@]}" python /work/scripts/ws4a_selftest.py; then
    echo "PRE-FLIGHT FAILED — not starting the pipeline." >&2
    echo "A call site disagrees with a vendored module; fix it before resubmitting." >&2
    exit 1
fi

# The compare stage reads two finished runs; it fits nothing and needs no data bind
# beyond the outputs. Directories are derived from the two configs' `paths.outputs`.
if [[ "$STAGE" == "compare" ]]; then
    BASE_OUT="${WS4A_BASELINE_DIR:-}"
    TUNED_OUT="${WS4A_TUNED_DIR:-}"
    if [[ -z "$BASE_OUT" || -z "$TUNED_OUT" ]]; then
        echo "--stage compare needs both directories, since only you know where the" >&2
        echo "two runs were written:" >&2
        echo "  sbatch --export=ALL,WS4A_BASELINE_DIR=/path/ws4a/ml,WS4A_TUNED_DIR=/path/ws4a_tuned/ml \\" >&2
        echo "         slurm/ws4a_pipeline.sbatch --stage compare" >&2
        exit 2
    fi
    run_stage compare "${COMMON[@]}" --baseline "$BASE_OUT" --tuned "$TUNED_OUT"
    exit 0
fi

case "$STAGE" in
    stats) run_stage stats "${COMMON[@]}" ;;
    ml)    run_stage ml    "${COMMON[@]}" "${ML_ONLY[@]+"${ML_ONLY[@]}"}" ;;
    xai)   run_stage xai   "${COMMON[@]}" ;;
    plots) run_stage plots "${COMMON[@]}" ;;
    all)
        # Order matters: plots read what stats and ml wrote.
        run_stage stats "${COMMON[@]}"
        run_stage ml    "${COMMON[@]}" --target all "${ML_ONLY[@]+"${ML_ONLY[@]}"}"
        run_stage plots "${COMMON[@]}"
        ;;
    *) echo "unknown --stage $STAGE (want stats|ml|xai|plots|compare|all)" >&2; exit 2 ;;
esac

echo
echo "=============================================================="
echo "results -> the outputs directory named in configs/ws4a.yaml"
echo "=============================================================="
