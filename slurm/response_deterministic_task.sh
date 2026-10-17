#!/usr/bin/env bash
set -euo pipefail
index=${1:?development prompt index required}
case "$index" in 0|1) ;; *) exit 2 ;; esac
ROOT=/work/09032/saini_2/vista/Projects/Exploration/constancy_rope
R=/scratch/09032/saini_2/constancy_rope
PY=/work/09032/saini_2/vista/library/uv-envs/crope/bin/python
cd "$ROOT"
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export CROPE_TRANSFORMER=/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28
export CROPE_ROPE=sega_official
export TORCH_HOME=/scratch/09032/saini_2/tmp/cache XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache
export WANDB_DIR=$R/wandb WANDB_PROJECT=constancy-rope
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then export WANDB_MODE=offline; fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
out="$R/response_probe_deterministic/job${SLURM_JOB_ID:?allocated job required}/prompt${index}"
# Above-native smoke exercises SEGA allocation before 4K; failed no-op stops the chain.
"$PY" scripts/response_probe_deterministic.py --prompt-index "$index" --res 2048x2048 --steps 4 \
    --probe-steps 1 --out "$out/smoke"
"$PY" scripts/response_probe_deterministic.py --prompt-index "$index" --res 4096x4096 --steps 28 \
    --probe-steps 6,14,22 --out "$out/4k"
"$PY" scripts/response_analysis.py "$out/4k" --out "$out/summary.json"
