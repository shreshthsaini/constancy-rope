#!/usr/bin/env bash
set -euo pipefail
method=$1
start=$2
end=$3
cd /work/09032/saini_2/vista/Projects/Exploration/constancy_rope
R=/scratch/09032/saini_2/constancy_rope
PY=/work/09032/saini_2/vista/library/uv-envs/crope/bin/python
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export CROPE_TRANSFORMER=/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28
export CROPE_ROPE=sega_official
export TORCH_HOME=/scratch/09032/saini_2/tmp/cache XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache
export WANDB_DIR=$R/wandb WANDB_PROJECT=constancy-rope
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then export WANDB_MODE=offline; fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" scripts/confirmation_generate.py --method "$method" --start "$start" --end "$end"
