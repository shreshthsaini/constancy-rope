#!/usr/bin/env bash
set -euo pipefail
method=$1
case "$method" in sega_official|sega_rms_uniform|sega_rms_normalized|sega_ctm_lag) ;; *) exit 2 ;; esac
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
mkdir -p "$R/eval_krea/ir"
"$PY" scripts/dev3_generate.py --method "$method"
"$PY" scripts/eval.py --root "$R/out_krea/dev3_$method" \
    --prompts "$R/prompts/aesthetic4k_eval.jsonl" --res 4096x4096 \
    --out "$R/eval_krea/dev3_$method" --metrics clip,pickscore,musiq,clipiqa --device cuda
/work/09032/saini_2/vista/library/uv-envs/mend/bin/python scripts/imagereward_score.py \
    "$R/out_krea/dev3_$method" "$R/prompts/aesthetic4k_eval.jsonl" \
    "$R/eval_krea/ir/dev3_$method.csv" --device cuda
