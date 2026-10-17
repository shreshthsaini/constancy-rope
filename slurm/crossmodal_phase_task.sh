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
unset CROPE_ROPE
export TORCH_HOME=/scratch/09032/saini_2/tmp/cache XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache
export WANDB_DIR=$R/wandb WANDB_PROJECT=constancy-rope
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then export WANDB_MODE=offline; fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
protocol_sha=${2:?frozen protocol SHA256 required}
out="$R/crossmodal_phase/job${SLURM_JOB_ID:?allocated job required}/prompt${index}"
"$PY" scripts/crossmodal_phase_scores.py --out "$out" --prompt-index "$index" \
    --protocol-sha256 "$protocol_sha" --preflight
"$PY" scripts/crossmodal_phase_pilot.py --prompt-index "$index" \
    --protocol .agent/notes/crossmodal_phase_protocol.json --protocol-sha256 "$protocol_sha" --out "$out"
"$PY" scripts/eval.py --root "$out/images" --prompts "$R/prompts/aesthetic4k_eval.jsonl" \
    --res 4096x4096 --reference-method baseline --out "$out/evaluation" \
    --metrics clip,pickscore,musiq,clipiqa --device cuda --threads 2 --bootstrap 0 --crop-seed 0 --crops 4
/work/09032/saini_2/vista/library/uv-envs/mend/bin/python scripts/imagereward_score.py \
    "$out/images" "$R/prompts/aesthetic4k_eval.jsonl" "$out/evaluation/imagereward.csv" --device cuda
"$PY" scripts/crossmodal_phase_scores.py --out "$out" --prompt-index "$index" --protocol-sha256 "$protocol_sha"
