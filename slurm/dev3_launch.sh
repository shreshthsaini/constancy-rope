#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/09032/saini_2/vista/Projects/Exploration/constancy_rope
R=/scratch/09032/saini_2/constancy_rope
count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$count" -eq 1 || "$count" -eq 4 ]] || { echo "Expected one or four GPUs; found $count"; exit 2; }
/work/09032/saini_2/vista/library/bin/gpu_telemetry.sh "$R/telemetry" 30 &
telemetry_pid=$!
pids=()
for ((gpu=0; gpu<count; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu bash "$ROOT/slurm/dev3_worker.sh" &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
# Signal only the recorded logger child, after verifying its live identity.
if ps -o pid,ppid,lstart,stat,args -p "$telemetry_pid"; then
    parent=$(ps -o ppid= -p "$telemetry_pid")
    if [[ "${parent// /}" == "$$" ]]; then kill -- "$telemetry_pid"; fi
fi
wait "$telemetry_pid" || true
exit "$rc"
