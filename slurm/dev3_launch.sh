#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/09032/saini_2/vista/Projects/Exploration/constancy_rope
R=/scratch/09032/saini_2/constancy_rope
count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$count" -eq 1 || "$count" -eq 4 ]] || { echo "Expected one or four GPUs; found $count"; exit 2; }
setsid bash /work/09032/saini_2/vista/library/bin/gpu_telemetry.sh "$R/telemetry" 30 &
telemetry_pid=$!
pids=()
for ((gpu=0; gpu<count; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu bash "$ROOT/slurm/dev3_worker.sh" &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
# Signal only the recorded logger child, after verifying its live identity.
if ps -o pid,ppid,pgid,lstart,stat,args -p "$telemetry_pid"; then
    parent=$(ps -o ppid= -p "$telemetry_pid")
    group=$(ps -o pgid= -p "$telemetry_pid")
    if [[ "${parent// /}" == "$$" && "${group// /}" == "$telemetry_pid" ]]; then
        kill -- "-$telemetry_pid"
    fi
fi
wait "$telemetry_pid" || true
Q="$R/taskq_dev3"
if [[ "$rc" -eq 0 ]] && [[ $(find "$Q/done" -maxdepth 1 -name '*.sh' | wc -l) -eq 4 ]]; then
    if mkdir "$Q/finalize.lock" 2>/dev/null; then
        export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
        /work/09032/saini_2/vista/library/uv-envs/crope/bin/python "$ROOT/scripts/dev3_finalize.py" \
            > "$Q/logs/finalize.log" 2>&1 || rc=1
    fi
fi
exit "$rc"
