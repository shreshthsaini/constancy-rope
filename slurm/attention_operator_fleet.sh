#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/09032/saini_2/vista/Projects/Exploration/constancy_rope
R=/scratch/09032/saini_2/constancy_rope
Q=$R/taskq_attention_operator
setsid bash /work/09032/saini_2/vista/library/bin/gpu_telemetry.sh "$R/telemetry" 30 &
logger=$!
cleanup() {
    if ps -o pid,ppid,pgid,lstart,stat,args -p "$logger"; then
        parent=$(ps -o ppid= -p "$logger")
        group=$(ps -o pgid= -p "$logger")
        if [[ "${parent// /}" == "$$" && "${group// /}" == "$logger" ]]; then
            kill -- "-$logger"
        fi
    fi
    wait "$logger" || true
}
trap cleanup EXIT
trap 'exit 143' TERM
shopt -s nullglob
status=0
for task in "$Q"/pending/*.sh; do
    name=${task##*/}
    claimed="$Q/running/$name.$(hostname -s)"
    if mv -- "$task" "$claimed" 2>/dev/null; then
        if bash "$claimed" > "$Q/logs/$name.log" 2>&1; then
            mv -- "$claimed" "$Q/done/$name"
        else
            status=1
            mv -- "$claimed" "$Q/failed/$name"
        fi
    fi
done
exit "$status"
