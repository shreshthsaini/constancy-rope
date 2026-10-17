#!/usr/bin/env bash
set -euo pipefail
Q=/scratch/09032/saini_2/constancy_rope/taskq_dev3
shopt -s nullglob
for task in "$Q"/pending/*.sh; do
    name=${task##*/}
    claimed="$Q/running/$name.$(hostname -s).g${CUDA_VISIBLE_DEVICES}"
    if mv -- "$task" "$claimed" 2>/dev/null; then
        if bash "$claimed" > "$Q/logs/$name.log" 2>&1; then
            mv -- "$claimed" "$Q/done/$name"
        else
            code=$?
            mv -- "$claimed" "$Q/failed/$name"
            exit "$code"
        fi
    fi
done
