#!/bin/bash
# Start telemetry and one worker_loop per visible GPU on this node. When every worker has exited, stop the
# telemetry logger so the Slurm step (and job) ends instead of holding idle nodes until walltime.
R=/scratch/09032/saini_2/constancy_rope
/work/09032/saini_2/vista/library/bin/gpu_telemetry.sh $R/telemetry 30 & tel=$!
N=$(nvidia-smi -L 2>/dev/null | grep -c GPU); [ "$N" -lt 1 ] && N=1
echo "$(hostname -s) gpus=$N"
pids=()
for g in $(seq 0 $((N-1))); do CUDA_VISIBLE_DEVICES=$g bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/worker_loop.sh & pids+=($!); done
wait "${pids[@]}"
kill $tel 2>/dev/null
