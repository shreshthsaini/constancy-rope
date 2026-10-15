#!/bin/bash
# One worker per GPU node: atomically claim task scripts from the spool and run them.
# Exits when $Q/STOP exists, or after IDLE_MAX seconds with nothing pending.
Q=/scratch/09032/saini_2/constancy_rope/taskq; H=$(hostname -s)-g${CUDA_VISIBLE_DEVICES:-0}; IDLE=0; IDLE_MAX=${IDLE_MAX:-5400}
cur=""; child=""
# On walltime SIGTERM, stop the running task and put it back in the queue so another worker reruns it.
requeue(){ [ -n "$child" ] && kill $child 2>/dev/null; [ -n "$cur" ] && [ -f $Q/running/$cur.$H ] && mv $Q/running/$cur.$H $Q/pending/$cur && echo "$(date +%FT%T) $H requeued $cur"; exit 0; }
trap requeue TERM INT HUP
while [ ! -f $Q/STOP ]; do
  t=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -z "$t" ]; then sleep 30; IDLE=$((IDLE+30)); [ $IDLE -ge $IDLE_MAX ] && break; continue; fi
  if mv $Q/pending/$t $Q/running/$t.$H 2>/dev/null; then
    IDLE=0; echo "$(date +%FT%T) $H start $t"
    cur=$t; bash $Q/running/$t.$H > /scratch/09032/saini_2/constancy_rope/logs/task_${t%.sh}.log 2>&1 & child=$!
    wait $child; rc=$?; child=""; cur=""; if [ $rc -eq 0 ]; then mv $Q/running/$t.$H $Q/done/$t; else mv $Q/running/$t.$H $Q/failed/$t; fi
    echo "$(date +%FT%T) $H end $t rc=$rc"
  fi
done
echo "$(date +%FT%T) $H worker exit"
