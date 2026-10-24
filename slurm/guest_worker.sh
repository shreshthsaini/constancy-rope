#!/bin/bash
# Guest worker for a GPU that is idle inside one of my other allocations (user-authorized 2026-10-19).
# Claims a task only while the GPU runs no other compute process; stops claiming the moment the owner uses the GPU;
# if killed (owner job ends), puts its task back in pending.
Q=/scratch/09032/saini_2/constancy_rope/taskq; L=/scratch/09032/saini_2/constancy_rope/logs; H=$(hostname -s)-guest
cur=""; child=""; IDLE=0
requeue(){ [ -n "$child" ] && kill $child 2>/dev/null; if [ -n "$cur" ] && [ -f $Q/running/$cur.$H ]; then mv $Q/running/$cur.$H $Q/pending/$cur; echo "$(date +%FT%T) $H requeued $cur"; fi; exit 0; }
trap requeue TERM INT HUP
echo "$(date +%FT%T) $H guest start"
while [ ! -f $Q/STOP ]; do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$busy" -gt 0 ] && { echo "$(date +%FT%T) $H owner using GPU; guest exits"; break; }
  t=$(ls $Q/pending 2>/dev/null | grep -v -E "^7[0-9]_qwen|^82_" | sort | head -1)
  if [ -z "$t" ]; then sleep 60; IDLE=$((IDLE+60)); [ $IDLE -ge 1800 ] && break; continue; fi
  if mv $Q/pending/$t $Q/running/$t.$H 2>/dev/null; then
    IDLE=0; cur=$t; echo "$(date +%FT%T) $H start $t"
    bash $Q/running/$t.$H > $L/task_${t%.sh}.log 2>&1 & child=$!; wait $child; rc=$?; child=""
    if [ $rc -eq 0 ]; then mv $Q/running/$t.$H $Q/done/$t; else mv $Q/running/$t.$H $Q/failed/$t; fi
    echo "$(date +%FT%T) $H end $t rc=$rc"; cur=""
  fi
done
echo "$(date +%FT%T) $H guest exit"
