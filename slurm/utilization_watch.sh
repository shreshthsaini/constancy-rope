#!/bin/bash
# Event stream for the operator: emits one line per change worth acting on.
# FLEET <job> <state>      a CR fleet job changed state (PENDING/RUNNING/ended)
# FAILED <task>            a new task landed in taskq/failed
# UNDERUSED <job> ...      a running CR fleet has fewer busy workers than GPUs while tasks are pending
# QUEUE_EMPTY              nothing pending or running
# IDLE_OTHER <node> <job>  a GPU node in one of my OTHER jobs idle for >=2 checks (job older than 30 min)
Q=/scratch/09032/saini_2/constancy_rope/taskq
declare -A last; declare -A idle_seen; prev_failed=$(ls $Q/failed | sort); empty_said=0; n=0
while true; do
  while read -r id name state; do
    [ "${last[$id]}" != "$state" ] && echo "FLEET $id $name $state" && last[$id]=$state
  done < <(squeue -u $USER -h -o "%i %j %T" 2>/dev/null | grep -E " CR[0-9]+ ")
  for id in "${!last[@]}"; do
    squeue -j $id -h >/dev/null 2>&1 && [ -n "$(squeue -j $id -h 2>/dev/null)" ] || { [ "${last[$id]}" != "ENDED" ] && echo "FLEET $id ENDED" && last[$id]=ENDED; }
  done
  cur_failed=$(ls $Q/failed | sort); new=$(comm -13 <(echo "$prev_failed") <(echo "$cur_failed")); [ -n "$new" ] && echo "FAILED $(echo $new | tr '\n' ' ')"; prev_failed=$cur_failed
  pend=$(ls $Q/pending | wc -l); run=$(ls $Q/running | wc -l)
  if [ "$pend" -eq 0 ] && [ "$run" -eq 0 ]; then [ $empty_said -eq 0 ] && echo "QUEUE_EMPTY" && empty_said=1; else empty_said=0; fi
  if [ "$pend" -gt 0 ]; then
    while read -r id name nodes; do
      gpus=$(squeue -j $id -h -o %D); busy=$(ls $Q/running | grep -c -E "\.($(scontrol show hostnames $nodes 2>/dev/null | tr '\n' '|' | sed 's/|$//'))-g")
      [ "$busy" -lt "$gpus" ] && echo "UNDERUSED $id $name busy=$busy nodes=$gpus pending=$pend"
    done < <(squeue -u $USER -h -t RUNNING -o "%i %j %N" | grep -E " CR[0-9]+ " | grep -v " CR23 ")
  fi
  n=$((n+1))
  if [ $((n % 3)) -eq 0 ]; then
    for node in $(bash $HOME/.claude/tools/slurm-idle.sh 2>/dev/null | sed -n '/FREE-IDLE/,/RESERVED/p' | grep -o "c[0-9]*-[0-9]*"); do
      j=$(squeue -u $USER -h -w $node -o "%j %M" | head -1); case "$j" in CR*) continue;; esac
      idle_seen[$node]=$(( ${idle_seen[$node]:-0} + 1 ))
      [ ${idle_seen[$node]} -eq 2 ] && echo "IDLE_OTHER $node $j"
    done
  fi
  sleep 300
done
