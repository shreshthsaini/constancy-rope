#!/bin/bash
# Launch the guarded guest worker on every GPU node of mine that is currently free-idle (per slurm-idle.sh) and not
# already part of a CR fleet. Usage: bash launch_guests.sh [node ...]   (no args = detect idle nodes now)
# The guest claims a task only while the GPU has no other compute process, exits as soon as the owner job uses the GPU,
# and requeues its task if killed. Logs: /scratch/09032/saini_2/constancy_rope/logs/guest_<node>.log
P=/work/09032/saini_2/vista/Projects/Exploration/constancy_rope; R=/scratch/09032/saini_2/constancy_rope
nodes="$@"
[ -z "$nodes" ] && nodes=$(bash $HOME/.claude/tools/slurm-idle.sh 2>/dev/null | sed -n '/FREE-IDLE/,/RESERVED/p' | grep -o "c[0-9]*-[0-9]*")
for n in $nodes; do
  j=$(squeue -u $USER -h -w $n -o "%i %j %M" | head -1); case "$j" in *" CR"[0-9]*) echo "$n [$j] skip (fleet)"; continue;; esac
  r=$(ssh -o ConnectTimeout=12 -o StrictHostKeyChecking=no $n "pgrep -f guest_worker.sh >/dev/null && echo already || { nohup setsid bash $P/slurm/guest_worker.sh > $R/logs/guest_$n.log 2>&1 < /dev/null & echo launched; }" 2>&1 | tail -1)
  echo "$n [$j] $r"
done
