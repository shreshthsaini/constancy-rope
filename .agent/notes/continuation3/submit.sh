#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == login* ]] || { echo 'Submission preflight must run on login'; exit 2; }
count=$(ps -u "$USER" --no-headers | wc -l)
[[ "$count" -lt 40 ]] || { echo "High login process count: $count"; exit 2; }
/usr/local/etc/taccinfo
awk -F: '$1=="ASC26048" && $2=="Vista" {found=1; printf "Allocation burn %.4f percent, remaining %.3f SUs\n",100*$6/$5,$7; if($6/$5>=0.96) exit 1} END {if(!found) exit 2}' /usr/local/etc/usage.map
cd /work/09032/saini_2/vista/Projects/Exploration/constancy_rope
sbatch --parsable slurm/fleet_CR14.sbatch
