#!/usr/bin/env bash
#SBATCH -J CR14
#SBATCH -p gb
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/logs/fleet_%j.out
# One four-GPU node packs four complete generation/scoring workflows.
set -euo pipefail
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/dev3_launch.sh
