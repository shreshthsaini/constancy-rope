#!/usr/bin/env bash
#SBATCH -J CR14
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:30:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/logs/fleet_%j.out
# Two GPUs share four complete generation/scoring workflows through one finite spool.
set -euo pipefail
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/dev3_launch.sh
