#!/usr/bin/env bash
#SBATCH -J CR16
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:15:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/logs/repeatability_%j.out
set -euo pipefail
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/repeatability_fleet.sh
