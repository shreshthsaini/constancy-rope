#!/bin/bash
#SBATCH -J CR25
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/logs/fleet_%j.out
export IDLE_MAX=14400
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/node_launch.sh
