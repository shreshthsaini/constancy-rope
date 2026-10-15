#!/bin/bash
#SBATCH -J CR2
#SBATCH -p gb
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/logs/fleet_%j.out
srun --ntasks-per-node=1 bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/node_launch.sh
