#!/bin/bash
#SBATCH -J CR1
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -A ASC26048
#SBATCH -o /scratch/09032/saini_2/constancy_rope/logs/fleet_%j.out
R=/scratch/09032/saini_2/constancy_rope
srun --ntasks-per-node=1 bash -c "/work/09032/saini_2/vista/library/bin/gpu_telemetry.sh $R/telemetry 30 & bash /work/09032/saini_2/vista/Projects/Exploration/constancy_rope/slurm/worker_loop.sh"
