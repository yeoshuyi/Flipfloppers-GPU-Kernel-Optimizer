#!/bin/bash
#SBATCH --job-name=ncu_discover2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/runs/%j.out

set -euo pipefail

apptainer exec --nv --cleanenv \
    /scratch/kernel.sif \
    ncu --query-metrics 2>&1 | grep -iE \
    "^(launch__|sm__cycles_elapsed|sm__sass_thread_inst_executed_op|sm__pipe_tensor|sm__inst_executed_pipe_tensor)" \
    | grep -v "\.min\.\|\.max\." \
    | sort
