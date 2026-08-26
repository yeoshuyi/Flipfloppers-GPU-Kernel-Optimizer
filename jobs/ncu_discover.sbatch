#!/bin/bash
#SBATCH --job-name=ncu_discover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/runs/%j.out

set -euo pipefail

apptainer exec --nv --cleanenv \
    /scratch/kernel.sif \
    ncu --query-metrics 2>&1 | grep -iE \
    "^(sm__throughput|dram__throughput|lts__throughput|sm__warps_active|launch__registers_per_thread|launch__shared_mem_per_block|launch__waves_per_multiprocessor|gpu__time_duration|smsp__warp_issue_stalled|l1tex__data_bank_conflicts)" \
    | grep -v "\.min\.\|\.max\." \
    | sort
