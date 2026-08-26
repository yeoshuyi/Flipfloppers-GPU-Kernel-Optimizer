#!/bin/bash
#SBATCH --job-name=l2_discover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/runs/%j.out

set -euo pipefail

apptainer exec --nv --cleanenv /scratch/kernel.sif bash -c '
echo "=== cudaLimit enum (driver_types.h) ==="
grep -A 12 "enum.*cudaLimit\b" /usr/local/cuda/include/driver_types.h | head -20
echo
echo "=== cudaStreamAttrID enum ==="
grep -B2 -A 12 "cudaStreamAttrID" /usr/local/cuda/include/driver_types.h | head -25
echo
echo "=== cudaAccessProperty / cudaAccessPolicyWindow ==="
grep -B2 -A 10 "cudaAccessProperty\b\|struct.*cudaAccessPolicyWindow" /usr/local/cuda/include/driver_types.h | head -40
echo
echo "=== python: torch.cuda.cudart() availability ==="
python3 -c "
import torch
c = torch.cuda.cudart()
print(type(c))
print([n for n in dir(c) if \"Limit\" in n or \"Attribute\" in n or \"StreamSet\" in n])
"
'
