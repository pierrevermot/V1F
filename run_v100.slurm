#!/bin/bash
#SBATCH --job-name=NEBRAA_V1E
#SBATCH --output=NEBRAA_V1E-%j.out
#SBATCH --error=NEBRAA_V1E-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=nab@v100
# Default V100 mixed pool (gpu_p13). Uncomment one of the following if you must constrain:
##SBATCH -C v100-16g   # Only 16 GB V100 nodes
##SBATCH -C v100-32g   # Only 32 GB V100 nodes
##SBATCH --partition=gpu_p2    # 8x V100 nodes (mixed memory)
##SBATCH --partition=gpu_p2s   # 8x V100 360 GB RAM nodes
##SBATCH --partition=gpu_p2l   # 8x V100 720 GB RAM nodes
# QoS examples (uncomment if you need >20h or dev mode)
##SBATCH --qos=qos_gpu-t4      # Up to 100h on V100
##SBATCH --qos=qos_gpu-dev     # Dev (<=2h)

set -euo pipefail

module purge
conda deactivate || true
module load tensorflow-gpu/py3/2.11.0

echo "[INFO] Host: $(hostname)" 
 echo "[INFO] Start: $(date)" 

srun python -u main.py

echo "[INFO] End: $(date)" 
