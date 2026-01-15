#!/bin/bash
# Simple A100 run script
#SBATCH --job-name=NEBRAA_V1E_A100
#SBATCH --output=NEBRAA_V1E_A100-%j.out
#SBATCH --error=NEBRAA_V1E_A100-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=nab@a100
#SBATCH -C a100
##SBATCH --qos=qos_gpu_a100-dev   # Uncomment for <=2h dev runs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

module purge
conda deactivate || true
module load arch/a100
module load tensorflow-gpu/py3/2.11.0

echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"

srun python -u main.py

echo "[INFO] End: $(date)"

