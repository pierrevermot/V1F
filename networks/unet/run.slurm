#!/bin/bash
#SBATCH --job-name=ASTRON_MIX # name of job
#SBATCH --output=XX%j.out # output file (%j = job ID)
#SBATCH --error=XX%j.err # error file (%j = job ID)
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=1 # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:1 # reserve 4 GPUs
#SBATCH --cpus-per-task=10 # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=20:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t3 # QoS 
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=hfk@v100 # V100 accounting
##SBATCH -C a100
cat run.slurm
cat *.py

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
#module load cpuarch/amd
module load tensorflow-gpu/py3/2.11.0 # charger les modules

set -x # activate echo of launched commands
srun python -u model.py # execute script
