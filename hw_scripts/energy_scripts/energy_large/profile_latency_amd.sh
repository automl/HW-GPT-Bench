#!/bin/bash

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=0-06:00:00
#SBATCH --partition=booster
#SBATCH --job-name=job
module load Stages/2024
module load CUDA/12
module load GCC/12.3.0
module load Python/3.11.3
source /p/scratch/ccstdl/sukthanker1/gpt/bin/activate
export PYTHONPATH=.
python profiler/profile/gpt_profiler_l.py --config "config_latency_large/latency_amd.yaml" --start_index 0 --end_index 10000
