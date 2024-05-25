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
python profiler/profile/gpt_profiler.py --config "config_latency/latency_amd.yaml" --resume latency_amd/efficiency_observations_0_10000.pkl --start_index 0 --end_index 10000
