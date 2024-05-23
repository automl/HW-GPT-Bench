#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=1-00:00:00
#SBATCH --partition=ml_gpu-rtxA6000
#SBATCH --job-name=job

export PYTHONPATH=.
python profiler/profile/gpt_profiler_l.py --config "config_latency_large/latency_a6000.yaml" --start_index 0 --end_index 2500
