#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=ml_gpu-teslaP100
#SBATCH --job-name=latency_p100
export PYTHONPATH=.
python profiler/profile/gpt_profiler.py --config "config_latency/latency_P100.yaml" --start_index 5000 --end_index 7500