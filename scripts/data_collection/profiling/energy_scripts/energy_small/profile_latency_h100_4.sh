#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu_4_h100
#SBATCH --job-name=job
export PYTHONPATH=.
python profiler/profile/gpt_profiler_energy_tracker.py --config "config_latency/latency_h100.yaml" --start_index 7500 --end_index 10000