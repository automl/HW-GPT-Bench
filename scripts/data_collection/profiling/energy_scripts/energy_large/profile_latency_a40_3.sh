#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=5-00:00:00
#SBATCH --partition=single
#SBATCH --job-name=profile_latency
export PYTHONPATH=.
python profiler/profile/gpt_profiler_energy_tracker_l.py --config "config_latency_large/latency_a40.yaml" --start_index 5000 --end_index 7500
