#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=5-00:00:00
#SBATCH --partition=single
#SBATCH --job-name=profile_latency
export PYTHONPATH=.
python data_collection/gpt_profiler/profile/gpt_latency_profiler.py --config "configs/config_lat_en_large/latency_helix_cpu.yaml" --start_index 0 --end_index 10000 --scale l
