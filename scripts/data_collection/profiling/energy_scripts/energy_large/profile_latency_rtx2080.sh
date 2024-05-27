#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=1-00:00:00
#SBATCH --partition=alldlc_gpu-rtx2080
#SBATCH --job-name=job
export PYTHONPATH=.
python data_collection/gpt_profiler/profile/gpt_profiler_energy_tracker.py --config "configs/config_lat_en_large/latency_rtx2080.yaml" --start_index 0 --end_index 10000 --scale l
