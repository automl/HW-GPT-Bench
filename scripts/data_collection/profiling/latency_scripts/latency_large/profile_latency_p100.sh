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
python data_collection/gpt_profiler/profile/gpt_latency_profiler.py --config "configs/config_lat_en_large/latency_P100.yaml" --start_index 0 --end_index 10000 --scale l
