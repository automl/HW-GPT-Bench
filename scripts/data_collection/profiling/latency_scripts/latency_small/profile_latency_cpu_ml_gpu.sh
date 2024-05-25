#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=ml_gpu-rtx2080
#SBATCH --job-name=job
export PYTHONPATH=.
python data_collection/gpt_profiler/profile/gpt_latency_profiler.py --config "configs/config_lat_en_small/latency_cpu_mlgpu.yaml" --start_index 0 --end_index 10000 --scale s