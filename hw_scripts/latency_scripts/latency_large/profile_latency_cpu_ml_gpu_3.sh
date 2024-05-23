#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=ml_gpu-rtx2080
#SBATCH --job-name=job

export PYTHONPATH=.
python profiler/profile/gpt_profiler_l.py --config "config_latency_large/latency_cpu_mlgpu.yaml" --start_index 5000 --end_index 7500