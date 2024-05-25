#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=ml_gpu-teslaP100
#SBATCH --job-name=job

export PYTHONPATH=.
python profiler/profile/gpt_profiler.py --config "config_latency/latency_cpu_p100.yaml" --start_index 7500 --end_index 10000