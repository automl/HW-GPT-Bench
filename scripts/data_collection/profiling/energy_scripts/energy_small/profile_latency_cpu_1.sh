#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=3-00:00:00
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --job-name=job

export PYTHONPATH=.
python profiler/profile/gpt_profiler.py --config "config_latency/latency_cpu_meta.yaml" --start_index 0 --end_index 2500