#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=1-00:00:00
#SBATCH --partition=single
#SBATCH --job-name=profile_latency
#SBATCH --mem=50GB
export PYTHONPATH=.
python profiler/profile/gpt_profiler_m.py --config "config_latency_medium/latency_helix_cpu.yaml"  --start_index 5000 --end_index 7500
