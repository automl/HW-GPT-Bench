#!/bin/bash

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=0-06:00:00
#SBATCH --partition=booster
#SBATCH --job-name=job

python data_collection/gpt_profiler/profile/gpt_mem_profiler.py --config configs/config_mem_medium/mem_a100.yaml --start_index 0 --end_index 10000 --scale m
