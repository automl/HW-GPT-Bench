#!/bin/bash

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=0-02:00:00
#SBATCH --partition=develbooster
#SBATCH --job-name=job
export PYTHONPATH=.
python data_collection/gpt_profiler/profile/gpt_mem_profiler.py --config configs/config_mem_large/mem_a100.yaml --start_index 0 --end_index 10000 --scale l
