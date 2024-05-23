#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=mldlc_gpu-rtx2080
#SBATCH --job-name=owt_eval
python profiler/profile/gpt_perplexity_profiler.py