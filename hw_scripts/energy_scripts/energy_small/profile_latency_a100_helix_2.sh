#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=A100:1
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=5-00:00:00
#SBATCH --partition=single
#SBATCH --job-name=profile_latency
module load Stages/2024
module load CUDA/12
module load GCC/12.3.0
module load Python/3.11.3
source /p/scratch/ccstdl/sukthanker1/gpt/bin/activate
export PYTHONPATH=.
python profiler/profile/gpt_profiler_energy.py --config "config_latency/latency_a100.yaml" --start_index 2500 --end_index 5000
