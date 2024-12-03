#!/bin/bash
#SBATCH -p partition
#SBATCH -t 4-00:00:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1 # number of GPUs (per node)
#SBATCH -c 2 # number of cores
#SBATCH -o logs_final/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs_final/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)

method=$1
python baselines/run_nas_gpt_3d.py --method $method --search_space l --device_1 "a100" --device_2 "cpu_xeon_gold" --objective_1 "latencies" --objective_2 "energies"  --random_seed 1234