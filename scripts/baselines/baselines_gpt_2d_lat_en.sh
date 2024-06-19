#!/bin/bash
#SBATCH -p partition
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1 # number of GPUs (per node)
#SBATCH -c 8 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
search_space=$1
device=$2
method=$3
objective=$4
seed=$5

python baselines/run_nas_gpt_2d.py --search_space $search_space --device $device --method $method --objective $objective --random_seed $seed