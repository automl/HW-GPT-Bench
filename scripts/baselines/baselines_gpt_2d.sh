#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1 # number of GPUs (per node)
#SBATCH -c 2 # number of cores
#SBATCH -o logs_final/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs_final/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)

method=$1
device=$2
seed=$3

python baselines/run_nas_gpt_2d.py --method $method --device $device 