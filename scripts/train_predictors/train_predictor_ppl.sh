#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1 # number of GPUs (per node)
#SBATCH -c 16 # number of cores
#SBATCH --job-name=fit_predictors
#SBATCH --mem=100GB
python hwgpt/predictors/metric/train.py --search_space m
python hwgpt/predictors/metric/train.py --search_space s
python hwgpt/predictors/metric/train.py --search_space l

