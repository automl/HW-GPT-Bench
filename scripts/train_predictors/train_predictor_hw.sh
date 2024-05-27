#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -t 1-00:00:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1 # number of GPUs (per node)
#SBATCH -c 16 # number of cores
#SBATCH --job-name=fit_predictors
#SBATCH --mem=100GB
types=("quantile")
models=("conformal_quantile" "mlp" "quantile")
metrics=("latencies" "energies" "float16_memory" "bfloat16_memory")
search_spaces=("m" "s" "l")
devices=("a100" "a40" "h100" "rtx2080" "rtx3080" "a6000" "v100" "P100" "cpu_xeon_silver" "cpu_xeon_gold" "cpu_amd_7502" "cpu_amd_7513" "cpu_amd_7452")
for type in "${types[@]}"
do
    for device in "${devices[@]}"
    do
       for search_space in "${search_spaces[@]}"
       do 
          for model in "${models[@]}"
          do
            for metric in "${metrics[@]}"
            do
                python hwgpt/predictors/hwmetric/train.py --search_space $search_space --device $device --model $model --metric $metric --type $type
            done
          done
       done
    done
done