#!/bin/bash
methods=("RS" "MOREA" "LS" "NSGA2" "LSBO" "RSBO" "MOASHA" "EHVI")
metrics=("float16_memory" "bloat16_memory")
search_spaces=("s" "m" "l")
seeds=(9001 9002 9003 9004 9005)
for search_space in "${search_spaces[@]}"
  do
    for metric in "${metrics[@]}"
    do
        for method in "${methods[@]}"
        do
              for seed in "${seeds[@]}"
              do 
                exp_name="baseline-${method}-${search_space}-${metric}-${seed}"
                echo Submitting job $exp_name
                sbatch -J $exp_name scripts/baselines/baselines_gpt_agnostic.sh $search_space $method $metric $seed
              done
        done
    done    
  done