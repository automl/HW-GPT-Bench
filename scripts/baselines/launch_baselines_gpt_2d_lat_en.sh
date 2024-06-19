#!/bin/bash
methods=("RS" "MOREA" "LS" "NSGA2" "LSBO" "RSBO" "MOASHA" "EHVI")
devices=("rtx2080" "h100" "cpu_xeon_silver" "cpu_xeon_gold") #("a100" "a40" "h100" "rtx2080" "rtx3080" "a6000" "v100" "P100" "cpu_xeon_silver" "cpu_xeon_gold" "cpu_amd_7502" "cpu_amd_7513" "cpu_amd_7452")
metrics=("latencies" "energies") # "flops" "params" "float16_memory" "bloat16_memory")
search_spaces=("s" "m" "l")
seeds=(9001 9002 9003 9004 9005)
for device in "${devices[@]}"
do
   for search_space in "${search_spaces[@]}"
   do
    for metric in "${metrics[@]}"
    do
        for method in "${methods[@]}"
        do
              for seed in "${seeds[@]}"
              do 
                exp_name="baseline-${method}-${device}-${search_space}-${metric}-${seed}"
                echo Submitting job $exp_name
                sbatch -J $exp_name scripts/baselines/baselines_gpt_2d_lat_en.sh $search_space $device $method $metric $seed
              done
        done
    done    
   done
done

