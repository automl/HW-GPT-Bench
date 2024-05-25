#!/bin/bash
methods=("RS" "BO" "MOREA" "LS" "NSGA2" "LSBO" "RSBO" "MOASHA" "EHVI")
devices=("rtx2080" "rtx3080" "v100" "a40" "a100" "h100" "v100" "a6000")
for method in "${methods[@]}"
do
    for device in "${devices[@]}"
    do
        exp_name="test-${method}-${device}"
        echo Submitting job $exp_name
        sbatch --bosch -J $exp_name scripts/baselines/baselines_gpt_2d.sh $method $device $seed
    done
done