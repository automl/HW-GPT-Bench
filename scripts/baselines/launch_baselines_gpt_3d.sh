#!/bin/bash
methods=("RS" "BO" "MOREA" "LS" "NSGA2" "LSBO" "RSBO" "MOASHA" "EHVI")
for method in "${methods[@]}"
do
    exp_name="test-${method}-${device}"
    echo Submitting job $exp_name
    sbatch --bosch -J $exp_name scripts/baselines/baselines_gpt_3d.sh $method 
done