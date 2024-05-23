#!/bin/bash
methods=("LSBO" "RSBO")
for method in "${methods[@]}"
do
    exp_name="test-${method}-${device}"
    echo Submitting job $exp_name
    sbatch --bosch -J $exp_name bash_scripts_baselines/baselines_gpt_3_objs.sh $method 
done