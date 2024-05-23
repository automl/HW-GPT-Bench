#!/bin/bash
methods=("EHVI")
devices=("rtx2080" "rtx3080" "v100" "a40")
for method in "${methods[@]}"
do
    for device in "${devices[@]}"
    do
        exp_name="test-${method}-${device}"
        echo Submitting job $exp_name
        sbatch --bosch -J $exp_name bash_scripts_baselines/baselines_gpt.sh $method $device $seed
    done
done