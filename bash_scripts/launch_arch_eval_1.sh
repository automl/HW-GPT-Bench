#!/bin/bash -x

#SBATCH --account=projectnucleus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --job-name=owt_eval
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export MASTER_PORT=12802
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR

module load Stages/2024
module load CUDA/12
module load GCC/12.3.0
module load Python/3.11.3
source /p/scratch/ccstdl/sukthanker1/gpt/bin/activate
export PYTHONPATH=.
PYTHON_SCRIPT=profiler/profile/gpt_perplexity_profiler.py
srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python -u $PYTHON_SCRIPT --start_index 0 --end_index 1000 $@