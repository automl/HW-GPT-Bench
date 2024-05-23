#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/mpi-out.%j
#SBATCH --error=logs/mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=bosch_cpu-cascadelake
#SBATCH --job-name=fit_predictors
#SBATCH --mem=100GB
python predictors/hwmetric/fit_all_quantile_regression_energy.py
