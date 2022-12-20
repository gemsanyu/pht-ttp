#!/bin/bash
#
#SBATCH --job-name=att-single
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=20
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

module load Anaconda3/2022.05
module load CUDA
python generate_dataset.py
