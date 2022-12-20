#!/bin/bash
#
#SBATCH --job-name=att-single
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

module load Anaconda3/2022.05
module load CUDA
python main_single.py --title single-agent-attv2 --max-epoch 500 --batch-size 128 --device cuda --num-training-samples 10000
