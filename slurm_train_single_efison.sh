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
python main.py --title new_single_agent --max-epoch 1000 --batch-size 256 --device cuda --num-training-samples 10000
