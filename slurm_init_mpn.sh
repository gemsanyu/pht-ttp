#!/bin/bash
#
#SBATCH --job-name=phn-ttp
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --partition=gpu_ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
python main.py --title mpn_init_128_latest --max-epoch 1000 --device cuda --num-training-samples 10000 --batch-size 128
