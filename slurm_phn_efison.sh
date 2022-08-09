#!/bin/bash
#
#SBATCH --job-name=phn-ttp
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --partition=gpu_ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=10
#SBATCH --mem=32GB
#SBATCH --time=12:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
python main_phn.py --title newv1 --max-epoch 200 --device cuda --num-training-samples 5000
