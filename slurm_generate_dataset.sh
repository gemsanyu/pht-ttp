#!/bin/bash
#
#SBATCH --job-name=phn-ttp
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --ntasks=20
#SBATCH --mem=32GB
#SBATCH --time=1:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
python generate_dataset.py
