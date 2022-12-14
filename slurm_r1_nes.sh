#!/bin/bash
#
#SBATCH --job-name=att-r1nes
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --ntasks=32
#SBATCH --mem=64GB
#SBATCH --time=1:00:00

module load Anaconda3/2022.05

python main_r1nes.py --title att-phn-final --max-epoch 500