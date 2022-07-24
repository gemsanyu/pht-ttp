#!/bin/bash
#
#SBATCH --job-name=nes-tsp
#SBATCH --output=logs/tsp_%A.out
#SBATCH --error=logs/tsp_%A.err
#
#SBATCH --time=30-00:00:00
#SBATCH --nodelist=komputasi08

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttp
srun python main.py --max-epoch 100000 --policy-device cpu --actor-device cpu --num-threads 20 --model-name r1_nes_mis
