#!/bin/bash                      

#SBATCH --ntasks=32         # Contoh menggunakan 32 core CPU.
#SBATCH --mem=64GB               # Contoh menggunakan RAM 16GB.
#SBATCH --time=24:00:00          # Contoh menetapkan walltime maks 30 menit.
#SBATCH --output=logs/result-%j.out   # Output terminal program.
#SBATCH --error=logs/result-%j.err    # Output verbose program.
#SBATCH --partition=ampere

module load Anaconda3

python test.py --title att-r1nesv2 --dataset-name a280-n279;
python test.py --title att-r1nesv2 --dataset-name a280-n1395;
python test.py --title att-r1nesv2 --dataset-name a280-n2790;
python test.py --title att-r1nesv2 --dataset-name fnl4461-n4460;
python test.py --title att-r1nesv2 --dataset-name fnl4461-n22300;
