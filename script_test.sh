#!/bin/bash
#
#SBATCH --job-name=testing-mpn-r1nes
#SBATCH --output=logs/mpn-r1nes_%A.out
#SBATCH --error=logs/mpn-r1nes_%A.err
#
#SBATCH --time=30-00:00:00
#SBATCH --nodelist=komputasi09

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttp

python test.py --title att-r1nes  --dataset-name a280-n279 --encoder-size 128 &
python test.py --title att-r1nesv1  --dataset-name a280-n279 --encoder-size 128 &
python test.py --title att-r1nesv2  --dataset-name a280-n279 --encoder-size 128 &
wait;

python test.py --title att-r1nes  --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title att-r1nesv1  --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title att-r1nesv2  --dataset-name a280-n1395 --encoder-size 128 &
wait;

python test.py --title att-r1nes  --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title att-r1nesv1  --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title att-r1nesv2  --dataset-name a280-n2790 --encoder-size 128 &
wait;

python test.py --title att-r1nes  --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title att-r1nesv1  --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title att-r1nesv2  --dataset-name fnl4461-n4460 --encoder-size 128 &
wait;

python test.py --title att-r1nes  --dataset-name fnl4461-n22300 --encoder-size 128 &
python test.py --title att-r1nesv1  --dataset-name fnl4461-n22300 --encoder-size 128 &
python test.py --title att-r1nesv2  --dataset-name fnl4461-n22300 --encoder-size 128 &
wait;