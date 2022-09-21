#!/bin/bash
#
#SBATCH --job-name=drlmoa-mpn
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#
#SBATCH --nodelist=komputasi09
#SBATCH --time=30-00:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttp
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 1
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 2
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 3
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 4
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 5
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 6
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 7
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 8
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 9
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 10
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 11
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 12
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 13
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 14
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 15
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 16
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 17
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 18
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 19
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 20
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 21
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 22
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 23
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 24
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 25
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 26
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 27
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 28
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 29
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 30
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 31
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 32
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 33
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 34
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 35
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 36
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 37
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 38
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 39
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 40
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 41
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 42
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 43
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 44
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 45
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 46
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 47
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 48
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 49
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 50
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 51
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 52
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 53
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 54
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 55
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 56
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 57
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 58
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 59
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 60
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 61
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 62
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 63
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 64
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 65
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 66
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 67
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 68
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 69
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 70
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 71
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 72
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 73
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 74
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 75
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 76
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 77
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 78
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 79
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 80
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 81
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 82
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 83
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 84
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 85
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 86
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 87
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 88
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 89
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 90
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 91
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 92
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 93
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 94
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 95
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 96
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 97
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 98
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 99
python main_drlmoa.py --title mpn_drlmoa --max-epoch 200 --device cuda --num-training-samples 10000 --batch-size 96 --lr 1e-4  --total-weight 100 --weight-idx 100