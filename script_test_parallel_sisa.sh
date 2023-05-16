#/bin/bash

python test.py --title AM-PHN  --dataset-name eil76_n75_uncorr_09 &
python test.py --title AM-PHN  --dataset-name eil76_n75_uncorr_10 &
wait;
