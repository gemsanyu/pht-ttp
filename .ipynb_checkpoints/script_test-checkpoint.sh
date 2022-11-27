#!/bin/bash

python test.py --title mpn_r1nes_final --device cuda --dataset-name a280-n279 --encoder-size 128 &
python test.py --title mpn_r1nes_final --device cuda --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title mpn_r1nes_final --device cuda --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title mpn_r1nes_final --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title mpn_r1nes_final --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 &
wait;
python test.py --title mpn_r1nes_finalv1 --device cuda --dataset-name a280-n279 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv1 --device cuda --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv1 --device cuda --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv1 --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv1 --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 &
wait;
python test.py --title mpn_r1nes_finalv2 --device cuda --dataset-name a280-n279 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv2 --device cuda --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv2 --device cuda --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv2 --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title mpn_r1nes_finalv2 --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 &
wait;
python test.py --title mpn_r1nes_final_18_10 --device cuda --dataset-name a280-n279 --encoder-size 128 &
python test.py --title mpn_r1nes_final_18_10 --device cuda --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title mpn_r1nes_final_18_10 --device cuda --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title mpn_r1nes_final_18_10 --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title mpn_r1nes_final_18_10 --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 & 
wait;