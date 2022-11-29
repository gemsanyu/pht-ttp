#!/bin/bash

python test.py --title mpn-phn --device cuda --dataset-name a280-n279 --encoder-size 128 &
python test.py --title mpn-phn-rs64 --device cuda --dataset-name a280-n279 --encoder-size 128 &
python test.py --title mpn-phn-rs256 --device cuda --dataset-name a280-n279 --encoder-size 128 &
wait;
python test.py --title mpn-phn --device cuda --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title mpn-phn-rs64 --device cuda --dataset-name a280-n1395 --encoder-size 128 &
python test.py --title mpn-phn-rs256 --device cuda --dataset-name a280-n1395 --encoder-size 128 &
wait;
python test.py --title mpn-phn --device cuda --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title mpn-phn-rs64 --device cuda --dataset-name a280-n2790 --encoder-size 128 &
python test.py --title mpn-phn-rs256 --device cuda --dataset-name a280-n2790 --encoder-size 128 &
wait;
python test.py --title mpn-phn --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title mpn-phn-rs64 --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
python test.py --title mpn-phn-rs256 --device cuda --dataset-name fnl4461-n4460 --encoder-size 128 &
wait;
python test.py --title mpn-phn --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 &
python test.py --title mpn-phn-rs64 --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 &
python test.py --title mpn-phn-rs256 --device cuda --dataset-name fnl4461-n22300 --encoder-size 128 &
wait;
