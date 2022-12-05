#!/bin/bash

# python test.py --title att_phn --device cuda --dataset-name a280-n279;
# python test.py --title att_phn --device cuda --dataset-name a280-n1395;
# python test.py --title att_phn --device cuda --dataset-name a280-n2790;
# python test.py --title att_phn --device cuda --dataset-name fnl4461-n4460;
# python test.py --title att_phn --device cuda --dataset-name fnl4461-n22300;

python test.py --title att_phn-rs64 --ray-hidden-size 64 --device cuda --dataset-name a280-n279;
python test.py --title att_phn-rs64 --ray-hidden-size 64 --device cuda --dataset-name a280-n1395;
python test.py --title att_phn-rs64 --ray-hidden-size 64 --device cuda --dataset-name a280-n2790;
python test.py --title att_phn-rs64 --ray-hidden-size 64 --device cuda --dataset-name fnl4461-n4460;
python test.py --title att_phn-rs64 --ray-hidden-size 64 --device cuda --dataset-name fnl4461-n22300;

python test.py --title att_phn-rs256 --ray-hidden-size 256 --device cuda --dataset-name a280-n279;
python test.py --title att_phn-rs256 --ray-hidden-size 256 --device cuda --dataset-name a280-n1395;
python test.py --title att_phn-rs256 --ray-hidden-size 256 --device cuda --dataset-name a280-n2790;
python test.py --title att_phn-rs256 --ray-hidden-size 256 --device cuda --dataset-name fnl4461-n4460;
python test.py --title att_phn-rs256 --ray-hidden-size 256 --device cuda --dataset-name fnl4461-n22300;

# python test.py --title att_phn_pure --device cuda --dataset-name a280-n279;
# python test.py --title att_phn_pure --device cuda --dataset-name a280-n1395;
# python test.py --title att_phn_pure --device cuda --dataset-name a280-n2790;
# python test.py --title att_phn_pure --device cuda --dataset-name fnl4461-n4460;
# python test.py --title att_phn_pure --device cuda --dataset-name fnl4461-n22300;