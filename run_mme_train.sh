#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python $PWD/MME.py --method $2 --dataset multi --source real --target sketch --net $3 --save_check
