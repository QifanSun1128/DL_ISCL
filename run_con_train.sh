#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python $PWD/con_main.py --dataset multi --source real --target sketch --net $2 --save_check
