#!/bin/sh

export LD_LIBRARY_PATH=/home/diego/amrnd/am/anaconda3/envs/cs7643-a4/lib:$LD_LIBRARY_PATH 
python3 train.py --config_file $1 
