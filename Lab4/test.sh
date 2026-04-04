#!/bin/sh

export LD_LIBRARY_PATH=/home/diego/amrnd/am/anaconda3/envs/cs7643-a4/lib:$LD_LIBRARY_PATH 
pytest tests/test_VAE.py tests/test_VAEloss.py 
pytest tests/test_GAN.py
