#!/bin/sh

python3 -m pytorch_fid outputs/vae/images tests/assets/fid-stats-fashion.npz --device cuda 
