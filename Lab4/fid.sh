#!/bin/bash

MODEL=$1

fid() {
    python3 -m pytorch_fid outputs/$MODEL/images tests/assets/fid-stats-fashion.npz --device cuda
}

time (rm -rf outputs/$MODEL && ./train.sh configs/config_$MODEL.yaml && ./generate.sh configs/config_$MODEL.yaml && fid)
