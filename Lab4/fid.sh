#!/bin/bash

fid() {
    python3 -m pytorch_fid outputs/$1/images tests/assets/fid-stats-fashion.npz --device cuda
}

#time (rm -rf outputs && ./train.sh configs/config_vae.yaml && ./generate.sh configs/config_vae.yaml && fid vae)
time (rm -rf outputs && ./train.sh configs/config_gan.yaml && ./generate.sh configs/config_gan.yaml && fid gan)
