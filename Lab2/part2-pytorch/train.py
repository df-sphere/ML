#!/usr/bin/env python3
"""
Simple training script to train models locally without Jupyter notebook.

Usage:
    python train.py --config config_twolayer
    python train.py --config config_vanilla_cnn
    python train.py --config config_mymodel
"""

import argparse
import yaml
import torch
from solver import Solver


def main():
    parser = argparse.ArgumentParser(description='Train PyTorch models on CIFAR-10')
    parser.add_argument('--config', type=str, default='config_twolayer',
                        help='Config file name (without .yaml extension)')
    args = parser.parse_args()

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("WARNING: Using CPU will cause slower train times")

    # Load config file
    config_file = f"./configs/{args.config}.yaml"
    print(f"Training a model using configuration file: {config_file}")

    with open(config_file, "r") as read_file:
        config = yaml.safe_load(read_file)

    # Parse config into kwargs
    kwargs = {}
    for key in config:
        for k, v in config[key].items():
            if k != 'description':
                kwargs[k] = v

    # Set device and path prefix
    kwargs['device'] = device
    kwargs['path_prefix'] = '.'

    print("Training configuration:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print()

    # Create solver and train
    solver = Solver(**kwargs)
    solver.train()

    print(f"\nTraining complete! Checkpoint saved to ./checkpoints/")


if __name__ == '__main__':
    main()
