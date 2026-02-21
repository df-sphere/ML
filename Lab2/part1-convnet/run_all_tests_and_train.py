"""
Complete script for Assignment 2 Part 1
Runs all tests and training experiment from the notebook
"""

import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import subprocess

# Set up path (equivalent to notebook's GOOGLE_DRIVE_PATH = '.')
GOOGLE_DRIVE_PATH = '.'
print('Running locally.')
print(f'Working directory: {os.getcwd()}')

# ==============================================================================
# Environment check - say hello
# ==============================================================================
print('\n' + '='*80)
print('ENVIRONMENT CHECK')
print('='*80)

from cs7643.env_prob import say_hello_do_you_copy
say_hello_do_you_copy(GOOGLE_DRIVE_PATH)

# ==============================================================================
# Load CIFAR-10 dataset
# ==============================================================================
print('\n' + '='*80)
print('LOADING CIFAR-10 DATASET')
print('='*80)

from cs7643.cifar10 import CIFAR10

train_ds = CIFAR10(GOOGLE_DRIVE_PATH + '/data/cifar10', download=True, train=True)
test_ds = CIFAR10(GOOGLE_DRIVE_PATH + '/data/cifar10', download=True, train=False)

print(f'Training dataset size: {len(train_ds)}')
print(f'Test dataset size: {len(test_ds)}')

# ==============================================================================
# Test Module Implementations
# ==============================================================================
print('\n' + '='*80)
print('TESTING MODULE IMPLEMENTATIONS')
print('='*80)

# Test Linear
print('\n--- Testing Linear Module ---')
result = subprocess.run([sys.executable, '-m', 'pytest', '-s', 'tests/test_linear.py'],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERRORS:", result.stderr)

# Test MaxPooling
print('\n--- Testing MaxPooling Module ---')
result = subprocess.run([sys.executable, '-m', 'pytest', '-s', 'tests/test_maxpool.py'],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERRORS:", result.stderr)

# Test ReLU
print('\n--- Testing ReLU Module ---')
result = subprocess.run([sys.executable, '-m', 'pytest', '-s', 'tests/test_relu.py'],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERRORS:", result.stderr)

# Test Conv2D
print('\n--- Testing Conv2D Module ---')
result = subprocess.run([sys.executable, '-m', 'pytest', '-s', 'tests/test_conv.py'],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERRORS:", result.stderr)

# ==============================================================================
# Test Optimizer Implementation
# ==============================================================================
print('\n' + '='*80)
print('TESTING OPTIMIZER (SGD WITH MOMENTUM)')
print('='*80)

result = subprocess.run([sys.executable, '-m', 'pytest', '-s', 'tests/test_sgd.py'],
                       capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERRORS:", result.stderr)

# ==============================================================================
# Training Experiment - 50 Samples Overfitting
# ==============================================================================
print('\n' + '='*80)
print('TRAINING EXPERIMENT - 50 SAMPLES OVERFITTING')
print('='*80)

from modules import ConvNet
from optimizer import SGD
from cs7643.solver import Solver
from data import get_CIFAR10_data

# Load processed CIFAR-10 data
root = GOOGLE_DRIVE_PATH + '/data/cifar10/cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(root)

print(f'\nNumber of training samples: {len(X_train)}')
print(f'Training on only 50 samples to verify model can overfit')

# Define network architecture
model_list = [
    dict(type='Conv2D', in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    dict(type='ReLU'),
    dict(type='MaxPooling', kernel_size=2, stride=2),
    dict(type='Linear', in_dim=8192, out_dim=10)
]

criterion = dict(type='SoftmaxCrossEntropy')
model = ConvNet(model_list, criterion)

# Create optimizer with momentum
optimizer = SGD(model, learning_rate=0.0001, reg=0.001, momentum=0.9)

# Create trainer
trainer = Solver()

# Train on only 50 samples
print('\nStarting training...')
print('-' * 80)

loss_history, train_acc_history = trainer.train(
    X_train[:50], y_train[:50], model, batch_size=10, num_epochs=10,
    verbose=True, optimizer=optimizer
)

# Plot results
print('\n' + '='*80)
print('GENERATING PLOT')
print('='*80)

plt.figure(figsize=(10, 6))
plt.plot(train_acc_history, linewidth=2, marker='o')
plt.legend(['train'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training Accuracy on 50 CIFAR-10 Samples')
plt.grid(True, alpha=0.3)
plt.savefig(GOOGLE_DRIVE_PATH + '/train.png', dpi=150, bbox_inches='tight')

print(f'\nPlot saved to: {GOOGLE_DRIVE_PATH}/train.png')

# ==============================================================================
# Summary
# ==============================================================================
print('\n' + '='*80)
print('TRAINING SUMMARY')
print('='*80)

print(f'\nFinal training accuracy: {train_acc_history[-1]:.4f}')
print(f'Target accuracy: ~0.9')

if train_acc_history[-1] >= 0.85:
    print('\n✓ SUCCESS: Model successfully overfits 50 samples!')
else:
    print('\n✗ WARNING: Model may not be learning correctly. Check implementations.')

print('\n' + '='*80)
print('COMPLETE - All tests and training finished')
print('='*80)
print('\nNext steps:')
print('1. Check train.png for the accuracy plot')
print('2. Include train.png in your report')
print('3. If running submission, execute: python -c "from cs7643.submit import make_a2_1_submission; make_a2_1_submission(\'.\')"')
