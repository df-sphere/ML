#!/usr/bin/env python3
"""
Simple test for MyModel
Usage: python test_mymodel_blocks.py
"""

import torch
from models.my_model import MyModel


print("="*60)
print("Testing MyModel (2 Blocks)")
print("="*60)

# Create model
model = MyModel()
model.eval()

# Show architecture
print("\nModel Architecture:")
print("-"*60)
print(model)
print("-"*60)

# Test with dummy CIFAR-10 batch
batch_size = 4
dummy_input = torch.randn(batch_size, 3, 32, 32)

print(f"\nInput shape: {dummy_input.shape}")

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
print(f"Expected: torch.Size([{batch_size}, 10])")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")

# Verify output shape
assert output.shape == torch.Size([batch_size, 10]), f"Shape mismatch! Got {output.shape}"

print("\n" + "="*60)
print("✓ TEST PASSED!")
print("="*60)
