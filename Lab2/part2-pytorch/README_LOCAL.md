# Local Development Workflow

This guide shows you how to work on Assignment 2 Part 2 locally without using Jupyter notebooks.

## Setup

1. Activate your conda environment:
```bash
conda activate cs7643-a2
```

2. Verify PyTorch is installed:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Workflow

### Step 1: Implement the Models

Edit the model files:
- `models/twolayer.py` - Two-layer fully connected network
- `models/cnn.py` - Vanilla CNN
- `models/my_model.py` - Your custom model

### Step 2: Train the Models

Use the `train.py` script:

```bash
# Train two-layer network
python train.py --config config_twolayer

# Train vanilla CNN
python train.py --config config_vanilla_cnn

# Train your custom model
python train.py --config config_mymodel
```

This will:
- Load the configuration from `configs/`
- Train the model
- Save checkpoint to `checkpoints/`

### Step 3: Test the Models

Use the `test_models.py` script or pytest directly:

```bash
# Test individual models
python test_models.py --model twolayer
python test_models.py --model cnn
python test_models.py --model mymodel

# Or test all at once
python test_models.py --all

# Or use pytest directly
pytest tests/test_twolayer.py -v
pytest tests/test_vanilla_cnn.py -v
pytest tests/test_mymodel.py -v
```

## Expected Accuracies

- **TwoLayerNet**: ~0.30-0.40
- **VanillaCNN**: ~0.40+
- **MyModel**: Must be >0.50 for credit

## Quick Start Example

```bash
# 1. Activate environment
conda activate cs7643-a2

# 2. Implement TwoLayerNet in models/twolayer.py
# (edit the file)

# 3. Train it
python train.py --config config_twolayer

# 4. Test it
python test_models.py --model twolayer

# 5. Repeat for other models
```

## Directory Structure

```
part2-pytorch/
├── train.py              # Training script (replaces notebook cell 21)
├── test_models.py        # Testing script (replaces notebook cells 24-26)
├── solver.py             # Training loop implementation
├── configs/              # Configuration files
│   ├── config_twolayer.yaml
│   ├── config_vanilla_cnn.yaml
│   └── config_mymodel.yaml
├── models/               # Your implementations
│   ├── twolayer.py
│   ├── cnn.py
│   └── my_model.py
├── tests/                # Test files
│   ├── test_twolayer.py
│   ├── test_vanilla_cnn.py
│   └── test_mymodel.py
└── checkpoints/          # Saved model weights
    ├── twolayernet.pth
    ├── vanillacnn.pth
    └── mymodel.pth
```

## Troubleshooting

**Import errors**: Make sure you're in the `part2-pytorch` directory when running scripts.

**Missing dependencies**: Install with `pip install pyyaml pytest`

**No checkpoint file**: Train the model first before testing.

## Benefits of Local Development

✓ Faster iteration - no upload/download delays
✓ Better debugging - use your IDE/editor
✓ No Google Drive mounting
✓ Works offline
✓ Full control over environment
