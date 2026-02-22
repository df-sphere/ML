#!/usr/bin/env python3
"""
Simple test script to verify trained models locally.

Usage:
    python test_models.py --model twolayer
    python test_models.py --model cnn
    python test_models.py --model mymodel
    python test_models.py --all  # Test all models
"""

import argparse
import subprocess
import sys


def run_pytest(test_file):
    """Run pytest on a specific test file."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    result = subprocess.run(['pytest', '-v', test_file], capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Test trained PyTorch models')
    parser.add_argument('--model', type=str, choices=['twolayer', 'cnn', 'mymodel'],
                        help='Which model to test')
    parser.add_argument('--all', action='store_true',
                        help='Test all models')
    args = parser.parse_args()

    tests = {
        'twolayer': 'tests/test_twolayer.py',
        'cnn': 'tests/test_vanilla_cnn.py',
        'mymodel': 'tests/test_mymodel.py'
    }

    if args.all:
        # Run all tests
        print("Testing all models...")
        failed = []
        for model, test_file in tests.items():
            if run_pytest(test_file) != 0:
                failed.append(model)

        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        if failed:
            print(f"Failed tests: {', '.join(failed)}")
            sys.exit(1)
        else:
            print("All tests passed!")
            sys.exit(0)

    elif args.model:
        # Run specific test
        test_file = tests[args.model]
        sys.exit(run_pytest(test_file))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
