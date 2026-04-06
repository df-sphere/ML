#!/usr/bin/env python3
"""
Validates the DDPM noise budget: alpha_bar_T should be < 0.01.
Only depends on timesteps, noise_start, and noise_end.

Usage:
    python tune_diffusion.py                                      # check current config
    python tune_diffusion.py --timesteps 200 --noise_end 0.02     # test overrides
    python tune_diffusion.py --sweep                               # sweep combos
"""

import argparse
import itertools
import yaml
import torch
import numpy as np


def compute_alpha_bar_T(timesteps, noise_start, noise_end):
    beta = torch.linspace(noise_start, noise_end, steps=timesteps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha_bar[-1].item(), alpha_bar


def validate(timesteps, noise_start, noise_end, verbose=True):
    alpha_bar_T, alpha_bar = compute_alpha_bar_T(timesteps, noise_start, noise_end)
    passed = alpha_bar_T < 0.01

    if verbose:
        print(f"  timesteps={timesteps}, noise_start={noise_start}, noise_end={noise_end}")
        print(f"  alpha_bar_T = {alpha_bar_T:.6f}")
        print(f"  Target: < 0.01  ->  {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"  Signal remaining at T: {alpha_bar_T*100:.2f}% (should be < 1%)")
            below = (alpha_bar < 0.01).nonzero(as_tuple=True)[0]
            if len(below) > 0:
                print(f"  alpha_bar drops below 0.01 at step {below[0].item()+1}/{timesteps}")
            else:
                print(f"  alpha_bar NEVER drops below 0.01 with these settings!")
            # Suggest fixes
            print("\n  SUGGESTIONS:")
            for trial_end in np.arange(noise_end + 0.005, 1.0, 0.005):
                if compute_alpha_bar_T(timesteps, noise_start, trial_end)[0] < 0.01:
                    print(f"    - Keep timesteps={timesteps}, increase noise_end to ~{trial_end:.4f}")
                    break
            for trial_steps in range(timesteps + 10, 2000, 10):
                if compute_alpha_bar_T(trial_steps, noise_start, noise_end)[0] < 0.01:
                    print(f"    - Keep noise_end={noise_end}, increase timesteps to ~{trial_steps}")
                    break
        print()

    return passed, alpha_bar_T


def sweep(noise_start=0.0001):
    timesteps_range = [100, 150, 200, 300, 400, 500]
    noise_end_range = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

    print(f"{'timesteps':>10} {'noise_end':>10} {'alpha_bar_T':>12} {'status':>8}")
    print("-" * 45)

    good = []
    for ts, ne in itertools.product(timesteps_range, noise_end_range):
        _, val = compute_alpha_bar_T(ts, noise_start, ne)
        passed = val < 0.01
        print(f"{ts:>10} {ne:>10.4f} {val:>12.6f} {'PASS' if passed else 'FAIL':>8}")
        if passed:
            good.append((ts, ne, val))

    print(f"\n{len(good)} passing configurations.")
    if good:
        good.sort(key=lambda x: x[2])
        print("\nTop 5 closest to zero:")
        for ts, ne, val in good[:5]:
            print(f"  timesteps={ts}, noise_end={ne:.4f}, alpha_bar_T={val:.6f}")


def main():
    parser = argparse.ArgumentParser(description="DDPM noise budget validator")
    parser.add_argument("--config", default="configs/config_diffusion.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--noise_start", type=float, default=None)
    parser.add_argument("--noise_end", type=float, default=None)
    parser.add_argument("--sweep", action="store_true", help="Sweep parameter combos")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    timesteps = args.timesteps or cfg['diffusion']['timesteps']
    noise_start = args.noise_start or cfg['diffusion']['noise_start']
    noise_end = args.noise_end or cfg['diffusion']['noise_end']

    if args.sweep:
        sweep(noise_start)
    else:
        validate(timesteps, noise_start, noise_end)


if __name__ == "__main__":
    main()
