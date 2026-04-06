import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tune_diffusion import compute_alpha_bar_T, validate


class TestComputeAlphaBarT(unittest.TestCase):
    """Verify compute_alpha_bar_T matches the trainer's implementation exactly."""

    def test_matches_trainer_logic(self):
        """Compare against the same math used in DiffusionTrainer.__init__."""
        timesteps, noise_start, noise_end = 120, 0.0001, 0.12

        # Trainer's logic (from trainer_diffusion.py lines 43-45)
        beta = torch.linspace(noise_start, noise_end, steps=timesteps)
        alpha = 1 - beta
        alphas_bar = torch.cumprod(alpha, dim=0)
        expected = alphas_bar[-1].item()

        # Our script
        result, _ = compute_alpha_bar_T(timesteps, noise_start, noise_end)
        self.assertAlmostEqual(result, expected, places=8)

    def test_matches_trainer_full_schedule(self):
        """Verify the full alpha_bar schedule, not just the last value."""
        timesteps, noise_start, noise_end = 10, 0.0001, 0.02

        beta = torch.linspace(noise_start, noise_end, steps=timesteps)
        alpha = 1 - beta
        expected_alphas_bar = torch.cumprod(alpha, dim=0)

        _, alpha_bar = compute_alpha_bar_T(timesteps, noise_start, noise_end)
        self.assertTrue(torch.allclose(alpha_bar, expected_alphas_bar, atol=1e-8))

    def test_known_values_from_test_assets(self):
        """Cross-check against the expected values in test_diffusion.py.
        That test uses timesteps=10, noise_start=0.0001, noise_end=0.02."""
        timesteps, noise_start, noise_end = 10, 0.0001, 0.02

        _, alpha_bar = compute_alpha_bar_T(timesteps, noise_start, noise_end)

        # Expected from tests/test_diffusion.py lines 33-35
        expected = torch.tensor([
            0.9999, 0.9976, 0.9931, 0.9864, 0.9776,
            0.9667, 0.9537, 0.9389, 0.9222, 0.9037
        ])
        self.assertTrue(torch.allclose(alpha_bar, expected, atol=1e-3))


class TestValidationRule(unittest.TestCase):
    """Test the alpha_bar_T < 0.01 validation rule."""

    def test_current_config_passes(self):
        """Current config (timesteps=120, noise_end=0.12) should pass."""
        passed, val = validate(120, 0.0001, 0.12, verbose=False)
        self.assertTrue(passed)
        self.assertLess(val, 0.01)

    def test_low_noise_fails(self):
        """A weak noise schedule should fail the check."""
        passed, val = validate(100, 0.0001, 0.02, verbose=False)
        self.assertFalse(passed)
        self.assertGreater(val, 0.01)

    def test_high_noise_passes(self):
        """A strong noise schedule should pass."""
        passed, val = validate(500, 0.0001, 0.20, verbose=False)
        self.assertTrue(passed)
        self.assertLess(val, 0.001)

    def test_boundary_near_threshold(self):
        """Test near the 0.01 boundary."""
        # timesteps=100, noise_end=0.08 gives ~0.016 (FAIL)
        passed, val = validate(100, 0.0001, 0.08, verbose=False)
        self.assertFalse(passed)

        # timesteps=100, noise_end=0.10 gives ~0.005 (PASS)
        passed, val = validate(100, 0.0001, 0.10, verbose=False)
        self.assertTrue(passed)

    def test_single_timestep(self):
        """With 1 timestep, alpha_bar_T = 1 - noise_start (linspace returns noise_start)."""
        val, _ = compute_alpha_bar_T(1, 0.0001, 0.5)
        # linspace(0.0001, 0.5, steps=1) = [0.0001], so alpha_bar = 1 - 0.0001 = 0.9999
        self.assertAlmostEqual(val, 0.9999, places=4)

    def test_alpha_bar_decreases_monotonically(self):
        """alpha_bar should always decrease over timesteps."""
        _, alpha_bar = compute_alpha_bar_T(200, 0.0001, 0.10)
        diffs = alpha_bar[1:] - alpha_bar[:-1]
        self.assertTrue((diffs < 0).all(), "alpha_bar should be strictly decreasing")


if __name__ == '__main__':
    unittest.main()
