"""
Convolutional Classifier Tests.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import unittest
import numpy as np
from modules.conv_classifier import ConvNet
from .utils import *


class TestConvClassifier(unittest.TestCase):
    """ The class containing all test cases for ConvNet classifier"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def test_simple_forward(self):
        """Test forward pass with a simple network"""
        np.random.seed(42)

        # Define a simple network: Conv -> ReLU -> Linear
        modules = [
            {'type': 'Conv2D', 'in_channels': 3, 'out_channels': 8,
             'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_dim': 8 * 5 * 5, 'out_dim': 10}
        ]
        criterion = {'type': 'SoftmaxCrossEntropy'}

        model = ConvNet(modules, criterion)

        # Create sample input: 2 samples, 3 channels, 5x5 images
        x = np.random.randn(2, 3, 5, 5)
        y = np.array([0, 1])

        probs, loss = model.forward(x, y)

        # Check output shapes
        self.assertEqual(probs.shape, (2, 10), "Probs shape should be (2, 10)")
        self.assertIsInstance(loss, (float, np.floating), "Loss should be a scalar")

        # Check probabilities sum to 1
        prob_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(prob_sums, np.ones(2), rtol=1e-5,
                                   err_msg="Probabilities should sum to 1")

        # Check loss is positive
        self.assertGreater(loss, 0, "Loss should be positive")

    def test_forward_with_pooling(self):
        """Test forward pass with pooling layer"""
        np.random.seed(42)

        # Network: Conv -> ReLU -> MaxPool -> Linear
        modules = [
            {'type': 'Conv2D', 'in_channels': 3, 'out_channels': 4,
             'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'MaxPooling', 'kernel_size': 2, 'stride': 2},
            {'type': 'Linear', 'in_dim': 4 * 2 * 2, 'out_dim': 5}
        ]
        criterion = {'type': 'SoftmaxCrossEntropy'}

        model = ConvNet(modules, criterion)

        # Input: 3 samples, 3 channels, 4x4 images
        x = np.random.randn(3, 3, 4, 4)
        y = np.array([0, 2, 4])

        probs, loss = model.forward(x, y)

        self.assertEqual(probs.shape, (3, 5))
        self.assertGreater(loss, 0)

    def test_backward_simple(self):
        """Test backward pass updates gradients"""
        np.random.seed(42)

        modules = [
            {'type': 'Conv2D', 'in_channels': 2, 'out_channels': 3,
             'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_dim': 3 * 4 * 4, 'out_dim': 5}
        ]
        criterion = {'type': 'SoftmaxCrossEntropy'}

        model = ConvNet(modules, criterion)

        x = np.random.randn(2, 2, 4, 4)
        y = np.array([1, 3])

        # Forward pass
        probs, loss = model.forward(x, y)

        # Backward pass
        model.backward()

        # Check that gradients are computed for Conv2D layer
        conv_layer = model.modules[0]
        self.assertIsNotNone(conv_layer.dw, "Conv2D dw should be computed")
        self.assertIsNotNone(conv_layer.db, "Conv2D db should be computed")
        self.assertIsNotNone(conv_layer.dx, "Conv2D dx should be computed")

        # Check gradient shapes
        self.assertEqual(conv_layer.dw.shape, conv_layer.weight.shape)
        self.assertEqual(conv_layer.db.shape, conv_layer.bias.shape)

        # Check that gradients are computed for Linear layer
        linear_layer = model.modules[2]
        self.assertIsNotNone(linear_layer.dw, "Linear dw should be computed")
        self.assertIsNotNone(linear_layer.db, "Linear db should be computed")
        self.assertEqual(linear_layer.dw.shape, linear_layer.weight.shape)
        self.assertEqual(linear_layer.db.shape, linear_layer.bias.shape)

    def test_gradient_check_small_network(self):
        """Numerical gradient check for a small network"""
        np.random.seed(42)

        # Very small network for gradient checking
        modules = [
            {'type': 'Conv2D', 'in_channels': 2, 'out_channels': 2,
             'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_dim': 2 * 2 * 2, 'out_dim': 3}
        ]
        criterion = {'type': 'SoftmaxCrossEntropy'}

        model = ConvNet(modules, criterion)

        # Small input
        x = np.random.randn(2, 2, 3, 3) * 0.1
        y = np.array([0, 2])

        # Compute analytical gradients
        probs, loss = model.forward(x, y)
        model.backward()

        # Get conv layer
        conv_layer = model.modules[0]
        dw_analytical = conv_layer.dw.copy()
        db_analytical = conv_layer.db.copy()

        # Numerical gradient for conv weights
        def loss_fn(w):
            conv_layer.weight = w.reshape(conv_layer.weight.shape)
            probs, loss = model.forward(x, y)
            return loss

        w_flat = conv_layer.weight.flatten()
        dw_numerical = eval_numerical_gradient(loss_fn, w_flat, verbose=False)
        dw_numerical = dw_numerical.reshape(conv_layer.weight.shape)

        # Check gradient error
        error = rel_error(dw_analytical, dw_numerical)
        self.assertLess(error, 1e-3,
                       f"Gradient error {error} is too large for conv weights")

    def test_multiple_conv_layers(self):
        """Test network with multiple conv layers"""
        np.random.seed(42)

        modules = [
            {'type': 'Conv2D', 'in_channels': 3, 'out_channels': 8,
             'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'Conv2D', 'in_channels': 8, 'out_channels': 16,
             'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'MaxPooling', 'kernel_size': 2, 'stride': 2},
            {'type': 'Linear', 'in_dim': 16 * 2 * 2, 'out_dim': 10}
        ]
        criterion = {'type': 'SoftmaxCrossEntropy'}

        model = ConvNet(modules, criterion)

        x = np.random.randn(4, 3, 4, 4)
        y = np.array([0, 1, 5, 9])

        probs, loss = model.forward(x, y)
        model.backward()

        # Check all layers have gradients
        for i, module in enumerate(model.modules):
            if hasattr(module, 'weight'):
                self.assertIsNotNone(module.dw,
                                   f"Module {i} should have dw computed")

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with same seed"""
        np.random.seed(123)

        modules = [
            {'type': 'Conv2D', 'in_channels': 3, 'out_channels': 4,
             'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_dim': 4 * 5 * 5, 'out_dim': 10}
        ]
        criterion = {'type': 'SoftmaxCrossEntropy'}

        model1 = ConvNet(modules, criterion)
        x = np.random.randn(2, 3, 5, 5)
        y = np.array([0, 1])

        probs1, loss1 = model1.forward(x, y)

        # Reset and create new model with same seed
        np.random.seed(123)
        model2 = ConvNet(modules, criterion)
        probs2, loss2 = model2.forward(x, y)

        np.testing.assert_allclose(probs1, probs2, rtol=1e-10,
                                  err_msg="Forward pass should be deterministic")
        np.testing.assert_allclose(loss1, loss2, rtol=1e-10,
                                  err_msg="Loss should be deterministic")
