"""
Unit tests for TransformerTranslator
Fast tests for local verification before gradescope submission
"""

import os
import sys
import unittest

import torch
import torch.nn as nn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from Transformer import TransformerTranslator


class TestTransformerTranslator(unittest.TestCase):
    """Test suite for encoder-only TransformerTranslator."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.seq_len = 5
        self.input_size = 11
        self.output_size = 7
        self.hidden_dim = 8
        self.num_heads = 2
        self.dim_feedforward = 16
        self.dim_k = 4
        self.dim_q = 4
        self.dim_v = 4

        self.model = TransformerTranslator(
            input_size=self.input_size,
            output_size=self.output_size,
            device=self.device,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dim_k=self.dim_k,
            dim_q=self.dim_q,
            dim_v=self.dim_v,
            max_length=self.seq_len,
        )

        self.inputs = torch.randint(
            0, self.input_size, (self.batch_size, self.seq_len)
        )

    def test_embed_output_shape(self):
        embeddings = self.model.embed(self.inputs)
        expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        self.assertEqual(embeddings.shape, expected_shape)

    def test_multi_head_attention_output_shape(self):
        embeddings = self.model.embed(self.inputs)
        outputs = self.model.multi_head_attention(embeddings)
        expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_feedforward_output_shape(self):
        embeddings = self.model.embed(self.inputs)
        attn_outputs = self.model.multi_head_attention(embeddings)
        outputs = self.model.feedforward_layer(attn_outputs)
        expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        self.assertEqual(outputs.shape, expected_shape)

    def test_forward_output_shape(self):
        outputs = self.model(self.inputs)
        expected_shape = (self.batch_size, self.seq_len, self.output_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_forward_outputs_logits(self):
        outputs = self.model(self.inputs)
        probs = torch.softmax(outputs, dim=2)
        prob_sums = probs.sum(dim=2)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))

    def test_gradient_flow(self):
        outputs = self.model(self.inputs)
        target = torch.randint(0, self.output_size, (self.batch_size, self.seq_len))
        loss = nn.CrossEntropyLoss()(outputs.view(-1, self.output_size), target.view(-1))
        loss.backward()

        self.assertIsNotNone(self.model.embeddingL.weight.grad)
        self.assertIsNotNone(self.model.linear_out.weight.grad)


if __name__ == "__main__":
    unittest.main()
