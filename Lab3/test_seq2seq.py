"""
Unit tests for Seq2Seq model
Fast tests for local verification before gradescope submission
"""

import torch
import torch.nn as nn
import unittest
import sys
import os

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from seq2seq.Encoder import Encoder
from seq2seq.Decoder import Decoder
from seq2seq.Seq2Seq import Seq2Seq


class TestSeq2Seq(unittest.TestCase):
    """Test suite for Seq2Seq model"""

    def setUp(self):
        """Set up test fixtures with small dimensions for fast testing"""
        self.device = torch.device('cpu')
        self.batch_size = 2
        self.seq_len = 5
        self.vocab_size = 10
        self.emb_size = 8
        self.encoder_hidden_size = 16
        self.decoder_hidden_size = 16

    def test_seq2seq_initialization_rnn(self):
        """Test Seq2Seq model initializes correctly with RNN"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="RNN"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="RNN"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.decoder)

    def test_seq2seq_initialization_lstm(self):
        """Test Seq2Seq model initializes correctly with LSTM"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="LSTM"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="LSTM"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.decoder)

    def test_seq2seq_forward_rnn_output_shape(self):
        """Test Seq2Seq forward pass produces correct output shape with RNN"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="RNN"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="RNN"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        # Create random input
        source = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        # Forward pass
        output = model(source)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape,
                        f"Expected output shape {expected_shape}, got {output.shape}")

    def test_seq2seq_forward_lstm_output_shape(self):
        """Test Seq2Seq forward pass produces correct output shape with LSTM"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="LSTM"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="LSTM"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        # Create random input
        source = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        # Forward pass
        output = model(source)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape,
                        f"Expected output shape {expected_shape}, got {output.shape}")

    def test_seq2seq_forward_different_batch_sizes(self):
        """Test Seq2Seq works with different batch sizes"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="RNN"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="RNN"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        for batch_size in [1, 4, 8]:
            with self.subTest(batch_size=batch_size):
                source = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
                output = model(source)
                expected_shape = (batch_size, self.seq_len, self.vocab_size)
                self.assertEqual(output.shape, expected_shape)

    def test_seq2seq_output_is_log_probabilities(self):
        """Test that Seq2Seq output values are log probabilities (negative or zero)"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="RNN"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="RNN"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        source = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = model(source)

        # Log probabilities should be <= 0
        self.assertTrue(torch.all(output <= 0.01),  # Small epsilon for numerical errors
                       "Output should be log probabilities (negative or zero)")

    def test_seq2seq_with_attention(self):
        """Test Seq2Seq with attention mechanism"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="LSTM"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="LSTM",
            attention=True
        )

        model = Seq2Seq(encoder, decoder, self.device)

        source = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = model(source)

        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)

    def test_seq2seq_gradient_flow(self):
        """Test that gradients flow through the model"""
        encoder = Encoder(
            input_size=self.vocab_size,
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            model_type="RNN"
        )
        decoder = Decoder(
            emb_size=self.emb_size,
            encoder_hidden_size=self.encoder_hidden_size,
            decoder_hidden_size=self.decoder_hidden_size,
            output_size=self.vocab_size,
            model_type="RNN"
        )

        model = Seq2Seq(encoder, decoder, self.device)

        source = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        output = model(source)

        # Compute loss
        loss = nn.NLLLoss()(output.view(-1, self.vocab_size), target.view(-1))
        loss.backward()

        # Check that at least some gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        self.assertTrue(has_gradients, "Model should have gradients after backward pass")


if __name__ == '__main__':
    # Run tests with minimal verbosity for speed
    unittest.main(verbosity=2)
