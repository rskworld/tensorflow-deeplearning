"""
Tests for Neural Networks Module
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.neural_networks import (
    create_simple_neural_network,
    create_deep_neural_network
)

class TestNeuralNetworks(unittest.TestCase):
    """Test cases for neural networks module."""
    
    def test_create_simple_neural_network(self):
        """Test simple neural network creation."""
        model = create_simple_neural_network(input_shape=(784,), num_classes=10)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 784))
        self.assertEqual(model.output_shape, (None, 10))
    
    def test_create_deep_neural_network(self):
        """Test deep neural network creation."""
        model = create_deep_neural_network(
            input_shape=(784,),
            num_classes=10,
            hidden_layers=[256, 128, 64]
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 784))
        self.assertEqual(model.output_shape, (None, 10))
    
    def test_model_prediction(self):
        """Test model prediction."""
        model = create_simple_neural_network(input_shape=(784,), num_classes=10)
        test_input = np.random.randn(1, 784).astype('float32')
        prediction = model.predict(test_input, verbose=0)
        self.assertEqual(prediction.shape, (1, 10))
        self.assertAlmostEqual(np.sum(prediction[0]), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
