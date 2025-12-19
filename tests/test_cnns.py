"""
Tests for CNNs Module
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
from src.cnns import (
    create_simple_cnn,
    create_deep_cnn
)

class TestCNNs(unittest.TestCase):
    """Test cases for CNNs module."""
    
    def test_create_simple_cnn(self):
        """Test simple CNN creation."""
        model = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
    
    def test_create_deep_cnn(self):
        """Test deep CNN creation."""
        model = create_deep_cnn(input_shape=(32, 32, 3), num_classes=10)
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
    
    def test_cnn_prediction(self):
        """Test CNN prediction."""
        model = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
        test_input = np.random.randn(1, 32, 32, 3).astype('float32')
        prediction = model.predict(test_input, verbose=0)
        self.assertEqual(prediction.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()
