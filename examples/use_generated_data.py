"""
Example: Using Generated Data
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates how to use the generated data with different models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from src.neural_networks import create_simple_neural_network
from src.cnns import create_simple_cnn
from src.rnns import create_lstm_model

def example_classification():
    """Example using classification data."""
    print("=" * 60)
    print("Example: Classification with Generated Data")
    print("=" * 60)
    
    # Load data
    X = np.load('./data/classification_X.npy')
    y = np.load('./data/classification_y.npy')
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    model = create_simple_neural_network(
        input_shape=(X.shape[1],),
        num_classes=len(np.unique(y))
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return model, history

def example_image_classification():
    """Example using image data."""
    print("\n" + "=" * 60)
    print("Example: Image Classification with Generated Data")
    print("=" * 60)
    
    # Load data
    X = np.load('./data/images_X.npy')
    y = np.load('./data/images_y.npy')
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Reshape for CNN (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    model = create_simple_cnn(
        input_shape=(X.shape[1], X.shape[2], X.shape[3]),
        num_classes=len(np.unique(y))
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=5,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return model, history

def example_sequence_classification():
    """Example using sequence data."""
    print("\n" + "=" * 60)
    print("Example: Sequence Classification with Generated Data")
    print("=" * 60)
    
    # Load data
    X = np.load('./data/sequences_X.npy')
    y = np.load('./data/sequences_y.npy')
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    model = create_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=len(np.unique(y)),
        lstm_units=64,
        num_layers=2
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    return model, history

def main():
    """Run all examples."""
    print("Using Generated Data Examples")
    print("Author: RSK World - https://rskworld.in")
    print()
    
    # Check if data exists
    if not os.path.exists('./data/classification_X.npy'):
        print("Data not found! Generating data first...")
        from src.data_generator import generate_all_sample_data
        generate_all_sample_data()
        print()
    
    # Run examples
    try:
        example_classification()
    except Exception as e:
        print(f"Error in classification example: {e}")
    
    try:
        example_image_classification()
    except Exception as e:
        print(f"Error in image classification example: {e}")
    
    try:
        example_sequence_classification()
    except Exception as e:
        print(f"Error in sequence classification example: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
