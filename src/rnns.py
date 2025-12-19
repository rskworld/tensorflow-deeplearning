"""
Recurrent Neural Networks (RNNs) with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates RNN, LSTM, and GRU implementations for sequence modeling.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def create_simple_rnn(input_shape, num_classes, rnn_units=128):
    """
    Create a simple RNN for sequence classification.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        num_classes: Number of output classes
        rnn_units: Number of RNN units
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.SimpleRNN(rnn_units, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(input_shape, num_classes, lstm_units=128, num_layers=2):
    """
    Create an LSTM model for sequence classification.
    
    Args:
        input_shape: Shape of input sequences
        num_classes: Number of output classes
        lstm_units: Number of LSTM units per layer
        num_layers: Number of LSTM layers
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(layers.LSTM(lstm_units, return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(layers.Dropout(0.2))
    
    for _ in range(num_layers - 2):
        model.add(layers.LSTM(lstm_units, return_sequences=True))
        model.add(layers.Dropout(0.2))
    
    if num_layers > 1:
        model.add(layers.LSTM(lstm_units))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_gru_model(input_shape, num_classes, gru_units=128, num_layers=2):
    """
    Create a GRU model for sequence classification.
    
    Args:
        input_shape: Shape of input sequences
        num_classes: Number of output classes
        gru_units: Number of GRU units per layer
        num_layers: Number of GRU layers
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(layers.GRU(gru_units, return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(layers.Dropout(0.2))
    
    for _ in range(num_layers - 2):
        model.add(layers.GRU(gru_units, return_sequences=True))
        model.add(layers.Dropout(0.2))
    
    if num_layers > 1:
        model.add(layers.GRU(gru_units))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_bidirectional_lstm(input_shape, num_classes, lstm_units=128):
    """
    Create a bidirectional LSTM model.
    
    Args:
        input_shape: Shape of input sequences
        num_classes: Number of output classes
        lstm_units: Number of LSTM units
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_sequence_to_sequence_model(vocab_size, embedding_dim=128, lstm_units=256):
    """
    Create a sequence-to-sequence model for text generation or translation.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(0.2),
        layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_sequence_data(num_samples=1000, sequence_length=50, num_features=10, num_classes=3):
    """
    Generate synthetic sequence data for testing.
    
    Args:
        num_samples: Number of samples
        sequence_length: Length of each sequence
        num_features: Number of features per timestep
        num_classes: Number of classes
    
    Returns:
        Tuple of (X, y) arrays
    """
    X = np.random.randn(num_samples, sequence_length, num_features)
    y = np.random.randint(0, num_classes, num_samples)
    
    return X, y

def example_usage():
    """
    Example usage of RNN functions.
    """
    # Generate synthetic sequence data
    X_train, y_train = generate_sequence_data(num_samples=800, sequence_length=50, num_features=10, num_classes=3)
    X_test, y_test = generate_sequence_data(num_samples=200, sequence_length=50, num_features=10, num_classes=3)
    
    # Create LSTM model
    model = create_lstm_model(input_shape=(50, 10), num_classes=3, lstm_units=128, num_layers=2)
    
    # Display model architecture
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    return model, history

if __name__ == '__main__':
    print("Recurrent Neural Networks with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model, history = example_usage()
