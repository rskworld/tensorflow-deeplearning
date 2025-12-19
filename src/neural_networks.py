"""
Neural Networks with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates basic neural network construction using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def create_simple_neural_network(input_shape, num_classes):
    """
    Create a simple feedforward neural network.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_deep_neural_network(input_shape, num_classes, hidden_layers=[256, 128, 64]):
    """
    Create a deep neural network with configurable hidden layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        hidden_layers: List of units for each hidden layer
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train a neural network model.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def example_usage():
    """
    Example usage of neural network functions.
    """
    # Load sample data (using MNIST as example)
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
    X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
    
    # Create model
    model = create_simple_neural_network(input_shape=(784,), num_classes=10)
    
    # Display model architecture
    model.summary()
    
    # Train model
    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=5, batch_size=128
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

if __name__ == '__main__':
    print("Neural Networks with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model, history = example_usage()
