"""
Convolutional Neural Networks (CNNs) with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates CNN construction for image classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def create_simple_cnn(input_shape, num_classes):
    """
    Create a simple CNN for image classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_deep_cnn(input_shape, num_classes):
    """
    Create a deeper CNN with batch normalization and dropout.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_resnet_block(x, filters, kernel_size=3, stride=1):
    """
    Create a ResNet-style residual block.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
    
    Returns:
        Output tensor
    """
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_resnet_cnn(input_shape, num_classes):
    """
    Create a ResNet-inspired CNN using functional API.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = create_resnet_block(x, 64)
    x = create_resnet_block(x, 64)
    x = create_resnet_block(x, 128, stride=2)
    x = create_resnet_block(x, 128)
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_cnn_layers(model, sample_image):
    """
    Visualize CNN layer activations.
    
    Args:
        model: Trained CNN model
        sample_image: Sample image to visualize
    """
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(sample_image.reshape(1, *sample_image.shape))
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.ravel()
    
    for i, activation in enumerate(activations):
        if len(activation.shape) == 4:  # Convolutional layer
            # Take first filter
            axes[i].imshow(activation[0, :, :, 0], cmap='viridis')
            axes[i].set_title(f'Layer {i+1}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def example_usage():
    """
    Example usage of CNN functions.
    """
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Preprocess data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create model
    model = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
    
    # Display model architecture
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {test_accuracy:.4f}')
    
    return model, history

if __name__ == '__main__':
    print("Convolutional Neural Networks with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model, history = example_usage()
