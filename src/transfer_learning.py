"""
Transfer Learning with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates transfer learning using pre-trained models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np
import matplotlib.pyplot as plt

def create_transfer_learning_model(base_model_name='VGG16', num_classes=10, input_shape=(224, 224, 3)):
    """
    Create a transfer learning model using pre-trained base models.
    
    Args:
        base_model_name: Name of pre-trained model ('VGG16', 'ResNet50', 'MobileNet', 'InceptionV3')
        num_classes: Number of output classes
        input_shape: Input image shape
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained base model
    base_models = {
        'VGG16': applications.VGG16,
        'ResNet50': applications.ResNet50,
        'MobileNet': applications.MobileNet,
        'InceptionV3': applications.InceptionV3,
        'Xception': applications.Xception,
        'DenseNet121': applications.DenseNet121
    }
    
    if base_model_name not in base_models:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Create base model
    base_model = base_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def fine_tune_model(model, base_model, unfreeze_layers=10):
    """
    Fine-tune a transfer learning model by unfreezing some layers.
    
    Args:
        model: Transfer learning model
        base_model: Base pre-trained model
        unfreeze_layers: Number of top layers to unfreeze
    
    Returns:
        Recompiled model ready for fine-tuning
    """
    # Unfreeze top layers
    base_model.trainable = True
    
    # Freeze all layers except the last N layers
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_feature_extractor(base_model_name='VGG16', input_shape=(224, 224, 3)):
    """
    Create a feature extractor using pre-trained model.
    
    Args:
        base_model_name: Name of pre-trained model
        input_shape: Input image shape
    
    Returns:
        Feature extractor model
    """
    base_models = {
        'VGG16': applications.VGG16,
        'ResNet50': applications.ResNet50,
        'MobileNet': applications.MobileNet,
        'InceptionV3': applications.InceptionV3
    }
    
    base_model = base_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    # Add global pooling
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    feature_extractor = keras.Model(inputs, x)
    
    return feature_extractor

def extract_features(model, images, batch_size=32):
    """
    Extract features from images using a pre-trained model.
    
    Args:
        model: Feature extractor model
        images: Input images
        batch_size: Batch size for processing
    
    Returns:
        Extracted features
    """
    features = model.predict(images, batch_size=batch_size, verbose=0)
    return features

def example_usage():
    """
    Example usage of transfer learning functions.
    """
    # Create transfer learning model
    model, base_model = create_transfer_learning_model(
        base_model_name='MobileNet',
        num_classes=10,
        input_shape=(224, 224, 3)
    )
    
    print("Transfer Learning Model:")
    model.summary()
    
    # Generate dummy data for demonstration
    X_train = np.random.rand(100, 224, 224, 3).astype('float32')
    y_train = np.random.randint(0, 10, 100)
    
    # Train model (initial training with frozen base)
    print("\nTraining with frozen base model...")
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=2,
        verbose=1
    )
    
    # Fine-tune model
    print("\nFine-tuning model...")
    model = fine_tune_model(model, base_model, unfreeze_layers=10)
    
    # Continue training with fine-tuning
    history_finetune = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=2,
        verbose=1
    )
    
    return model, history, history_finetune

if __name__ == '__main__':
    print("Transfer Learning with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model, history, history_finetune = example_usage()
