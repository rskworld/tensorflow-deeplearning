"""
Example: Train a Custom Model
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates how to train a custom model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from src.model_training import create_training_callbacks
from src.model_evaluation import ModelEvaluator

def create_example_model(input_shape, num_classes):
    """Create an example model."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main training function."""
    print("=" * 60)
    print("Training Custom Model Example")
    print("Author: RSK World - https://rskworld.in")
    print("=" * 60)
    
    # Load data (using MNIST as example)
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
    X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
    
    # Create model
    model = create_example_model(input_shape=(784,), num_classes=10)
    model.summary()
    
    # Create callbacks
    callbacks = create_training_callbacks()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=5,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate_classification(X_test, y_test)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")
    
    # Save model
    model.save('./models/custom_model.h5')
    print("\nModel saved to ./models/custom_model.h5")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
