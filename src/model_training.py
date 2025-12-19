"""
Model Training and Optimization with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates advanced training techniques, callbacks, and optimization strategies.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os

def create_training_callbacks(checkpoint_dir='./checkpoints', log_dir='./logs'):
    """
    Create a set of useful training callbacks.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save TensorBoard logs
    
    Returns:
        List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callback_list = [
        # Model checkpointing
        callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_loss:.2f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ),
        
        # CSV logger
        callbacks.CSVLogger(
            filename=os.path.join(log_dir, 'training.log'),
            append=True
        )
    ]
    
    return callback_list

def create_learning_rate_scheduler(initial_lr=0.001):
    """
    Create a custom learning rate scheduler.
    
    Args:
        initial_lr: Initial learning rate
    
    Returns:
        Learning rate scheduler callback
    """
    def lr_schedule(epoch):
        """Learning rate schedule function."""
        if epoch < 10:
            return initial_lr
        elif epoch < 20:
            return initial_lr * 0.5
        elif epoch < 30:
            return initial_lr * 0.1
        else:
            return initial_lr * 0.01
    
    return callbacks.LearningRateScheduler(lr_schedule, verbose=1)

def train_with_data_augmentation(model, X_train, y_train, X_val, y_val):
    """
    Train model with data augmentation.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Training history
    """
    # Create data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Reshape data if needed (for images)
    if len(X_train.shape) == 2:
        # Assume it's flattened image data
        img_size = int(np.sqrt(X_train.shape[1]))
        X_train = X_train.reshape(-1, img_size, img_size, 1)
        X_val = X_val.reshape(-1, img_size, img_size, 1)
    
    # Fit data generator
    datagen.fit(X_train)
    
    # Create callbacks
    callback_list = create_training_callbacks()
    
    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callback_list,
        verbose=1
    )
    
    return history

def train_with_mixed_precision(model, X_train, y_train, X_val, y_val):
    """
    Train model with mixed precision for faster training.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Training history
    """
    # Enable mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    
    # Compile model with mixed precision
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callback_list = create_training_callbacks()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callback_list,
        verbose=1
    )
    
    return history

def train_with_distributed_strategy(model, X_train, y_train, X_val, y_val):
    """
    Train model using distributed strategy (MirroredStrategy).
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Training history
    """
    # Create distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        # Recompile model within strategy scope
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Create callbacks
    callback_list = create_training_callbacks()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=128 * strategy.num_replicas_in_sync,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callback_list,
        verbose=1
    )
    
    return history

def plot_training_metrics(history):
    """
    Plot comprehensive training metrics.
    
    Args:
        history: Training history from model.fit()
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.show()

def hyperparameter_tuning_example():
    """
    Example of hyperparameter tuning using Keras Tuner.
    """
    try:
        import keras_tuner as kt
    except ImportError:
        print("Warning: keras-tuner not installed. Install it with: pip install keras-tuner")
        return None
    
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        
        # Tune number of layers
        for i in range(hp.Int('num_layers', 2, 5)):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                activation='relu'
            ))
            model.add(layers.Dropout(
                hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
            ))
        
        model.add(layers.Dense(10, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        directory='./tuning',
        project_name='mnist_tuning'
    )
    
    return tuner

def example_usage():
    """
    Example usage of training and optimization functions.
    """
    # Load sample data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
    X_test = X_test.reshape(10000, 784).astype('float32') / 255.0
    
    # Create model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callback_list = create_training_callbacks()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callback_list,
        verbose=1
    )
    
    # Plot metrics
    plot_training_metrics(history)
    
    return model, history

if __name__ == '__main__':
    print("Model Training and Optimization with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model, history = example_usage()
