"""
Helper Utilities for TensorFlow Deep Learning Project
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module contains utility functions for data preprocessing, visualization, and common operations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras

def normalize_data(X, method='standard'):
    """
    Normalize data using different methods.
    
    Args:
        X: Input data
        method: Normalization method ('standard', 'minmax', or 'unit')
    
    Returns:
        Normalized data and scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'unit':
        # Unit normalization (divide by L2 norm)
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X_normalized, None
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if len(X.shape) == 2:
        X_normalized = scaler.fit_transform(X)
    else:
        # For higher dimensional data, flatten first
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        X_normalized = scaler.fit_transform(X_flat)
        X_normalized = X_normalized.reshape(original_shape)
    
    return X_normalized, scaler

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        val_size: Proportion of validation set (from training set)
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_classification_report(y_true, y_pred, class_names=None):
    """
    Plot classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    
    data = {metric: [report[cls][metric] for cls in classes] for metric in metrics}
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        ax.bar(x + offset, data[metric], width, label=metric)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=10, class_names=None):
    """
    Visualize model predictions on test samples.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        num_samples: Number of samples to visualize
        class_names: List of class names
    """
    # Get predictions
    predictions = model.predict(X_test[:num_samples], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Determine grid size
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image (assuming it's image data)
        if len(X_test[i].shape) == 2:
            ax.imshow(X_test[i], cmap='gray')
        elif len(X_test[i].shape) == 3:
            ax.imshow(X_test[i])
        else:
            # For flattened data, try to reshape
            img_size = int(np.sqrt(X_test[i].shape[0]))
            if img_size * img_size == X_test[i].shape[0]:
                ax.imshow(X_test[i].reshape(img_size, img_size), cmap='gray')
        
        # Add title with prediction
        true_label = y_test[i] if isinstance(y_test[i], (int, np.integer)) else np.argmax(y_test[i])
        pred_label = predicted_classes[i]
        confidence = predictions[i][pred_label]
        
        true_name = class_names[true_label] if class_names else str(true_label)
        pred_name = class_names[pred_label] if class_names else str(pred_label)
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_name}\nPred: {pred_name} ({confidence:.2f})', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_model_summary_dict(model):
    """
    Get model summary as a dictionary.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model information
    """
    summary = {
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'num_layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    return summary

def save_training_history(history, filepath='./training_history.json'):
    """
    Save training history to JSON file.
    
    Args:
        history: Training history from model.fit()
        filepath: Path to save history
    """
    import json
    
    # Convert numpy types to native Python types
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {filepath}")

def load_training_history(filepath='./training_history.json'):
    """
    Load training history from JSON file.
    
    Args:
        filepath: Path to history file
    
    Returns:
        Dictionary with training history
    """
    import json
    
    with open(filepath, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict

def create_data_generator(X, y, batch_size=32, shuffle=True):
    """
    Create a data generator for training.
    
    Args:
        X: Features
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
    
    Yields:
        Batches of (X_batch, y_batch)
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]

def calculate_model_size(model, filepath=None):
    """
    Calculate model size in MB.
    
    Args:
        model: Keras model
        filepath: Optional path to saved model file
    
    Returns:
        Model size in MB
    """
    if filepath and os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
    else:
        # Estimate from model parameters
        total_params = model.count_params()
        # Assume float32 (4 bytes per parameter)
        size_bytes = total_params * 4
    
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

if __name__ == '__main__':
    print("Helper Utilities for TensorFlow Deep Learning")
    print("Author: RSK World - https://rskworld.in")
