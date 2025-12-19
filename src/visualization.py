"""
Visualization Utilities for TensorFlow Deep Learning
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides comprehensive visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras

def plot_training_history(history, figsize=(15, 5)):
    """
    Plot comprehensive training history.
    
    Args:
        history: Training history from model.fit()
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Accuracy
    if 'accuracy' in history.history:
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[2].plot(history.history['lr'], label='Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_yscale('log')
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_model_architecture(model, show_shapes=True, show_layer_names=True, rankdir='TB'):
    """
    Visualize model architecture.
    
    Args:
        model: Keras model
        show_shapes: Show input/output shapes
        show_layer_names: Show layer names
        rankdir: Graph direction ('TB', 'LR')
    """
    try:
        keras.utils.plot_model(
            model,
            to_file='model_architecture.png',
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            rankdir=rankdir
        )
        print("Model architecture saved to 'model_architecture.png'")
    except Exception as e:
        print(f"Could not generate model plot: {e}")
        print("Model summary:")
        model.summary()

def plot_feature_importance(importance_scores, feature_names=None, top_n=20, figsize=(10, 6)):
    """
    Plot feature importance.
    
    Args:
        importance_scores: Array of importance scores
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance_scores))]
    
    # Get top N features
    indices = np.argsort(importance_scores)[-top_n:][::-1]
    top_scores = importance_scores[indices]
    top_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_scores)), top_scores)
    plt.yticks(range(len(top_scores)), top_names)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, figsize=(10, 6)):
    """
    Plot predictions vs actual values for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_layer_activations(model, sample_input, layer_names=None, figsize=(15, 10)):
    """
    Visualize activations of different layers.
    
    Args:
        model: Keras model
        sample_input: Sample input data
        layer_names: List of layer names to visualize
        figsize: Figure size
    """
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers if 'conv' in layer.name.lower() or 'dense' in layer.name.lower()]
    
    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(sample_input, verbose=0)
    
    fig, axes = plt.subplots(len(layer_names), 1, figsize=figsize)
    if len(layer_names) == 1:
        axes = [axes]
    
    for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        ax = axes[i]
        if len(activation.shape) == 4:  # Convolutional layer
            # Show first filter of first channel
            ax.imshow(activation[0, :, :, 0], cmap='viridis')
        elif len(activation.shape) == 2:  # Dense layer
            # Show as bar chart
            ax.bar(range(len(activation[0])), activation[0])
        ax.set_title(f'Layer: {layer_name}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_loss_landscape(model, X_train, y_train, resolution=20, figsize=(10, 8)):
    """
    Plot loss landscape (simplified 2D projection).
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        resolution: Resolution of the grid
        figsize: Figure size
    """
    # This is a simplified version - full loss landscape requires more complex implementation
    print("Note: This is a simplified loss landscape visualization")
    
    # Get model weights
    weights = model.get_weights()
    
    if len(weights) < 2:
        print("Model needs at least 2 weight matrices for visualization")
        return
    
    # Create a 2D grid around current weights
    w1_flat = weights[0].flatten()
    w2_flat = weights[1].flatten() if len(weights) > 1 else weights[0].flatten()
    
    # Sample a subset for visualization
    sample_size = min(100, len(w1_flat))
    indices = np.random.choice(len(w1_flat), sample_size, replace=False)
    
    plt.figure(figsize=figsize)
    plt.scatter(w1_flat[indices], w2_flat[indices], alpha=0.5)
    plt.xlabel('Weight Dimension 1')
    plt.ylabel('Weight Dimension 2')
    plt.title('Weight Distribution (2D Projection)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_training_animation(history, save_path='training_animation.gif'):
    """
    Create animated plot of training progress.
    
    Args:
        history: Training history
        save_path: Path to save animation
    """
    try:
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_data = []
        y_data = []
        line, = ax.plot([], [], 'b-')
        
        def animate(frame):
            if frame < len(history.history['loss']):
                x_data.append(frame)
                y_data.append(history.history['loss'][frame])
                line.set_data(x_data, y_data)
                ax.relim()
                ax.autoscale_view()
                ax.set_title(f'Training Loss - Epoch {frame + 1}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
        
        anim = FuncAnimation(fig, animate, frames=len(history.history['loss']), interval=200, repeat=False)
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Could not create animation: {e}")

def plot_data_distribution(data, labels=None, figsize=(12, 4)):
    """
    Plot data distribution.
    
    Args:
        data: Data array
        labels: Data labels (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(data.flatten(), bins=50, edgecolor='black')
    axes[0].set_title('Data Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    if labels is not None:
        data_by_label = [data[labels == i] for i in np.unique(labels)]
        axes[1].boxplot(data_by_label, labels=np.unique(labels))
    else:
        axes[1].boxplot([data.flatten()])
    axes[1].set_title('Data Distribution by Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def example_usage():
    """
    Example usage of visualization functions.
    """
    # Create dummy history
    history = type('obj', (object,), {
        'history': {
            'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
            'val_loss': [0.6, 0.45, 0.35, 0.3, 0.25],
            'accuracy': [0.8, 0.85, 0.9, 0.92, 0.95],
            'val_accuracy': [0.75, 0.82, 0.88, 0.9, 0.93]
        }
    })()
    
    print("Plotting training history...")
    plot_training_history(history)
    
    print("\nVisualization utilities ready!")

if __name__ == '__main__':
    print("Visualization Utilities for TensorFlow Deep Learning")
    print("Author: RSK World - https://rskworld.in")
    example_usage()
