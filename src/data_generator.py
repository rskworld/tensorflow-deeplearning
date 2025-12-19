"""
Data Generation Utilities for TensorFlow Deep Learning Project
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utilities to generate synthetic datasets for testing and demonstration.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def generate_classification_data(n_samples=1000, n_features=20, n_classes=3, noise=0.1):
    """
    Generate synthetic classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        noise: Noise level (0-1)
    
    Returns:
        X: Feature matrix, y: Labels
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels based on feature combinations
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Create class boundaries based on feature values
        feature_sum = np.sum(X[i, :n_features//2])
        if feature_sum < -2:
            y[i] = 0
        elif feature_sum < 2:
            y[i] = 1
        else:
            y[i] = 2
    
    # Add noise
    noise_mask = np.random.random(n_samples) < noise
    y[noise_mask] = np.random.randint(0, n_classes, np.sum(noise_mask))
    
    return X.astype('float32'), y

def generate_regression_data(n_samples=1000, n_features=10, noise=0.1):
    """
    Generate synthetic regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
    
    Returns:
        X: Feature matrix, y: Target values
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target as linear combination with some non-linearity
    coefficients = np.random.randn(n_features)
    y = np.dot(X, coefficients)
    y += 0.1 * np.sum(X ** 2, axis=1)  # Add non-linear term
    y += noise * np.random.randn(n_samples)  # Add noise
    
    return X.astype('float32'), y.astype('float32')

def generate_image_data(n_samples=100, img_size=(28, 28), n_classes=10):
    """
    Generate synthetic image dataset.
    
    Args:
        n_samples: Number of images
        img_size: Image dimensions (height, width)
        n_classes: Number of classes
    
    Returns:
        X: Image array, y: Labels
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        label = i % n_classes
        y.append(label)
        
        # Generate image with patterns based on label
        img = np.zeros(img_size, dtype='float32')
        
        # Create different patterns for different classes
        if label % 2 == 0:
            # Horizontal lines
            for j in range(0, img_size[0], 3):
                img[j, :] = 0.8
        else:
            # Vertical lines
            for j in range(0, img_size[1], 3):
                img[:, j] = 0.8
        
        # Add class-specific pattern
        center = (img_size[0] // 2, img_size[1] // 2)
        radius = 5 + label
        y_coords, x_coords = np.ogrid[:img_size[0], :img_size[1]]
        mask = (x_coords - center[1])**2 + (y_coords - center[0])**2 <= radius**2
        img[mask] = 1.0 - img[mask]
        
        # Add noise
        img += np.random.randn(*img_size) * 0.1
        img = np.clip(img, 0, 1)
        
        X.append(img)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def generate_sequence_data(n_samples=1000, sequence_length=50, n_features=10, n_classes=3):
    """
    Generate synthetic sequence data for RNN/LSTM models.
    
    Args:
        n_samples: Number of sequences
        sequence_length: Length of each sequence
        n_features: Number of features per timestep
        n_classes: Number of classes
    
    Returns:
        X: Sequence data, y: Labels
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        label = i % n_classes
        y.append(label)
        
        # Generate sequence with class-specific patterns
        sequence = np.random.randn(sequence_length, n_features)
        
        # Add class-specific trend
        trend = np.linspace(0, label * 0.5, sequence_length)
        sequence[:, 0] += trend
        
        # Add periodic pattern for some classes
        if label == 1:
            periodic = np.sin(np.linspace(0, 4 * np.pi, sequence_length))
            sequence[:, 1] += periodic * 0.5
        
        X.append(sequence)
    
    X = np.array(X).astype('float32')
    y = np.array(y)
    
    return X, y

def generate_tabular_data(n_samples=1000, save_path='./data/synthetic_tabular.csv'):
    """
    Generate synthetic tabular dataset and save to CSV.
    
    Args:
        n_samples: Number of samples
        save_path: Path to save CSV file
    
    Returns:
        DataFrame with generated data
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'city_size': np.random.choice(['Small', 'Medium', 'Large'], n_samples),
        'has_car': np.random.choice([0, 1], n_samples),
        'has_house': np.random.choice([0, 1], n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.normal(100000, 50000, n_samples),
        'interest_rate': np.random.normal(5.5, 2.0, n_samples),
    }
    
    # Create target variable based on features
    data['loan_approved'] = (
        (data['credit_score'] > 650).astype(int) &
        (data['income'] > 40000).astype(int) &
        (data['loan_amount'] < 200000).astype(int)
    )
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Generated tabular data saved to: {save_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

def generate_time_series_data(n_samples=1000, sequence_length=100, n_features=1, trend=True, seasonality=True):
    """
    Generate synthetic time series data.
    
    Args:
        n_samples: Number of time series
        sequence_length: Length of each series
        n_features: Number of features
        trend: Whether to include trend
        seasonality: Whether to include seasonality
    
    Returns:
        X: Time series data, y: Next value predictions
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Base series
        series = np.random.randn(sequence_length, n_features).astype('float32')
        
        # Add trend
        if trend:
            trend_component = np.linspace(0, 2, sequence_length).reshape(-1, 1)
            series += trend_component
        
        # Add seasonality
        if seasonality:
            seasonal_period = 20
            seasonal_component = np.sin(2 * np.pi * np.arange(sequence_length) / seasonal_period).reshape(-1, 1)
            series += seasonal_component * 0.5
        
        X.append(series)
        
        # Target is next value
        next_value = series[-1, 0] + np.random.randn() * 0.1
        y.append(next_value)
    
    X = np.array(X)
    y = np.array(y).astype('float32')
    
    return X, y

def load_and_prepare_mnist_data():
    """
    Load and prepare MNIST dataset.
    
    Returns:
        Preprocessed MNIST data
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return (X_train, y_train), (X_test, y_test)

def load_and_prepare_cifar10_data():
    """
    Load and prepare CIFAR-10 dataset.
    
    Returns:
        Preprocessed CIFAR-10 data
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return (X_train, y_train), (X_test, y_test)

def save_generated_data(X, y, name='dataset', data_dir='./data'):
    """
    Save generated data to files.
    
    Args:
        X: Features
        y: Labels
        name: Dataset name
        data_dir: Directory to save data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(data_dir, f'{name}_X.npy'), X)
    np.save(os.path.join(data_dir, f'{name}_y.npy'), y)
    
    # Save metadata
    metadata = {
        'name': name,
        'X_shape': list(X.shape),
        'y_shape': list(y.shape),
        'X_dtype': str(X.dtype),
        'y_dtype': str(y.dtype),
        'n_samples': X.shape[0]
    }
    
    with open(os.path.join(data_dir, f'{name}_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data saved to {data_dir}/{name}_*.npy")
    print(f"Metadata: {metadata}")

def generate_all_sample_data(data_dir='./data'):
    """
    Generate all sample datasets for the project.
    
    Args:
        data_dir: Directory to save all data
    """
    print("=" * 60)
    print("Generating Sample Data for TensorFlow Deep Learning Project")
    print("Author: RSK World - https://rskworld.in")
    print("=" * 60)
    print()
    
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Classification data
    print("1. Generating classification data...")
    X_cls, y_cls = generate_classification_data(n_samples=1000, n_features=20, n_classes=3)
    save_generated_data(X_cls, y_cls, 'classification', data_dir)
    print()
    
    # 2. Regression data
    print("2. Generating regression data...")
    X_reg, y_reg = generate_regression_data(n_samples=1000, n_features=10)
    save_generated_data(X_reg, y_reg, 'regression', data_dir)
    print()
    
    # 3. Image data
    print("3. Generating image data...")
    X_img, y_img = generate_image_data(n_samples=200, img_size=(28, 28), n_classes=10)
    save_generated_data(X_img, y_img, 'images', data_dir)
    print()
    
    # 4. Sequence data
    print("4. Generating sequence data...")
    X_seq, y_seq = generate_sequence_data(n_samples=500, sequence_length=50, n_features=10, n_classes=3)
    save_generated_data(X_seq, y_seq, 'sequences', data_dir)
    print()
    
    # 5. Time series data
    print("5. Generating time series data...")
    X_ts, y_ts = generate_time_series_data(n_samples=500, sequence_length=100, n_features=1)
    save_generated_data(X_ts, y_ts, 'time_series', data_dir)
    print()
    
    # 6. Tabular data
    print("6. Generating tabular data...")
    df_tab = generate_tabular_data(n_samples=1000, save_path=os.path.join(data_dir, 'tabular.csv'))
    print()
    
    # 7. Load and save MNIST (subset)
    print("7. Loading MNIST dataset...")
    try:
        (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = load_and_prepare_mnist_data()
        # Save subset
        save_generated_data(X_train_mnist[:5000], y_train_mnist[:5000], 'mnist_train_subset', data_dir)
        save_generated_data(X_test_mnist[:1000], y_test_mnist[:1000], 'mnist_test_subset', data_dir)
        print()
    except Exception as e:
        print(f"Could not load MNIST: {e}")
        print()
    
    print("=" * 60)
    print("All sample data generated successfully!")
    print(f"Data saved in: {data_dir}")
    print("=" * 60)

def visualize_generated_data(data_dir='./data'):
    """
    Create visualizations of generated data.
    
    Args:
        data_dir: Directory containing data files
    """
    os.makedirs(os.path.join(data_dir, 'visualizations'), exist_ok=True)
    
    # Visualize image data
    try:
        X_img = np.load(os.path.join(data_dir, 'images_X.npy'))
        y_img = np.load(os.path.join(data_dir, 'images_y.npy'))
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i in range(10):
            row, col = i // 5, i % 5
            axes[row, col].imshow(X_img[i], cmap='gray')
            axes[row, col].set_title(f'Class: {y_img[i]}')
            axes[row, col].axis('off')
        
        plt.suptitle('Generated Image Data Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'visualizations', 'image_samples.png'))
        plt.close()
        print("Image visualization saved")
    except Exception as e:
        print(f"Could not visualize images: {e}")
    
    # Visualize sequence data
    try:
        X_seq = np.load(os.path.join(data_dir, 'sequences_X.npy'))
        y_seq = np.load(os.path.join(data_dir, 'sequences_y.npy'))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for class_id in range(3):
            class_indices = np.where(y_seq == class_id)[0]
            if len(class_indices) > 0:
                sample_idx = class_indices[0]
                axes[class_id].plot(X_seq[sample_idx, :, 0])
                axes[class_id].set_title(f'Sequence Class {class_id}')
                axes[class_id].grid(True)
        
        plt.suptitle('Generated Sequence Data Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'visualizations', 'sequence_samples.png'))
        plt.close()
        print("Sequence visualization saved")
    except Exception as e:
        print(f"Could not visualize sequences: {e}")

if __name__ == '__main__':
    # Generate all sample data
    generate_all_sample_data()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_generated_data()
    print("\nDone!")
