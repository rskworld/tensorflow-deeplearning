"""
Standalone Data Generation Script (No TensorFlow Required)
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script generates data without requiring TensorFlow installation.
"""

import numpy as np
import pandas as pd
import os
import json

def generate_classification_data(n_samples=1000, n_features=20, n_classes=3, noise=0.1):
    """Generate synthetic classification dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        feature_sum = np.sum(X[i, :n_features//2])
        if feature_sum < -2:
            y[i] = 0
        elif feature_sum < 2:
            y[i] = 1
        else:
            y[i] = 2
    
    noise_mask = np.random.random(n_samples) < noise
    y[noise_mask] = np.random.randint(0, n_classes, np.sum(noise_mask))
    
    return X.astype('float32'), y

def generate_regression_data(n_samples=1000, n_features=10, noise=0.1):
    """Generate synthetic regression dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    coefficients = np.random.randn(n_features)
    y = np.dot(X, coefficients)
    y += 0.1 * np.sum(X ** 2, axis=1)
    y += noise * np.random.randn(n_samples)
    return X.astype('float32'), y.astype('float32')

def generate_image_data(n_samples=100, img_size=(28, 28), n_classes=10):
    """Generate synthetic image dataset."""
    np.random.seed(42)
    X = []
    y = []
    
    for i in range(n_samples):
        label = i % n_classes
        y.append(label)
        img = np.zeros(img_size, dtype='float32')
        
        if label % 2 == 0:
            for j in range(0, img_size[0], 3):
                img[j, :] = 0.8
        else:
            for j in range(0, img_size[1], 3):
                img[:, j] = 0.8
        
        center = (img_size[0] // 2, img_size[1] // 2)
        radius = 5 + label
        y_coords, x_coords = np.ogrid[:img_size[0], :img_size[1]]
        mask = (x_coords - center[1])**2 + (y_coords - center[0])**2 <= radius**2
        img[mask] = 1.0 - img[mask]
        img += np.random.randn(*img_size) * 0.1
        img = np.clip(img, 0, 1)
        X.append(img)
    
    return np.array(X), np.array(y)

def generate_sequence_data(n_samples=1000, sequence_length=50, n_features=10, n_classes=3):
    """Generate synthetic sequence data."""
    np.random.seed(42)
    X = []
    y = []
    
    for i in range(n_samples):
        label = i % n_classes
        y.append(label)
        sequence = np.random.randn(sequence_length, n_features)
        trend = np.linspace(0, label * 0.5, sequence_length)
        sequence[:, 0] += trend
        
        if label == 1:
            periodic = np.sin(np.linspace(0, 4 * np.pi, sequence_length))
            sequence[:, 1] += periodic * 0.5
        
        X.append(sequence)
    
    return np.array(X).astype('float32'), np.array(y)

def generate_tabular_data(n_samples=1000, save_path='./data/synthetic_tabular.csv'):
    """Generate synthetic tabular dataset."""
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
    
    data['loan_approved'] = (
        (data['credit_score'] > 650).astype(int) &
        (data['income'] > 40000).astype(int) &
        (data['loan_amount'] < 200000).astype(int)
    )
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

def save_data(X, y, name='dataset', data_dir='./data'):
    """Save generated data to files."""
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, f'{name}_X.npy'), X)
    np.save(os.path.join(data_dir, f'{name}_y.npy'), y)
    
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
    
    print(f"Saved {name}: X{X.shape}, y{y.shape}")

def main():
    """Generate all sample data."""
    print("=" * 60)
    print("Generating Sample Data")
    print("Author: RSK World - https://rskworld.in")
    print("=" * 60)
    print()
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate all datasets
    print("1. Classification data...")
    X, y = generate_classification_data(1000, 20, 3)
    save_data(X, y, 'classification', data_dir)
    
    print("2. Regression data...")
    X, y = generate_regression_data(1000, 10)
    save_data(X, y, 'regression', data_dir)
    
    print("3. Image data...")
    X, y = generate_image_data(200, (28, 28), 10)
    save_data(X, y, 'images', data_dir)
    
    print("4. Sequence data...")
    X, y = generate_sequence_data(500, 50, 10, 3)
    save_data(X, y, 'sequences', data_dir)
    
    print("5. Tabular data...")
    df = generate_tabular_data(1000, os.path.join(data_dir, 'tabular.csv'))
    print(f"Saved tabular: {df.shape}")
    
    print("\n" + "=" * 60)
    print("All data generated successfully!")
    print(f"Data saved in: {data_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
