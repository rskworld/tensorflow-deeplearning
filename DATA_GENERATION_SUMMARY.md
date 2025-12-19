# Data Generation Summary

**Author**: RSK World  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in  
**Phone**: +91 93305 39277

## Overview

This project includes comprehensive data generation utilities to create synthetic datasets for testing and demonstrating TensorFlow deep learning models.

## Generated Datasets

### 1. Classification Data
- **File**: `data/classification_X.npy`, `data/classification_y.npy`
- **Size**: 1000 samples, 20 features, 3 classes
- **Use Case**: Testing neural network classification models
- **Format**: NumPy arrays

### 2. Regression Data
- **File**: `data/regression_X.npy`, `data/regression_y.npy`
- **Size**: 1000 samples, 10 features
- **Use Case**: Testing regression models
- **Format**: NumPy arrays

### 3. Image Data
- **File**: `data/images_X.npy`, `data/images_y.npy`
- **Size**: 200 images, 28x28 pixels, 10 classes
- **Use Case**: Testing CNN models
- **Format**: NumPy arrays (grayscale images)

### 4. Sequence Data
- **File**: `data/sequences_X.npy`, `data/sequences_y.npy`
- **Size**: 500 sequences, length 50, 10 features, 3 classes
- **Use Case**: Testing RNN, LSTM, GRU models
- **Format**: NumPy arrays (3D: samples × timesteps × features)

### 5. Time Series Data
- **File**: `data/time_series_X.npy`, `data/time_series_y.npy`
- **Size**: 500 time series, length 100, 1 feature
- **Use Case**: Testing time series prediction models
- **Format**: NumPy arrays

### 6. Tabular Data
- **File**: `data/tabular.csv`
- **Size**: 1000 rows, 11 columns (10 features + 1 target)
- **Features**: age, income, education, experience, city_size, has_car, has_house, credit_score, loan_amount, interest_rate
- **Target**: loan_approved (binary)
- **Use Case**: Testing tabular data models
- **Format**: CSV (Pandas DataFrame)

## How to Generate Data

### Method 1: Standalone Script (No TensorFlow Required)
```bash
python scripts/generate_data_standalone.py
```

### Method 2: Full Script (Requires TensorFlow)
```bash
python scripts/generate_data.py
```

### Method 3: Python Module
```python
from src.data_generator import generate_all_sample_data
generate_all_sample_data(data_dir='./data')
```

## How to Use Generated Data

### Loading NumPy Arrays
```python
import numpy as np

# Load data
X = np.load('data/classification_X.npy')
y = np.load('data/classification_y.npy')

print(f"Shape: X={X.shape}, y={y.shape}")
```

### Loading CSV Data
```python
import pandas as pd

# Load tabular data
df = pd.read_csv('data/tabular.csv')
print(df.head())
print(df.describe())
```

### Using with TensorFlow/Keras
```python
import numpy as np
from tensorflow import keras

# Load data
X_train = np.load('data/images_X.npy')
y_train = np.load('data/images_y.npy')

# Reshape if needed (for CNN)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Use in model
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## Example Usage

See `examples/use_generated_data.py` for complete examples showing how to use each dataset type with different model architectures.

## Data Characteristics

### Classification Data
- Balanced classes
- 20 numerical features
- Some noise added for realism
- Suitable for multi-class classification

### Regression Data
- Linear and non-linear relationships
- 10 features
- Gaussian noise added
- Continuous target values

### Image Data
- 28x28 grayscale images
- Pattern-based generation
- 10 different classes
- Normalized to [0, 1]

### Sequence Data
- Time series with trends
- Periodic patterns for some classes
- Variable-length patterns
- Suitable for sequence classification

### Tabular Data
- Mixed data types (numerical, categorical)
- Realistic feature distributions
- Binary classification target
- Ready for preprocessing pipelines

## Metadata

Each dataset includes a JSON metadata file:
- Dataset name
- Shape information
- Data types
- Number of samples

Example:
```json
{
  "name": "classification",
  "X_shape": [1000, 20],
  "y_shape": [1000],
  "X_dtype": "float32",
  "y_dtype": "int64",
  "n_samples": 1000
}
```

## Data Directory Structure

```
data/
├── classification_X.npy
├── classification_y.npy
├── classification_metadata.json
├── regression_X.npy
├── regression_y.npy
├── regression_metadata.json
├── images_X.npy
├── images_y.npy
├── images_metadata.json
├── sequences_X.npy
├── sequences_y.npy
├── sequences_metadata.json
├── time_series_X.npy
├── time_series_y.npy
├── time_series_metadata.json
├── tabular.csv
├── visualizations/
│   ├── image_samples.png
│   └── sequence_samples.png
└── README.md
```

## Notes

- All data is generated with random seed 42 for reproducibility
- Data is normalized and ready to use
- Synthetic data is for testing and demonstration
- For production, use real datasets appropriate to your use case
- Data can be regenerated at any time with the same results

## Regenerating Data

To regenerate all data with the same characteristics:
```bash
python scripts/generate_data_standalone.py
```

To generate with different parameters, modify the script or use the functions directly in Python.
