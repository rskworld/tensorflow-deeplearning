# Generated Data Directory

**Author**: RSK World  
**Website**: https://rskworld.in  
**Email**: help@rskworld.in  
**Phone**: +91 93305 39277

This directory contains generated sample datasets for the TensorFlow Deep Learning project.

## Generated Datasets

### 1. Classification Data
- **Files**: `classification_X.npy`, `classification_y.npy`
- **Description**: Synthetic classification dataset
- **Shape**: (1000, 20) features, (1000,) labels
- **Classes**: 3
- **Usage**: For testing neural network classification models

### 2. Regression Data
- **Files**: `regression_X.npy`, `regression_y.npy`
- **Description**: Synthetic regression dataset
- **Shape**: (1000, 10) features, (1000,) targets
- **Usage**: For testing regression models

### 3. Image Data
- **Files**: `images_X.npy`, `images_y.npy`
- **Description**: Synthetic image dataset
- **Shape**: (200, 28, 28) images, (200,) labels
- **Classes**: 10
- **Usage**: For testing CNN models

### 4. Sequence Data
- **Files**: `sequences_X.npy`, `sequences_y.npy`
- **Description**: Synthetic sequence data for RNN/LSTM
- **Shape**: (500, 50, 10) sequences, (500,) labels
- **Classes**: 3
- **Usage**: For testing RNN, LSTM, GRU models

### 5. Time Series Data
- **Files**: `time_series_X.npy`, `time_series_y.npy`
- **Description**: Synthetic time series data
- **Shape**: (500, 100, 1) sequences, (500,) targets
- **Usage**: For testing time series prediction models

### 6. Tabular Data
- **File**: `tabular.csv`
- **Description**: Synthetic tabular dataset with multiple features
- **Shape**: (1000, 11) including target
- **Features**: age, income, education, experience, etc.
- **Usage**: For testing tabular data models

### 7. MNIST Subset
- **Files**: `mnist_train_subset_X.npy`, `mnist_train_subset_y.npy`
- **Files**: `mnist_test_subset_X.npy`, `mnist_test_subset_y.npy`
- **Description**: Subset of MNIST dataset (if available)
- **Shape**: (5000, 784) train, (1000, 784) test
- **Usage**: For quick testing without downloading full MNIST

## Loading Data

### Python Example

```python
import numpy as np
import pandas as pd

# Load numpy arrays
X = np.load('data/classification_X.npy')
y = np.load('data/classification_y.npy')

# Load CSV
df = pd.read_csv('data/tabular.csv')

# Load with metadata
import json
with open('data/classification_metadata.json', 'r') as f:
    metadata = json.load(f)
```

### TensorFlow/Keras Example

```python
import numpy as np
from tensorflow import keras

# Load data
X_train = np.load('data/images_X.npy')
y_train = np.load('data/images_y.npy')

# Use in model
model.fit(X_train, y_train, epochs=10)
```

## Regenerating Data

To regenerate all data:

```bash
python scripts/generate_data.py
```

Or use the module directly:

```python
from src.data_generator import generate_all_sample_data
generate_all_sample_data(data_dir='./data')
```

## Data Statistics

Each dataset includes a metadata JSON file with:
- Dataset name
- Shape information
- Data types
- Number of samples

## Visualizations

Check `data/visualizations/` for sample visualizations of the generated data.

## Notes

- All data is generated with a fixed random seed (42) for reproducibility
- Data is normalized and ready to use
- Synthetic data is for testing and demonstration purposes
- For production, use real datasets appropriate to your use case
