# TensorFlow Deep Learning

<!--
Project: TensorFlow Deep Learning
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Category: Deep Learning
Difficulty: Advanced
-->

Deep learning with TensorFlow including neural networks, CNNs, RNNs, and building custom models for various applications.

## Description

This project provides a comprehensive guide to TensorFlow, Google's deep learning framework. It covers neural network construction, convolutional neural networks (CNNs), recurrent neural networks (RNNs), custom layers, model training, and deployment. Perfect for building deep learning applications.

## Features

### Core Deep Learning Models
- **Neural Networks**: Feedforward networks, deep networks with batch normalization
- **CNNs**: Simple CNNs, deep CNNs, ResNet-style architectures
- **RNNs**: Simple RNN, LSTM, GRU, Bidirectional LSTM, Sequence-to-sequence models
- **Transformers**: Multi-head attention, encoder-decoder architectures
- **Transfer Learning**: Pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, etc.)
- **GANs**: Generative Adversarial Networks (DCGAN implementation)
- **Autoencoders**: Simple, Convolutional, and Variational Autoencoders

### Advanced Features
- **Custom Layers**: Custom dense, attention, and residual layers
- **Model Training**: Advanced training techniques, callbacks, data augmentation, mixed precision
- **Model Evaluation**: Comprehensive metrics, confusion matrices, ROC curves
- **Data Preprocessing**: Image, text, and tabular data preprocessing pipelines
- **Visualization**: Training history, model architecture, feature importance, layer activations
- **Model Deployment**: SavedModel, H5, TFLite, TensorFlow.js, REST API
- **Docker Support**: Containerized deployment with Docker and Docker Compose

## Technologies

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook, Python 3.8+
- **Deployment**: Flask, Docker, TensorFlow Serving
- **Utilities**: Pillow, TensorFlow.js

## Project Structure

```
tensorflow-deeplearning/
├── README.md
├── requirements.txt
├── setup.py
├── main.py
├── config.yaml
├── env.example
├── Dockerfile
├── docker-compose.yml
├── notebooks/
│   ├── 01_neural_networks.ipynb
│   ├── 02_cnns.ipynb
│   ├── 03_rnns.ipynb
│   └── 04_custom_models.ipynb
├── src/
│   ├── __init__.py
│   ├── neural_networks.py
│   ├── cnns.py
│   ├── rnns.py
│   ├── transformers.py
│   ├── transfer_learning.py
│   ├── gans.py
│   ├── autoencoders.py
│   ├── custom_layers.py
│   ├── model_training.py
│   ├── model_deployment.py
│   ├── model_evaluation.py
│   ├── data_preprocessing.py
│   ├── visualization.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── api/
│   ├── server.py
│   └── requirements.txt
├── examples/
│   ├── train_custom_model.py
│   └── transfer_learning_example.py
├── tests/
│   ├── test_neural_networks.py
│   └── test_cnns.py
├── models/
│   └── .gitkeep
└── data/
    └── .gitkeep
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For API server:
```bash
pip install -r api/requirements.txt
```

## Usage

### Running Python Scripts

```bash
# Using main entry point
python main.py --module neural_networks
python main.py --module cnns
python main.py --module rnns
python main.py --module transfer_learning
python main.py --module gans
python main.py --module autoencoders

# Direct module execution
python src/neural_networks.py
python src/cnns.py
python src/rnns.py
```

### Running Example Scripts

```bash
python examples/train_custom_model.py
python examples/transfer_learning_example.py
```

### Running Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

### Running Tests

```bash
python -m pytest tests/
# or
python -m unittest discover tests
```

### Running API Server

```bash
# Using Python directly
python api/server.py

# Using Docker
docker-compose up tensorflow-api

# The API will be available at http://localhost:5000
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Run Jupyter notebook in Docker
docker-compose up jupyter
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information

## Configuration

Edit `config.yaml` or create `.env` file from `env.example` to customize settings.

## Contact

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

## License

This project is for educational purposes.
