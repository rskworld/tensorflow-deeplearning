"""
Model Deployment and Serving with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates model saving, loading, and deployment strategies.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json

def save_model_complete(model, model_dir='./saved_models'):
    """
    Save model in multiple formats for different deployment scenarios.
    
    Args:
        model: Keras model to save
        model_dir: Directory to save models
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Save as SavedModel format (recommended)
    savedmodel_path = os.path.join(model_dir, 'savedmodel')
    model.save(savedmodel_path, save_format='tf')
    print(f"Model saved as SavedModel: {savedmodel_path}")
    
    # 2. Save as H5 format
    h5_path = os.path.join(model_dir, 'model.h5')
    model.save(h5_path, save_format='h5')
    print(f"Model saved as H5: {h5_path}")
    
    # 3. Save only weights
    weights_path = os.path.join(model_dir, 'weights.h5')
    model.save_weights(weights_path)
    print(f"Weights saved: {weights_path}")
    
    # 4. Save model architecture as JSON
    json_path = os.path.join(model_dir, 'model_architecture.json')
    model_json = model.to_json()
    with open(json_path, 'w') as f:
        json.dump(json.loads(model_json), f, indent=2)
    print(f"Model architecture saved: {json_path}")
    
    return savedmodel_path, h5_path, weights_path, json_path

def load_model_from_savedmodel(model_path):
    """
    Load model from SavedModel format.
    
    Args:
        model_path: Path to SavedModel directory
    
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model

def load_model_from_h5(h5_path):
    """
    Load model from H5 format.
    
    Args:
        h5_path: Path to H5 file
    
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(h5_path)
    print(f"Model loaded from: {h5_path}")
    return model

def convert_to_tflite(model, tflite_path='./model.tflite', quantize=False):
    """
    Convert model to TensorFlow Lite format for mobile deployment.
    
    Args:
        model: Keras model to convert
        tflite_path: Path to save TFLite model
        quantize: Whether to apply quantization
    
    Returns:
        Path to TFLite model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved: {tflite_path}")
    return tflite_path

def convert_to_tensorflow_js(model, js_dir='./tfjs_model'):
    """
    Convert model to TensorFlow.js format for web deployment.
    
    Args:
        model: Keras model to convert
        js_dir: Directory to save TensorFlow.js model
    """
    try:
        import tensorflowjs as tfjs
        os.makedirs(js_dir, exist_ok=True)
        tfjs.converters.save_keras_model(model, js_dir)
        print(f"TensorFlow.js model saved: {js_dir}")
    except ImportError:
        print("Warning: tensorflowjs not installed. Install it with: pip install tensorflowjs")
        raise

def create_tf_serving_model(model, serving_dir='./serving_model'):
    """
    Prepare model for TensorFlow Serving.
    
    Args:
        model: Keras model to prepare
        serving_dir: Directory to save serving model
    """
    os.makedirs(serving_dir, exist_ok=True)
    
    # Save model with version number (required by TF Serving)
    version_dir = os.path.join(serving_dir, '1')
    os.makedirs(version_dir, exist_ok=True)
    
    model.save(version_dir, save_format='tf')
    print(f"Model prepared for TF Serving: {serving_dir}")

def create_prediction_function(model):
    """
    Create a prediction function wrapper for easier deployment.
    
    Args:
        model: Trained Keras model
    
    Returns:
        Prediction function
    """
    def predict(input_data):
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data (numpy array or list)
        
        Returns:
            Predictions
        """
        # Preprocess input if needed
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        # Make prediction
        predictions = model.predict(input_data, verbose=0)
        
        return predictions
    
    return predict

def create_rest_api_wrapper(model, model_name='tensorflow_model'):
    """
    Create a REST API wrapper template for model serving.
    
    Args:
        model: Trained Keras model
        model_name: Name of the model
    
    Returns:
        Flask app code template (as string)
    """
    flask_code = f"""
# Flask REST API for {model_name}
# Author: RSK World - https://rskworld.in

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load model
model = keras.models.load_model('./saved_models/savedmodel')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        input_data = np.array(data['input'])
        
        # Make prediction
        predictions = model.predict(input_data, verbose=0)
        
        # Return results
        return jsonify({{
            'success': True,
            'predictions': predictions.tolist()
        }})
    except Exception as e:
        return jsonify({{
            'success': False,
            'error': str(e)
        }}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
    
    return flask_code

def benchmark_model(model, test_data, batch_sizes=[1, 8, 16, 32, 64]):
    """
    Benchmark model inference performance.
    
    Args:
        model: Keras model to benchmark
        test_data: Test data for benchmarking
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for batch_size in batch_sizes:
        # Warm up
        _ = model.predict(test_data[:batch_size], verbose=0)
        
        # Benchmark
        import time
        start_time = time.time()
        _ = model.predict(test_data[:batch_size*10], batch_size=batch_size, verbose=0)
        elapsed_time = time.time() - start_time
        
        results[batch_size] = {
            'time': elapsed_time,
            'samples_per_second': (batch_size * 10) / elapsed_time
        }
    
    return results

def example_usage():
    """
    Example usage of deployment functions.
    """
    # Create a simple model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model (using dummy data)
    X_train = np.random.randn(1000, 784).astype('float32')
    y_train = np.random.randint(0, 10, 1000)
    
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Save model in multiple formats
    savedmodel_path, h5_path, weights_path, json_path = save_model_complete(model)
    
    # Convert to TFLite
    tflite_path = convert_to_tflite(model, quantize=False)
    
    # Create prediction function
    predict_fn = create_prediction_function(model)
    
    # Test prediction
    test_input = np.random.randn(1, 784).astype('float32')
    predictions = predict_fn(test_input)
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Benchmark model
    test_data = np.random.randn(100, 784).astype('float32')
    benchmark_results = benchmark_model(model, test_data)
    print("\nBenchmark Results:")
    for batch_size, result in benchmark_results.items():
        print(f"Batch size {batch_size}: {result['samples_per_second']:.2f} samples/sec")
    
    return model

if __name__ == '__main__':
    print("Model Deployment and Serving with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model = example_usage()
