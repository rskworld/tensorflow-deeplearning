"""
Flask API Server for TensorFlow Deep Learning Models
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides a REST API for serving TensorFlow models.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model(model_path):
    """
    Load TensorFlow model.
    
    Args:
        model_path: Path to model file
    """
    global model
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess image for model input.
    
    Args:
        image_bytes: Image bytes
        target_size: Target image size
    
    Returns:
        Preprocessed image array
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    Expects JSON with 'image' field containing base64 encoded image or file upload.
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
        elif 'image' in request.json:
            import base64
            image_bytes = base64.b64decode(request.json['image'])
        else:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_predictions = [
            {
                'class': int(idx),
                'probability': float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        return jsonify({
            'success': True,
            'predictions': top_predictions,
            'predicted_class': int(np.argmax(predictions[0])),
            'confidence': float(np.max(predictions[0]))
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    Expects JSON with 'images' array containing base64 encoded images.
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        data = request.json
        if 'images' not in data:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        import base64
        processed_images = []
        for image_b64 in data['images']:
            image_bytes = base64.b64decode(image_b64)
            processed_image = preprocess_image(image_bytes)
            processed_images.append(processed_image[0])
        
        batch = np.array(processed_images)
        predictions = model.predict(batch, verbose=0)
        
        results = []
        for pred in predictions:
            top_idx = int(np.argmax(pred))
            results.append({
                'predicted_class': top_idx,
                'confidence': float(pred[top_idx])
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        info = {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': int(model.count_params()),
            'num_layers': len(model.layers)
        }
        
        return jsonify({
            'success': True,
            'model_info': info
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load model from environment variable or default path
    model_path = os.getenv('MODEL_PATH', './models/model.h5')
    
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        logger.warning(f"Model file not found at {model_path}. API will run without model.")
    
    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
