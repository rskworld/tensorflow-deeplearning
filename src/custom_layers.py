"""
Custom Layers and Models with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates how to create custom layers and models in TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

class DenseLayer(layers.Layer):
    """
    Custom dense layer with custom initialization.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, units, activation=None, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(DenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super(DenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation)
        })
        return config

class AttentionLayer(layers.Layer):
    """
    Custom attention layer for sequence models.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W1 = self.add_weight(
            name='W1',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W2 = self.add_weight(
            name='W2',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Compute attention scores
        attention_scores = tf.matmul(tf.tanh(tf.matmul(inputs, self.W1)), self.W2)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention weights
        context = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

class ResidualBlock(layers.Layer):
    """
    Custom residual block layer.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, units, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.units, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(self.units)
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        if input_shape[-1] != self.units:
            self.shortcut = layers.Dense(self.units)
        else:
            self.shortcut = lambda x: x
        
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=False):
        # Main path
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        shortcut = self.shortcut(inputs)
        
        # Add and activate
        output = layers.Activation('relu')(x + shortcut)
        return output
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({'units': self.units})
        return config

class CustomCNNModel(Model):
    """
    Custom CNN model using functional API.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, num_classes=10, **kwargs):
        super(CustomCNNModel, self).__init__(**kwargs)
        
        # Convolutional layers
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        
        self.conv2 = layers.Conv2D(64, 3, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        
        self.conv3 = layers.Conv2D(128, 3, activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        
        # Dense layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

class CustomRNNModel(Model):
    """
    Custom RNN model with attention mechanism.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, num_classes=10, **kwargs):
        super(CustomRNNModel, self).__init__(**kwargs)
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units, return_sequences=True)
        self.attention = AttentionLayer(units=128)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.attention(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def create_model_with_custom_layers(input_shape, num_classes):
    """
    Create a model using custom layers.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Use custom dense layer
    x = DenseLayer(128, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    
    # Use residual block
    x = ResidualBlock(64)(x)
    x = ResidualBlock(32)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def example_usage():
    """
    Example usage of custom layers and models.
    """
    # Generate sample data
    X_train = np.random.randn(1000, 784).astype('float32')
    y_train = np.random.randint(0, 10, 1000)
    
    # Create model with custom layers
    model = create_model_with_custom_layers(input_shape=(784,), num_classes=10)
    
    # Display model architecture
    model.summary()
    
    # Train model
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        verbose=1
    )
    
    return model

if __name__ == '__main__':
    print("Custom Layers and Models with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    model = example_usage()
