"""
Transformer Models with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates Transformer architecture implementation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import math

def positional_encoding(position, d_model):
    """
    Create positional encoding for transformer.
    
    Args:
        position: Maximum position
        d_model: Model dimension
    
    Returns:
        Positional encoding matrix
    """
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention layer.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Point-wise feed-forward network.
    
    Args:
        d_model: Model dimension
        dff: Feed-forward dimension
    
    Returns:
        Sequential model
    """
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

class EncoderLayer(layers.Layer):
    """
    Transformer encoder layer.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class DecoderLayer(layers.Layer):
    """
    Transformer decoder layer.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

def create_transformer_encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
    """
    Create a transformer encoder.
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward dimension
        input_vocab_size: Vocabulary size
        maximum_position_encoding: Maximum position encoding
        rate: Dropout rate
    
    Returns:
        Encoder model
    """
    inputs = keras.Input(shape=(None,))
    x = layers.Embedding(input_vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    x += pos_encoding[:, :tf.shape(x)[1], :]
    x = layers.Dropout(rate)(x)
    
    for i in range(num_layers):
        x = EncoderLayer(d_model, num_heads, dff, rate)(x, training=True)
    
    return Model(inputs, x, name='transformer_encoder')

def example_usage():
    """
    Example usage of transformer functions.
    """
    # Create a simple transformer encoder
    encoder = create_transformer_encoder(
        num_layers=2,
        d_model=128,
        num_heads=8,
        dff=512,
        input_vocab_size=10000,
        maximum_position_encoding=1000,
        rate=0.1
    )
    
    print("Transformer Encoder Model:")
    encoder.summary()
    
    # Test with dummy data
    sample_input = tf.random.uniform((32, 100), minval=0, maxval=10000, dtype=tf.int32)
    sample_output = encoder(sample_input, training=False)
    
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
    
    return encoder

if __name__ == '__main__':
    print("Transformer Models with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    encoder = example_usage()
