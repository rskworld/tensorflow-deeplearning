"""
Data Preprocessing Pipeline for TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides comprehensive data preprocessing utilities.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class ImagePreprocessor:
    """
    Image preprocessing utilities.
    Author: RSK World - https://rskworld.in
    """
    
    @staticmethod
    def load_and_preprocess_image(image_path, target_size=(224, 224)):
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            target_size: Target image size
        
        Returns:
            Preprocessed image tensor
        """
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    @staticmethod
    def create_image_dataset(image_dir, batch_size=32, target_size=(224, 224), validation_split=0.2):
        """
        Create image dataset from directory.
        
        Args:
            image_dir: Directory containing images
            batch_size: Batch size
            target_size: Target image size
            validation_split: Validation split ratio
        
        Returns:
            Training and validation datasets
        """
        train_ds = keras.utils.image_dataset_from_directory(
            image_dir,
            validation_split=validation_split,
            subset='training',
            seed=123,
            image_size=target_size,
            batch_size=batch_size
        )
        
        val_ds = keras.utils.image_dataset_from_directory(
            image_dir,
            validation_split=validation_split,
            subset='validation',
            seed=123,
            image_size=target_size,
            batch_size=batch_size
        )
        
        # Normalize pixel values
        normalization_layer = layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        return train_ds, val_ds
    
    @staticmethod
    def create_augmentation_pipeline():
        """
        Create data augmentation pipeline.
        
        Returns:
            Sequential model with augmentation layers
        """
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])

class TextPreprocessor:
    """
    Text preprocessing utilities.
    Author: RSK World - https://rskworld.in
    """
    
    @staticmethod
    def create_text_vectorization_layer(vocab_size=10000, max_length=100, output_mode='int'):
        """
        Create text vectorization layer.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            output_mode: Output mode ('int', 'binary', 'count', 'tf_idf')
        
        Returns:
            TextVectorization layer
        """
        return layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode=output_mode,
            output_sequence_length=max_length
        )
    
    @staticmethod
    def pad_sequences(sequences, max_length=None, padding='post', truncating='post'):
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences
            max_length: Maximum length
            padding: Padding type ('pre' or 'post')
            truncating: Truncating type ('pre' or 'post')
        
        Returns:
            Padded sequences
        """
        return pad_sequences(sequences, maxlen=max_length, padding=padding, truncating=truncating)
    
    @staticmethod
    def create_tokenizer(texts, num_words=10000):
        """
        Create tokenizer from texts.
        
        Args:
            texts: List of text strings
            num_words: Maximum number of words
        
        Returns:
            Tokenizer object
        """
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        return tokenizer

class TabularPreprocessor:
    """
    Tabular data preprocessing utilities.
    Author: RSK World - https://rskworld.in
    """
    
    @staticmethod
    def normalize_features(X, method='standard'):
        """
        Normalize features.
        
        Args:
            X: Feature matrix
            method: Normalization method ('standard' or 'minmax')
        
        Returns:
            Normalized features and scaler
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_normalized = scaler.fit_transform(X)
        return X_normalized, scaler
    
    @staticmethod
    def encode_categorical_features(df, columns):
        """
        Encode categorical features.
        
        Args:
            df: DataFrame
            columns: List of categorical column names
        
        Returns:
            DataFrame with encoded features and encoders
        """
        encoders = {}
        df_encoded = df.copy()
        
        for col in columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            encoders[col] = le
        
        return df_encoded, encoders
    
    @staticmethod
    def handle_missing_values(df, strategy='mean'):
        """
        Handle missing values.
        
        Args:
            df: DataFrame
            strategy: Strategy ('mean', 'median', 'mode', 'drop')
        
        Returns:
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == 'median':
            df_clean = df_clean.fillna(df_clean.median())
        elif strategy == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        
        return df_clean

class DataPipeline:
    """
    Complete data preprocessing pipeline.
    Author: RSK World - https://rskworld.in
    """
    
    def __init__(self):
        self.preprocessors = {}
    
    def add_preprocessor(self, name, preprocessor):
        """
        Add a preprocessor to the pipeline.
        
        Args:
            name: Preprocessor name
            preprocessor: Preprocessor function
        """
        self.preprocessors[name] = preprocessor
    
    def process(self, data, steps=None):
        """
        Process data through the pipeline.
        
        Args:
            data: Input data
            steps: List of preprocessing steps to apply
        
        Returns:
            Processed data
        """
        if steps is None:
            steps = list(self.preprocessors.keys())
        
        processed_data = data
        for step in steps:
            if step in self.preprocessors:
                processed_data = self.preprocessors[step](processed_data)
        
        return processed_data

def create_tf_dataset(X, y=None, batch_size=32, shuffle=True, buffer_size=1000):
    """
    Create TensorFlow dataset from numpy arrays.
    
    Args:
        X: Features
        y: Labels (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle
        buffer_size: Buffer size for shuffling
    
    Returns:
        TensorFlow dataset
    """
    if y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(X)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def example_usage():
    """
    Example usage of data preprocessing functions.
    """
    # Image preprocessing example
    print("Image Preprocessing Example:")
    image_preprocessor = ImagePreprocessor()
    augmentation = image_preprocessor.create_augmentation_pipeline()
    print("Augmentation pipeline created")
    
    # Text preprocessing example
    print("\nText Preprocessing Example:")
    text_preprocessor = TextPreprocessor()
    vectorization = text_preprocessor.create_text_vectorization_layer(
        vocab_size=10000, max_length=100
    )
    print("Text vectorization layer created")
    
    # Tabular preprocessing example
    print("\nTabular Preprocessing Example:")
    tabular_preprocessor = TabularPreprocessor()
    X = np.random.randn(1000, 10)
    X_normalized, scaler = tabular_preprocessor.normalize_features(X, method='standard')
    print(f"Normalized features shape: {X_normalized.shape}")
    
    # Create TF dataset
    print("\nCreating TensorFlow Dataset:")
    dataset = create_tf_dataset(X_normalized, batch_size=32, shuffle=True)
    print("Dataset created successfully")
    
    return dataset

if __name__ == '__main__':
    print("Data Preprocessing Pipeline for TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    dataset = example_usage()
