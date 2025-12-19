"""
Autoencoders with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates various autoencoder architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def build_simple_autoencoder(input_shape=(784,), encoding_dim=32):
    """
    Build a simple autoencoder.
    
    Args:
        input_shape: Shape of input data
        encoding_dim: Dimension of encoding layer
    
    Returns:
        Autoencoder model, encoder model, decoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=input_shape)
    encoded = layers.Dense(128, activation='relu')(encoder_input)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    encoder = Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=(encoding_dim,))
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(decoded)
    
    decoder = Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder
    autoencoder_input = keras.Input(shape=input_shape)
    encoded_output = encoder(autoencoder_input)
    decoded_output = decoder(encoded_output)
    
    autoencoder = Model(autoencoder_input, decoded_output, name='autoencoder')
    
    autoencoder.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return autoencoder, encoder, decoder

def build_convolutional_autoencoder(input_shape=(28, 28, 1), encoding_dim=32):
    """
    Build a convolutional autoencoder.
    
    Args:
        input_shape: Shape of input images
        encoding_dim: Dimension of encoding layer
    
    Returns:
        Autoencoder model, encoder model, decoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    encoder = Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=(encoding_dim,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(decoder_input)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    
    decoder = Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder
    autoencoder_input = keras.Input(shape=input_shape)
    encoded_output = encoder(autoencoder_input)
    decoded_output = decoder(encoded_output)
    
    autoencoder = Model(autoencoder_input, decoded_output, name='autoencoder')
    
    autoencoder.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return autoencoder, encoder, decoder

def build_variational_autoencoder(input_shape=(784,), latent_dim=2):
    """
    Build a Variational Autoencoder (VAE).
    
    Args:
        input_shape: Shape of input data
        latent_dim: Dimension of latent space
    
    Returns:
        VAE model, encoder model, decoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=input_shape)
    x = layers.Dense(512, activation='relu')(encoder_input)
    x = layers.Dense(256, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(decoder_input)
    x = layers.Dense(512, activation='relu')(x)
    decoded = layers.Dense(input_shape[0], activation='sigmoid')(x)
    
    decoder = Model(decoder_input, decoded, name='decoder')
    
    # VAE
    vae_input = keras.Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(vae_input)
    vae_output = decoder(z)
    
    # VAE loss
    reconstruction_loss = keras.losses.binary_crossentropy(vae_input, vae_output)
    reconstruction_loss *= input_shape[0]
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    
    vae = Model(vae_input, vae_output, name='vae')
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae, encoder, decoder

def visualize_reconstructions(autoencoder, test_data, num_samples=10):
    """
    Visualize original and reconstructed images.
    
    Args:
        autoencoder: Trained autoencoder model
        test_data: Test data
        num_samples: Number of samples to visualize
    """
    decoded_imgs = autoencoder.predict(test_data[:num_samples], verbose=0)
    
    n = num_samples
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        if len(test_data[i].shape) == 1:
            img_size = int(np.sqrt(test_data[i].shape[0]))
            plt.imshow(test_data[i].reshape(img_size, img_size), cmap='gray')
        else:
            plt.imshow(test_data[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if len(decoded_imgs[i].shape) == 1:
            img_size = int(np.sqrt(decoded_imgs[i].shape[0]))
            plt.imshow(decoded_imgs[i].reshape(img_size, img_size), cmap='gray')
        else:
            plt.imshow(decoded_imgs[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.show()

def example_usage():
    """
    Example usage of autoencoder functions.
    """
    # Load sample data
    (X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
    
    # Preprocess data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    
    # Build simple autoencoder
    autoencoder, encoder, decoder = build_simple_autoencoder(
        input_shape=(784,), encoding_dim=32
    )
    
    print("Autoencoder Model:")
    autoencoder.summary()
    
    # Train autoencoder
    history = autoencoder.fit(
        X_train, X_train,
        epochs=10,
        batch_size=256,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=1
    )
    
    # Visualize reconstructions
    visualize_reconstructions(autoencoder, X_test, num_samples=10)
    
    return autoencoder, encoder, decoder, history

if __name__ == '__main__':
    print("Autoencoders with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    autoencoder, encoder, decoder, history = example_usage()
