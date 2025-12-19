"""
Generative Adversarial Networks (GANs) with TensorFlow
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module demonstrates GAN implementations including DCGAN.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def build_generator(latent_dim=100):
    """
    Build a generator model for GAN.
    
    Args:
        latent_dim: Dimension of latent space
    
    Returns:
        Generator model
    """
    model = keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Reshape((7, 7, 256)),
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    
    return model

def build_discriminator(input_shape=(28, 28, 1)):
    """
    Build a discriminator model for GAN.
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Discriminator model
    """
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def build_dcgan(generator, discriminator):
    """
    Build a DCGAN model combining generator and discriminator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
    
    Returns:
        Combined GAN model
    """
    # Freeze discriminator during generator training
    discriminator.trainable = False
    
    # Create GAN
    gan_input = keras.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    
    gan = Model(gan_input, gan_output)
    gan.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    
    return gan

def train_gan(generator, discriminator, gan, dataset, epochs=50, batch_size=128, latent_dim=100):
    """
    Train a GAN model.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        gan: Combined GAN model
        dataset: Training dataset
        epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Dimension of latent space
    
    Returns:
        Training history
    """
    real_label = 1.0
    fake_label = 0.0
    
    history = {'d_loss': [], 'g_loss': []}
    
    for epoch in range(epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        
        for batch in dataset:
            batch_size_actual = batch.shape[0]
            
            # Train discriminator
            noise = tf.random.normal([batch_size_actual, latent_dim])
            generated_images = generator(noise, training=False)
            
            # Combine real and fake images
            real_images = batch
            combined_images = tf.concat([real_images, generated_images], axis=0)
            
            # Create labels
            labels = tf.concat([
                tf.ones((batch_size_actual, 1)) * real_label,
                tf.ones((batch_size_actual, 1)) * fake_label
            ], axis=0)
            
            # Add noise to labels (label smoothing)
            labels += 0.05 * tf.random.uniform(labels.shape)
            
            # Train discriminator
            d_loss = discriminator.train_on_batch(combined_images, labels)
            epoch_d_loss.append(d_loss)
            
            # Train generator
            noise = tf.random.normal([batch_size_actual, latent_dim])
            misleading_labels = tf.ones((batch_size_actual, 1)) * real_label
            
            g_loss = gan.train_on_batch(noise, misleading_labels)
            epoch_g_loss.append(g_loss)
        
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
            
            # Generate sample images
            generate_and_save_images(generator, epoch + 1, latent_dim)
    
    return history

def generate_and_save_images(generator, epoch, latent_dim, num_images=16):
    """
    Generate and save sample images from generator.
    
    Args:
        generator: Generator model
        epoch: Current epoch number
        latent_dim: Dimension of latent space
        num_images: Number of images to generate
    """
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = generator(noise, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'generated_images_epoch_{epoch}.png')
    plt.close()

def example_usage():
    """
    Example usage of GAN functions.
    """
    # Build models
    generator = build_generator(latent_dim=100)
    discriminator = build_discriminator(input_shape=(28, 28, 1))
    
    # Compile discriminator
    discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Build GAN
    gan = build_dcgan(generator, discriminator)
    
    print("Generator Model:")
    generator.summary()
    
    print("\nDiscriminator Model:")
    discriminator.summary()
    
    # Generate dummy dataset for demonstration
    dataset = tf.random.normal([1000, 28, 28, 1])
    dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(128)
    
    # Train GAN (short training for demo)
    print("\nTraining GAN...")
    history = train_gan(
        generator, discriminator, gan,
        dataset, epochs=5, batch_size=128, latent_dim=100
    )
    
    return generator, discriminator, gan, history

if __name__ == '__main__':
    print("Generative Adversarial Networks with TensorFlow")
    print("Author: RSK World - https://rskworld.in")
    generator, discriminator, gan, history = example_usage()
