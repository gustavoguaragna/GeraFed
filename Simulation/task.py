"""GeraFed: um framework para balancear dados heterogêneos em aprendizado federado."""

from collections import OrderedDict

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

import math

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def define_discriminator(img_size, classes, batch_size):
    image_input = Input(shape=(img_size*img_size,))
    label_input = Input(shape=(classes,))
    disc_input = Concatenate()([image_input, label_input])

    x = layers.Dense(batch_size*2*2*2, use_bias=True)(disc_input)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2*2, use_bias=True)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2, use_bias=True)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size, use_bias=True)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    discriminator = Model([image_input, label_input], output)
    return discriminator

def define_generator(noise_dim, classes, batch_size, img_size):
    noise_input = Input(shape=(noise_dim,))
    label_input = Input(shape=(classes,))
    gen_input = Concatenate()([noise_input, label_input])

    x = layers.Dense(batch_size, use_bias=True)(gen_input)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2, use_bias=True)(x)
    x = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2*2, use_bias=True)(x)
    x = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(batch_size*2*2*2, use_bias=True)(x)
    x = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Dense(img_size*img_size, activation='tanh')(x)

    generator = Model([noise_input, label_input], output)
    return generator

def define_gan(generator, discriminator, noise_dim, classes):
    #discriminator.trainable = False  # Freeze discriminator's weights during generator training
    noise_input = Input(shape=(noise_dim,))
    label_input = Input(shape=(classes,))
    gen_output = generator([noise_input, label_input])
    gan_output = discriminator([gen_output, label_input])

    gan = Model([noise_input, label_input], gan_output)
    return gan


def load_data(num_clientes):
    """Download and partitions the MNIST dataset."""

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32')
    x_train = x_train / 255.0
    x_train = (x_train - 0.5)/0.5
    x_train = tf.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [64, 64])
    x_train = tf.transpose(x_train, perm=[0, 3, 1, 2])


    partitions = []
    # We keep all partitions equal-sized in this example
    partition_size_train = math.floor(len(x_train) / num_clientes)
    #partition_size_test = math.floor(len(x_test) / NUM_CLIENTS)
    for cid in range(num_clientes):
        # Split dataset into non-overlapping NUM_CLIENT partitions
        idxtrain_from, idxtrain_to = int(cid) * partition_size_train, (int(cid) + 1) * partition_size_train

        x_train_partition = x_train[idxtrain_from:idxtrain_to]
        y_train_partition = y_train[idxtrain_from:idxtrain_to]

        partitions.append((x_train_partition, y_train_partition))

    print("dataset loaded")

    return partitions
