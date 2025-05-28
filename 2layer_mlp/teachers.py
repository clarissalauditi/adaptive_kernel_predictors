import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.scipy.linalg import solve
from functools import partial
from jax import lax
from jax.numpy.linalg import inv


import tensorflow as tf
import tensorflow_datasets as tfds

from jax.scipy.special import erf
from jax.nn import relu, tanh
from jax.scipy.linalg import cholesky
import numpy as np 


def load_mnist(P, P_test, lambda0, seed=15):
    # Load MNIST dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True,  
        with_info=True,
        shuffle_files=True  
    )

    mean = 0.1307
    std_dev = 0.3081
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
        image = (image - mean) / std_dev  
        image = tf.reshape(image, [-1])  
        return image, label

    ds_train = ds_train.shuffle(buffer_size=10000, seed=seed)
    train_label0 = ds_train.filter(lambda image, label: label == 0).take(P // 2).cache()
    train_label1 = ds_train.filter(lambda image, label: label == 1).take(P // 2).cache()
    train_ds = train_label0.concatenate(train_label1)

    train_ds = train_ds.shuffle(buffer_size=1000, seed=seed)
    train_ds = train_ds.map(preprocess).batch(P)  

    ds_test = ds_test.shuffle(buffer_size=10000, seed=seed)
    test_label0 = ds_test.filter(lambda image, label: label == 0).take(P_test // 2).cache()
    test_label1 = ds_test.filter(lambda image, label: label == 1).take(P_test // 2).cache()

    test_ds = test_label0.concatenate(test_label1)

    test_ds = test_ds.shuffle(buffer_size=1000, seed=seed)
    test_ds = test_ds.map(preprocess).batch(P_test)  

    # Extract training and test data for computation
    X_train, y_train = next(iter(train_ds))
    X_train = jnp.array(X_train.numpy())
    y_train = jnp.array(y_train.numpy())

    X_test, y_test = next(iter(test_ds))
    X_test = jnp.array(X_test.numpy())
    y_test = jnp.array(y_test.numpy())

    
    D = X_train.shape[1]  # Input dimension
    C = (X_train @ X_train.T) / (lambda0 * D)

    return C, X_train, y_train, X_test, y_test

def load_cifar(P, P_test, lambda0, seed=15):
    side_size = int(np.sqrt(784))
    
    def preprocess(image, label):
        image = tf.image.resize(image, [side_size, side_size])
        image = tf.image.rgb_to_grayscale(image)
        # Normalize the image
        image = (tf.cast(image, tf.float32) / 255.0 - 0.4799) / 0.2396
        # Flatten the image
        image = tf.reshape(image, [-1])
        return image, label

    # Load CIFAR-10 data
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10', 
        split=['train', 'test'], 
        as_supervised=True, 
        with_info=True,
        shuffle_files=True
    )

    ds_train = ds_train.shuffle(buffer_size=10000, seed=seed)
    train_label0 = ds_train.filter(lambda image, label: label == 0).take(P // 2).cache()
    train_label1 = ds_train.filter(lambda image, label: label == 1).take(P // 2).cache()
    train_ds = train_label0.concatenate(train_label1)

    train_ds = train_ds.shuffle(buffer_size=1000, seed=seed)
    train_ds = train_ds.map(lambda image, label: preprocess(image, label)).batch(P)

    ds_test = ds_test.shuffle(buffer_size=10000, seed=seed)
    test_label0 = ds_test.filter(lambda image, label: label == 0).take(P_test // 2).cache()
    test_label1 = ds_test.filter(lambda image, label: label == 1).take(P_test // 2).cache()
    test_ds = test_label0.concatenate(test_label1)

    test_ds = test_ds.shuffle(buffer_size=1000, seed=seed)
    test_ds = test_ds.map(lambda image, label: preprocess(image, label)).batch(P_test)


    # Extract training and test data for computation
    X_train, y_train = next(iter(train_ds))
    X_train = jax.device_put(np.array(X_train.numpy()))
    y_train = jax.device_put(np.array(y_train.numpy()))

    X_test, y_test = next(iter(test_ds))
    X_test = jax.device_put(np.array(X_test.numpy()))
    y_test = jax.device_put(np.array(y_test.numpy()))

    
    D = X_train.shape[1]  # Input dimension
    print("X_train shape:", X_train.shape)
    C = (X_train @ X_train.T) / (lambda0 * D)

    return C, X_train, y_train, X_test, y_test




