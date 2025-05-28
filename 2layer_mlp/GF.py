import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
import numpy as np
import seaborn as sns
import tensorflow_datasets as tfds
import tensorflow as tf

import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.scipy.linalg import solve, inv
from jax import vjp, vmap
import os

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


def sqrt_svd(Sigma, eps=1e-4):
    s, v = jnp.linalg.eigh(Sigma + eps*jnp.eye(Sigma.shape[0]))
    s = s * (s > 0.0)  # Ensure non-negative eigenvalues
    return v @ jnp.diag(jnp.sqrt(s)) @ v.T

P = 1000
Ptest = 1000
lambda0 = 1.0
seedx = 15
C, X_train, y_train, X_test, y_test = load_mnist(P, Ptest, lambda0, seed=seedx)

y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

X = jnp.vstack([X_train, X_test])
D = X.shape[1]
Kx = (1 / (lambda0 * D)) * X @ X.T
Kx_root = sqrt_svd(Kx)
y = jnp.concatenate([y_train, y_test])
lamb = 1.0


# Lazy

def NN_func_lazy(params, X):
    W, z = params
    N = z.shape[0]
    D = X.shape[0]  # X is assumed to have shape (D, num_samples)
    h = W @ X / jnp.sqrt(D)
    phi = h * (h > 0.0)
    f = phi.T @ z / jnp.sqrt(N) 
    return f


def NN_train_lazy(X_train, y_train, X_test, y_test, lamb, N=5000, lr=1e-3, steps=20000):
    X_train = X_train.T
    X_test = X_test.T
    D = X_train.shape[0]  # D is now the feature dimension

    # Initialize parameters
    W = random.normal(random.PRNGKey(0), (N, D))
    z = random.normal(random.PRNGKey(1), (N,))

    # Compute initial phi and Phi0 for both training and test sets
    X = jnp.hstack([X_train, X_test])
    h = W @ X / jnp.sqrt(D)
    phi = h * (h > 0.0)
    Phi0 = phi.T @ phi / N

    # Define loss function and its gradient
    loss = lambda params, X, y: 0.5 * jnp.sum((NN_func_lazy(params, X) - y) ** 2)
    grad_loss = grad(loss)

    params = [W, z]
    train_losses = []
    test_losses = []

    # Training loop
    for n in range(steps):
        # Compute current train and test losses
        train_loss = 0.5 * jnp.mean((NN_func_lazy(params, X_train) - y_train) ** 2)
        test_loss = jnp.mean((NN_func_lazy(params, X_test) - y_test) ** 2)

        print(f"Iteration {n + 1}/{steps}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

        # Append losses to respective lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Compute gradients
        grads = grad_loss(params, X_train, y_train)

        # Update parameters with gradient descent (training data only)
        W += -lr * grads[0] - lr * lamb * W
        z += -lr * grads[1] - lr * lamb * z
        params = [W, z]

    # Compute final phi and Phi for both training and test sets
    h = W @ X / jnp.sqrt(D)
    phi = h * (h > 0.0)
    Phi = phi.T @ phi / N

    return train_losses, test_losses, Phi, Phi0, h, z


# Rich

def NN_func(params, X, gamma):
    W, z = params
    N = z.shape[0]
    D = X.shape[0]  # X is assumed to have shape (D, num_samples)
    h = W @ X / jnp.sqrt(D)
    phi = h * (h > 0.0)
    f = phi.T @ z / N / gamma
    return f

def NN_train(X_train, y_train, X_test, y_test, gamma, lamb, N=5000, lr=1e-3, steps=20000):

    X_train = X_train.T
    X_test = X_test.T
    D = X_train.shape[0]  # D is now the feature dimension

    # Initialize parameters
    W = random.normal(random.PRNGKey(0), (N, D))
    z = random.normal(random.PRNGKey(1), (N,))

    # Compute initial phi and Phi0 for both training and test sets
    X = jnp.hstack([X_train, X_test])
    h = W @ X / jnp.sqrt(D)
    phi = h * (h > 0.0)
    Phi0 = phi.T @ phi / N

    # Define loss function and its gradient
    #loss = lambda params, X, y: 0.5 * jnp.sum((NN_func(params, X, gamma) - y) ** 2)
    loss = lambda params, X, y: 0.5 * jnp.sum((NN_func(params, X, gamma) - y) ** 2)
    grad_loss = grad(loss)

    params = [W, z]
    train_losses = []
    test_losses = []

    # Training loop
    for n in range(steps):
        # Compute current train and test losses
        train_loss = 0.5 * jnp.mean((NN_func(params, X_train, gamma) - y_train) ** 2)
        test_loss = jnp.mean((NN_func(params, X_test, gamma) - y_test) ** 2)

        print(f"Iteration {n + 1}/{steps}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

        # Append losses to respective lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Compute gradients
        grads = grad_loss(params, X_train, y_train)

        # Update parameters with gradient descent (training data only)
        W += -lr * N * gamma**2 * grads[0] - lr * lamb * W
        z += -lr * N * gamma**2 * grads[1] - lr * lamb * z
        params = [W, z]

    # Compute final phi and Phi for both training and test sets
    h = W @ X / jnp.sqrt(D)
    phi = h * (h > 0.0)
    Phi = phi.T @ phi / N

    # kernel alignment train
    y_train = y[:P]
    yyT_train = jnp.outer(y_train,y_train.T)
    yyT_norm = jnp.linalg.norm(yyT_train)
    Phi_train = Phi[:P, :P]
    Phi_train_norm = jnp.linalg.norm(Phi_train)
    numerator_train = jnp.sum(Phi_train * yyT_train)

    train_alignment = numerator_train / (Phi_train_norm * yyT_norm)


    # kernel alignment test 
    y_test = y[P:]
    yyT_test = jnp.outer(y_test,y_test.T)
    yyT_norm = jnp.linalg.norm(yyT_test)
    Phi_test = Phi[P:, P:]
    Phi_test_norm = jnp.linalg.norm(Phi_test)
    numerator_test = jnp.sum(Phi_test* yyT_test)

    test_alignment = numerator_test / (Phi_test_norm * yyT_norm)

    return train_losses, test_losses, Phi, Phi0, h, z, y, train_alignment, test_alignment

# Define ranges for lamb and gamma
lamb_values = np.arange(0.34, 0.51, 0.01)
gamma_values = np.arange(0.05, 5.15, 0.1)
steps = 10000

# How to correctly whiten the data
# train
mu_train = jnp.mean(X_train, axis=0)
X_c = X_train - mu_train
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_train = np.sqrt(D) * np.dot(U, Vt)
#cov_white = (1 / D) * (X_white @ X_white.T)

# test
mu_test = jnp.mean(X_test, axis=0)
X_c = X_test - mu_test
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_test = np.sqrt(D) * np.dot(U, Vt)

# File to save results
alignment_file = f"alignments_spangamma_spanlamb_mnist_P_{P}_whitened.txt"

# Check if the file exists and write header if not
if not os.path.exists(alignment_file):
    with open(alignment_file, 'w') as f:
        f.write("#P lamb gamma train_loss test_loss train_alignment test_alignment\n")

# Loop over lamb and gamma values
for lamb in lamb_values:
    for gamma in gamma_values:
        print(f"Training NN for lamb = {lamb}, gamma = {gamma}")
        
        # Train the neural network
        train_losses_nn, test_losses_nn, Phi_nn, Phi0_nn, h_nn, z_nn, y, train_alignment_nn, test_alignment_nn = NN_train(
            X_train, y_train, X_test, y_test, gamma, lamb, steps=steps
        )
        
        # Get the last values of train and test losses
        train_loss = train_losses_nn[-1]
        test_loss = test_losses_nn[-1]

        # Save results to file
        with open(alignment_file, 'a') as f:
            f.write(f"{P} {lamb:.6f} {gamma:.6f} {train_loss:.6f} {test_loss:.6f} {train_alignment_nn:.6f} {test_alignment_nn:.6f}\n")

        print(f"Results saved for lamb = {lamb}, gamma = {gamma}")
