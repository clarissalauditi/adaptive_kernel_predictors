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


P = 200
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

def solve_dynamics_reg(Kx, y, P, gamma, lamb, sigma=1.0, num_iter=20000, samples=10000, lr=1e-3):
    # Initialize variables
    Kx_root = sqrt_svd(Kx)  # Compute the square root of Kx, assuming sqrt_svd is defined
    h = sigma * random.normal(random.PRNGKey(0), (samples, y.shape[0])) @ Kx_root
    z = sigma * random.normal(random.PRNGKey(1), (samples,))

    # Initial Delta and compute initial phi and Phi0
    #Delta = y
    phi = h * (h > 0.0)
    Phi0 = phi.T @ phi / samples
    f0 = 1.0/gamma * jnp.einsum('ij,i->j', phi, z) / samples
    Delta0 = 1.0 * y - 1/gamma * phi.T @ z / samples + f0

    train_losses = []
    test_losses = []

    for n in range(num_iter):
        # Compute phi and g for current h and z values
        phi = h * (h > 0.0)
        g = (h > 0.0) * z[:, jnp.newaxis]
        Delta = 1.0 * y - 1/gamma * phi.T @ z / samples + f0

        # Full update for h and z based on all training points only
        h += lr * (gamma * jnp.einsum('ij,kj->ik', g[:, :P] * Delta[jnp.newaxis, :P], Kx[:, :P]) - lamb * h)
        z += lr * (gamma * jnp.einsum('ij,j->i', phi[:, :P], Delta[:P]) - lamb * z)

        # Compute and store train and test losses
        train_loss = 0.5 * jnp.mean(Delta[:P] ** 2) # Train loss based on first P points
        test_loss = jnp.mean(Delta[P:] ** 2)   # Test loss based on remaining points
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Iteration {n + 1}/{num_iter}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

    # Final kernel matrix
    Phi = phi.T @ phi / samples

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

    return train_losses, test_losses, Phi, y, Delta[:P], Delta[P:], Phi0, Delta0, h, z, train_alignment, test_alignment

# Define the parameters
gamma_values = [0.5, 1.0]
lamb = 1.0

# Containers to store alignments
train_alignments = []
test_alignments = []


# Iterate over gamma values
for gamma in gamma_values:
    print(f"Running dynamics for gamma = {gamma}")
    
    # Solve dynamics for the current gamma
    train_losses, test_losses, Phi, y, Delta_train, Delta_test, Phi0, Delta0, h, z, train_alignment, test_alignment = solve_dynamics_reg(Kx, y, P, gamma, lamb)


    print(f"Results for gamma = {gamma} saved successfully.")



