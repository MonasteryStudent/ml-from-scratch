import numpy as np


def generate_two_moons_dataset(n_samples=200, noise=0.1, seed=42):
    """Generate a noisy Two Moons dataset for binary classification."""
    rng = np.random.default_rng(seed)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    outer_angles = np.linspace(0, np.pi, n_outer)
    inner_angles = np.linspace(0, np.pi, n_inner)

    outer_moon = np.column_stack((
        np.cos(outer_angles),
        np.sin(outer_angles),
    ))

    inner_moon = np.column_stack((
        1 - np.cos(inner_angles),
        0.5 - np.sin(inner_angles),
    ))

    X = np.vstack((outer_moon, inner_moon))
    X += rng.normal(0, noise, X.shape)

    y = np.concatenate((
        np.zeros(n_outer, dtype=int),
        np.ones(n_inner, dtype=int),
    ))

    indices = rng.permutation(n_samples)

    return X[indices], y[indices]


def train_test_split(X, y, train_ratio=0.8, seed=42):
    """Randomly split a dataset into training and test sets."""
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(X))
    split_index = int(train_ratio * len(X))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test


def standardize_features(X_train, X_test):
    """Standardize features using statistics from the training set."""
    mean = np.mean(X_train, axis=0)
    standard_deviation = np.std(X_train, axis=0)

    X_train_standardized = (
        X_train - mean
    ) / standard_deviation

    X_test_standardized = (
        X_test - mean
    ) / standard_deviation

    return X_train_standardized, X_test_standardized