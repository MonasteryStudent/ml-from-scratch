import numpy as np

from ml_from_scratch.activation_functions import (
    relu,
    relu_derivative,
    sigmoid,
)


def initialize_parameters(n_features, n_hidden, seed=42):
    """Initialize the parameters of a two-layer neural network."""
    rng = np.random.default_rng(seed)

    parameters = {
        "W1": (
            rng.standard_normal((n_features, n_hidden))
            * np.sqrt(2 / n_features)
        ),
        "b1": np.zeros((1, n_hidden)),
        "W2": (
            rng.standard_normal((n_hidden, 1))
            * np.sqrt(1 / n_hidden)
        ),
        "b2": np.zeros((1, 1)),
    }

    return parameters


def forward_propagation(X, parameters):
    """Compute predictions and cache intermediate values."""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return A2, cache


def compute_cost(y, y_hat):
    """Compute the average binary cross-entropy loss."""
    Y = np.asarray(y).reshape(-1, 1)
    y_hat_safe = np.clip(y_hat, 1e-15, 1 - 1e-15)

    cost = -np.mean(
        Y * np.log(y_hat_safe)
        + (1 - Y) * np.log(1 - y_hat_safe)
    )

    return float(cost)


def backward_propagation(X, y, parameters, cache):
    """Compute the gradients of the cost with respect to all parameters."""
    Y = np.asarray(y).reshape(-1, 1)
    m = X.shape[0]

    W2 = parameters["W2"]

    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }

    return gradients


def update_parameters(parameters, gradients, alpha):
    """Update all parameters using gradient descent."""
    updated_parameters = {
        "W1": parameters["W1"] - alpha * gradients["dW1"],
        "b1": parameters["b1"] - alpha * gradients["db1"],
        "W2": parameters["W2"] - alpha * gradients["dW2"],
        "b2": parameters["b2"] - alpha * gradients["db2"],
    }

    return updated_parameters


def gradient_descent(X, y, parameters, alpha, iterations):
    """Train the network using batch gradient descent."""
    cost_history = []

    for _ in range(iterations):
        y_hat, cache = forward_propagation(X, parameters)
        cost = compute_cost(y, y_hat)
        gradients = backward_propagation(X, y, parameters, cache)
        parameters = update_parameters(parameters, gradients, alpha)
        cost_history.append(cost)

    return parameters, cost_history


def fit(X, y, n_hidden, alpha, iterations, seed=42):
    """Initialize and train a two-layer neural network."""
    parameters = initialize_parameters(
        n_features=X.shape[1],
        n_hidden=n_hidden,
        seed=seed,
    )

    return gradient_descent(
        X,
        y,
        parameters,
        alpha,
        iterations,
    )


def predict_proba(X, parameters):
    """Estimate probabilities for the positive class."""
    y_hat, _ = forward_propagation(X, parameters)
    return y_hat.ravel()


def predict(X, parameters, threshold=0.5):
    """Predict binary class labels."""
    probabilities = predict_proba(X, parameters)
    return (probabilities >= threshold).astype(int)