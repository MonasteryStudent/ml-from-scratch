import numpy as np


def compute_logits(X, w, b):
    """Compute the linear model outputs z before applying the sigmoid function."""
    return X @ w + b


def sigmoid(z):
    """Compute sigmoid values in the range (0, 1) for the linear model outputs z."""
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    """Compute the average logistic regression loss."""
    m = X.shape[0]
    z = compute_logits(X, w, b)
    f_wb = sigmoid(z)
    return -1 / m * np.sum(
        y * np.log(f_wb)
        + (1 - y) * np.log(1 - f_wb)
    )


def compute_gradient(X, y, w, b):
    """Compute the gradient of the cost function."""
    m = X.shape[0]
    z = compute_logits(X, w, b)
    f_wb = sigmoid(z)
    error = f_wb - y
    dj_dw = X.T @ error / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, iterations):
    """Optimize the model parameters using gradient descent."""
    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b


def fit(X, y, alpha, iterations):
    """Fit a logistic regression model."""
    w = np.zeros(X.shape[1])
    b = 0.0
    return gradient_descent(X, y, w, b, alpha, iterations)


def predict_proba(X, w, b):
    """Estimate the probability of the positive class."""
    z = compute_logits(X, w, b)
    return sigmoid(z)


def predict(X, w, b, threshold=0.5):
    """Predict binary class labels using a probability threshold."""
    probabilities = predict_proba(X, w, b)
    predictions = probabilities >= threshold
    return (predictions).astype(int) 