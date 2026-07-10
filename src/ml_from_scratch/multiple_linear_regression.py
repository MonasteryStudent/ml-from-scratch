import numpy as np

def predict(X, w, b):
    """Compute predictions for multiple linear regression."""
    return X @ w + b

def compute_cost(X, y, w, b):
    """Compute the mean squared error cost."""
    m = X.shape[0]
    error = predict(X, w, b) - y
    return np.sum(error**2) / (2 * m)

def compute_gradient(X, y, w, b):
    """Compute the gradient of the cost function."""
    m = X.shape[0]
    error = predict(X, w, b) - y
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
    """Fit a multiple linear regression model."""
    w = np.zeros(X.shape[1])
    b = 0.0
    return gradient_descent(X, y, w, b, alpha, iterations)