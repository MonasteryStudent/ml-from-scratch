import numpy as np

def predict(x, w, b):
    """Compute predictions for univariate linear regression."""
    return x * w + b

def compute_cost(x, y, w, b):
    """Compute the mean squared error cost."""
    m = len(x)
    error = predict(x, w, b) - y
    return np.sum(error**2) / (2 * m)

def compute_gradient(x, y, w, b):
    """Compute the gradient of the cost function."""
    m = len(x)
    error = predict(x, w, b) - y
    dj_dw = np.sum(error * x) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, iterations):
    """Optimize w and b using gradient descent."""
    for _ in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

def fit(x, y, alpha, iterations):
    """Fit a univariate linear regression model using gradient descent."""
    w = 0
    b = 0
    return gradient_descent(x, y, w, b, alpha, iterations)

