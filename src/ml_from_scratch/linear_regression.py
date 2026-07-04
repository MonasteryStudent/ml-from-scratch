import numpy as np

def predict(x, w, b):
    """Compute predictions for univariate linear regression."""
    return x * w + b

def compute_cost(x, y, w, b):
    """Compute the mean squared error cost."""
    m = len(x)
    error = predict(x, w, b) - y
    return np.sum(error**2) / (2*m)

def gradient_descent(x, y, w, b, alpha, iterations):
    """Optimize w and b using gradient descent."""
    m = len(x)
    for _ in range(iterations):
        error = predict(x, w, b) - y
        dj_dw = np.sum(error * x) / m
        dj_db = np.sum(error) / m
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

def uni_linear_regression(x, y, alpha, iterations):
    """Fit a univariate linear regression model using gradient descent."""
    w = 0
    b = 0
    return gradient_descent(x, y, w, b, alpha, iterations)