import numpy as np

def predict(x, w, b):
    return x * w + b

def compute_cost(x, y, w, b):
    m = len(x)
    error = predict(x, w, b) - y
    return np.sum(error**2) / (2*m)

def gradient_descent(x, y, w, b, alpha, iterations):
    m = len(x)
    cost_history = []
    for _ in range(iterations):
        error = predict(x, w, b) - y
        dj_dw = np.sum(error * x) / m
        dj_db = np.sum(error) / m
        w_temp = w - alpha * dj_dw
        b_temp = b - alpha * dj_db
        w = w_temp
        b = b_temp
        cost_history.append(compute_cost(x, y, w, b))
    return w, b, cost_history

def uni_linear_regression(x, y, alpha, iterations):
    w = 0
    b = 0
    return gradient_descent(x, y, w, b, alpha, iterations)