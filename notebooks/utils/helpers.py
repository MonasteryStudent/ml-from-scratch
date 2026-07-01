import numpy as np
from ml_from_scratch.linear_regression import (
    compute_cost, 
    predict,
    gradient_descent,
)

def generate_study_hour_dataset():
    np.random.seed(42)
    X = np.arange(1, 21)
    y = 4 * X + 20 + np.random.normal(0, 10, len(X))
    return X, y

def fit_linear_regression_model(X, y):
    """
    Fit a univariate linear regression model using Gradient Descent.

    This wrapper hides the learning rate, number of iterations, and
    cost history because they are not relevant at this point in the notebook.
    """
    w = 0
    b = 0
    alpha = 0.01
    iterations = 5000

    w, b, _ = gradient_descent(X, y, w, b, alpha, iterations)

    return w, b

def simple_cost_curve(X, y):
    w_values = np.linspace(0, 10, 200)
    costs = [
        compute_cost(X, y, w, 0)
        for w in w_values
    ]
    w_width = 5
    w_min = w_values[np.argmin(costs)]
    w_values = np.linspace(w_min - w_width, w_min + w_width, 200)
    costs = [
        compute_cost(X, y, w, 0)
        for w in w_values
    ]
    return w_values, costs, w_min

def tangent_line(X, y, w0):
    m = X.shape[0]
    w_tangent = np.linspace(w0-1.5, w0+1.5, 50)
    j0 = compute_cost(X, y, w0, 0)
    slope = np.sum((predict(X, w0, 0) - y) * X) / m
    j_tangent = j0 + slope * (w_tangent - w0)
    return w_tangent, j_tangent

def uni_gradient_descent(X, y, w, alpha, iterations):
    m = X.shape[0]
    cost_history = []
    for _ in range(iterations):
        error = predict(X, w, 0) - y
        dj_dw = np.sum(error * X) / m
        w_temp = w - alpha * dj_dw
        w = w_temp
        cost_history.append(compute_cost(X, y, w, 0))
    return w, cost_history

def compute_cost_history(X, y, w):
    _, cost_history = uni_gradient_descent(X, y, w, alpha=0.0001,
    iterations=400)
    return cost_history

def compute_cost_surface(X, y, w_opt, b_opt, w_range, b_range):
    """
    Compute the cost surface J(w, b) for all parameter combinations.
    """

    w_values = np.linspace(w_opt - w_range, w_opt + w_range, 100)
    b_values = np.linspace(b_opt - b_range, b_opt + b_range, 100)
    
    W, B = np.meshgrid(w_values, b_values)

    J = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost(
                X,
                y,
                W[i, j],
                B[i, j],
            )

    return W, B, J


def compute_gradient_descent_path(X, y, w_start, b_start, alpha=0.01, iterations=5000):
    """
    Compute the parameter update path of Gradient Descent.

    Returns
    -------
    path_w : list[float]
        History of the weight parameter.
    path_b : list[float]
        History of the bias parameter.
    """
    w = w_start
    b = b_start

    path_w = [w]
    path_b = [b]

    m = len(X)

    for _ in range(iterations):
        y_hat = predict(X, w, b)
        error = y_hat - y

        dJ_dw = np.sum(error * X) / m
        dJ_db = np.sum(error) / m

        w -= alpha * dJ_dw
        b -= alpha * dJ_db

        path_w.append(w)
        path_b.append(b)

    return path_w, path_b