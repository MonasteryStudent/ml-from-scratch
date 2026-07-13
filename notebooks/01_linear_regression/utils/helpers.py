import numpy as np
from ml_from_scratch.univariate_linear_regression import (
    compute_cost, 
    predict,
    gradient_descent,
)

def generate_study_hour_dataset():
    """
    Generate a synthetic study-hours dataset for the notebooks.

    Returns
    -------
    x : ndarray of shape (20,)
        Feature vector with study hours.
    y : ndarray of shape (20,)
        Target vector containing the exam scores.
    """
    np.random.seed(42)
    x = np.arange(1, 21)
    y = 4 * x + 20 + np.random.normal(0, 10, len(x))
    return x, y

def fit_linear_regression_model(x, y):
    """Fit a univariate linear regression model for notebook demonstrations."""
    w = 0
    b = 0
    alpha = 0.01
    iterations = 5000
    w, b = gradient_descent(x, y, w, b, alpha, iterations)
    return w, b

def simple_cost_curve(x, y):
    """Compute a one-parameter cost curve with b fixed to zero."""
    w_values = np.linspace(0, 10, 200)
    costs = [compute_cost(x, y, w, 0) for w in w_values]
    w_width = 5
    w_min = w_values[np.argmin(costs)]
    w_values = np.linspace(w_min - w_width, w_min + w_width, 200)
    costs = [compute_cost(x, y, w, 0) for w in w_values]
    return w_values, costs, w_min

def tangent_line(x, y, w0):
    """Compute a tangent line of the simplified cost function J(w)."""
    m = len(x)
    w_tangent = np.linspace(w0-1.5, w0+1.5, 50)
    j0 = compute_cost(x, y, w0, 0)
    slope = np.sum((predict(x, w0, 0) - y) * x) / m
    j_tangent = j0 + slope * (w_tangent - w0)
    return w_tangent, j_tangent

def compute_cost_history(x, y, w):
    """Compute the cost history for Gradient Descent with b fixed to zero."""
    alpha=0.0001
    iterations=400
    m = len(x)
    cost_history = []
    for _ in range(iterations):
        error = predict(x, w, 0) - y
        dj_dw = np.sum(error * x) / m
        w_temp = w - alpha * dj_dw
        w = w_temp
        cost_history.append(compute_cost(x, y, w, 0))
    return cost_history

def compute_cost_surface_loop(x, y, w_values, b_values):
    """Compute the cost surface using explicit Python loops."""
    W, B = np.meshgrid(w_values, b_values)
    J = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            J[i, j] = compute_cost(x, y, W[i, j], B[i, j])
    return W, B, J

def compute_cost_surface_vectorized(x, y, w_values, b_values):
    """Compute the cost surface using NumPy vectorization."""
    W, B = np.meshgrid(w_values, b_values)
    predictions = predict(x, W[..., None], B[..., None])
    errors = predictions - y
    J = np.sum(errors**2, axis=2) / (2*len(x))
    return W, B, J

def benchmark_cost_surface_computation(x, y, w_values, b_values):
    """
    Benchmark loop-based and vectorized cost surface computation.

    This helper is intended for experimentation and illustrates the
    performance benefit of NumPy vectorization. The reported times are
    averaged over 20 runs.
    """
    import timeit
    loop_time = timeit.timeit(
    lambda: compute_cost_surface_loop(x, y, w_values, b_values),
    number=20) / 20
    vectorized_time = timeit.timeit(
        lambda: compute_cost_surface_vectorized(x, y, w_values, b_values),
        number=20) / 20
    print(f"Loop:       {loop_time:.6f} s")
    print(f"Vectorized: {vectorized_time:.6f} s")


def compute_gradient_descent_path(x, y, w_start, b_start, alpha=0.01, iterations=5000):
    """Compute the parameter update path of Gradient Descent."""
    w = w_start
    b = b_start
    path_w = [w]
    path_b = [b]
    m = len(x)
    for _ in range(iterations):
        y_hat = predict(x, w, b)
        error = y_hat - y
        dJ_dw = np.sum(error * x) / m
        dJ_db = np.sum(error) / m
        w -= alpha * dJ_dw
        b -= alpha * dJ_db
        path_w.append(w)
        path_b.append(b)
    return path_w, path_b