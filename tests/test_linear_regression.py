import numpy as np
from ml_from_scratch.linear_regression import (
    predict, 
    compute_cost, 
    gradient_descent, 
    uni_linear_regression,
)

def test_prediction():
    x = np.array([1, 2, 3])
    y = np.array([3, 5, 7])
    w = 2
    b = 1
    assert np.allclose(predict(x, w, b), y)

def test_compute_cost_returns_zero_for_perfect_predictions():
    x = np.array([1, 2, 3])
    y = np.array([3, 5, 7])
    w = 2
    b = 1
    cost = compute_cost(x, y, w, b)
    assert cost == 0

def test_gradient_descent_learns_parameters():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])
    w_expected = 2
    b_expected = 1
    w, b = gradient_descent(x, y, w=0, b=0, alpha=0.01, iterations=5000)
    assert np.isclose(w, w_expected, atol=0.01)
    assert np.isclose(b, b_expected, atol=0.01)

def test_uni_linear_regression_learns_parameters():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])  # y = 2x + 1

    w, b = uni_linear_regression(x, y, alpha=0.01, iterations=5000)

    assert np.isclose(w, 2, atol=0.01)
    assert np.isclose(b, 1, atol=0.01)
