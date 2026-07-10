import numpy as np
import pytest
from ml_from_scratch.multiple_linear_regression import (
    predict,
    compute_cost,
    compute_gradient,
    gradient_descent,
    fit
)

def test_predict():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([9, 19])
    w = np.array([2, 3])
    b = 1
    assert np.array_equal(predict(X, w, b), y)

@pytest.mark.parametrize(
        "X, y, w, b, expected_cost",
        [
            pytest.param(
                np.array([[1, 2], [3, 4]]), 
                np.array([9, 19]), 
                np.array([2, 3]), 
                1, 
                0.0, 
                id="zero_cost"
            ),
            pytest.param(
                np.array([[1, 2]]),
                np.array([7]),
                np.array([2, 3]),
                1,
                2.0,
                id="known_cost"
            )
        ]
)
def test_compute_cost(X, y, w, b, expected_cost):
    cost = compute_cost(X, y, w, b)

    assert cost == pytest.approx(expected_cost)

@pytest.mark.parametrize(
        "X, y, w, b, expected_gradient",
        [
            pytest.param(
                np.array([[1, 2], [3, 4]]),
                np.array([9, 19]),
                np.array([2, 3]),
                1,
                [np.array([0.0, 0.0]), 0.0],
                id="zero_gradient"
            ),
            pytest.param(
                np.array([[1, 2]]),
                np.array([7]),
                np.array([2, 3]),
                1,
                [np.array([2.0, 4.0]), 2.0],
                id="known_gradient"
            )
        ]
)
def test_compute_gradient(X, y, w, b, expected_gradient):
    dj_dw, dj_db = compute_gradient(X, y, w, b)

    assert np.allclose(dj_dw, expected_gradient[0])
    assert dj_db == pytest.approx(expected_gradient[1])

def test_gradient_descent_reduces_cost():
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
    ])
    y = np.array([8, 7, 16, 15])

    w = np.zeros(X.shape[1])
    b = 0.0

    initial_cost = compute_cost(X, y, w, b)

    w_final, b_final = gradient_descent(
        X,
        y,
        w,
        b,
        alpha=1e-2,
        iterations=1000,
    )

    final_cost = compute_cost(X, y, w_final, b_final)

    assert final_cost < initial_cost

def test_fit_returns_parameters_with_correct_shape():
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
    ])
    y = np.array([8, 7, 16, 15])

    w, b = fit(X, y, alpha=0.01, iterations=1000)

    assert w.shape == (X.shape[1],)
    assert isinstance(b, float)