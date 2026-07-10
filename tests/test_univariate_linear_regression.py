import numpy as np
import pytest
from ml_from_scratch.univariate_linear_regression import (
    predict, 
    compute_cost, 
    compute_gradient,
    gradient_descent, 
    fit,
)

def test_predict():
    x = np.array([1, 2, 3])
    y = np.array([3, 5, 7])
    w = 2
    b = 1
    assert np.allclose(predict(x, w, b), y)

@pytest.mark.parametrize(
        "x, y, w, b, expected_cost",
        [
            pytest.param(
                np.array([1, 2, 3]),
                np.array([3, 5, 7]),
                2,
                1,
                0.0,
                id="zero_cost"
            ),
            pytest.param(
                np.array([1, 2]),
                np.array([-2, -5]),
                -5,
                2,
                2.5,
                id="known_cost"
            )
        ]
        
)
def test_compute_cost(x, y, w, b, expected_cost):
    cost = compute_cost(x, y, w, b)
    assert cost == pytest.approx(expected_cost)

@pytest.mark.parametrize(
      "x, y, w, b, expected_gradient",
      [
        pytest.param(
            np.array([1, 2, 3]),
            np.array([3, 5, 7]),
            2,
            1,
            np.array([0.0, 0.0]),
            id="zero_gradient"
        ),
        pytest.param(
            np.array([1, 2]),
            np.array([-2, -5]),
            -5,
            2,
            np.array([-3.5, -2.0]),
            id="known_gradient"          
        )
      ]  
)
def test_compute_gradient(x, y, w, b, expected_gradient):
    dj_dw, dj_db = compute_gradient(x, y, w, b)
    assert dj_dw == pytest.approx(expected_gradient[0])
    assert dj_db == pytest.approx(expected_gradient[1])

def test_gradient_descent_learns_parameters():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])
    w_expected = 2
    b_expected = 1
    w, b = gradient_descent(x, y, w=0, b=0, alpha=0.01, iterations=5000)
    assert np.isclose(w, w_expected, atol=0.01)
    assert np.isclose(b, b_expected, atol=0.01)

def test_fit_uni_linear_regression_model():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])  # y = 2x + 1

    w, b = fit(x, y, alpha=0.01, iterations=5000)

    assert np.isclose(w, 2, atol=0.01)
    assert np.isclose(b, 1, atol=0.01)