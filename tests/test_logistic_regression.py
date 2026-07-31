import pytest
import numpy as np

from ml_from_scratch.logistic_regression import (
    compute_logits,
    sigmoid,
    compute_cost,
    compute_gradient,
    gradient_descent,
    fit,
    predict_proba,
    predict
)

X = np.array([
    [1.0, 2.0],
    [3.0, 1.0],
])

y = np.array([0, 1])

w = np.array([0.5, -0.2])

b = 0.1


def test_compute_logits():
    expected_logits = np.array([0.2, 1.4])
    assert np.allclose(compute_logits(X, w, b), expected_logits)


@pytest.mark.parametrize(
        "z, expected_proba",
        [
            (0.0, 0.5),
            (np.log(3), 0.75),
            (-np.log(3), 0.25)
        ]
)
def test_sigmoid(z, expected_proba):
    assert np.allclose(sigmoid(z), expected_proba)


def test_compute_cost():
    expected_cost = 0.5092781397
    assert np.isclose(compute_cost(X, y, w, b), expected_cost)


def test_compute_gradient():
    expected_dj_dw = np.array([-0.02180716845, 0.4509259416])
    expected_dj_db = 0.176008943
    result_dj_dw, result_dj_db = compute_gradient(X, y, w, b)
    assert np.allclose(result_dj_dw, expected_dj_dw)
    assert np.isclose(result_dj_db, expected_dj_db)


def test_gradient_descent():
    alpha = 0.1
    iterations = 1
    dj_dw, dj_db = compute_gradient(X, y, w, b)

    expected_w = w - alpha * dj_dw
    expected_b = b - alpha * dj_db

    result_w, result_b = gradient_descent(X, y, w, b, alpha, iterations)

    assert np.allclose(result_w, expected_w)
    assert np.isclose(result_b, expected_b)


def test_gradient_descent_reduces_cost():
    alpha = 0.1
    iterations = 10
    initial_cost = compute_cost(X, y, w, b)

    result_w, result_b = gradient_descent(X, y, w, b, alpha, iterations)

    final_cost = compute_cost(X, y, result_w, result_b)

    assert final_cost < initial_cost


def test_fit():
    alpha = 0.1
    iterations = 10
    w_init = np.zeros(X.shape[1])
    b_init = 0.0

    expected_w, expected_b = gradient_descent(X, y, w_init, b_init, alpha, iterations)

    result_w, result_b = fit(X, y, alpha, iterations)

    assert np.allclose(result_w, expected_w)
    assert np.isclose(result_b, expected_b)


def test_predict_proba():
    expected_probas = np.array([0.5498339973, 0.8021838886])
    result_probas = predict_proba(X, w, b)

    assert np.allclose(result_probas, expected_probas)


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (0.5, np.array([1, 1])),
        (0.7, np.array([0, 1])),
        (0.9, np.array([0, 0])),
    ],
)
def test_predict(threshold, expected):
    result = predict(X, w, b, threshold)
    assert np.array_equal(result, expected)