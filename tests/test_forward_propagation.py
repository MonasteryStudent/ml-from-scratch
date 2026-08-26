import numpy as np

from ml_from_scratch.forward_propagation import (
    dense_iterative,
    dense_vectorized,
)

from ml_from_scratch.logistic_regression import (
    sigmoid,
)

A_batch = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.0, 3.0]
])

W = np.array([
    [0.5, -1.0, 0.3],
    [1.0,  0.5, -0.2]
])

B = np.array([
    [0.1, -0.2, 0.3]
])

Z = np.array([
    [2.6, -0.2, 0.2],
    [2.1, -1.7, 0.7],
    [3.1, 1.3, -0.3]
])


def test_dense_iterative():
    expected = sigmoid(Z[0])
    result = dense_iterative(A_batch[0], W, B[0], sigmoid)

    assert np.allclose(result, expected)


def test_dense_vectorized():
    expected = sigmoid(Z)
    result = dense_vectorized(A_batch, W, B, sigmoid)

    assert np.allclose(result, expected)


def test_dense_vectorized_matches_iterative():
    iterative = dense_iterative(A_batch[0], W, B[0], sigmoid)
    vectorized = dense_vectorized(A_batch[0], W, B[0], sigmoid)

    assert np.allclose(iterative, vectorized)