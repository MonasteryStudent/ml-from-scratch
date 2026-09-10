import numpy as np
import pytest

from ml_from_scratch.activation_functions import (
    relu,
    relu_derivative,
    sigmoid,
)


@pytest.mark.parametrize(
    "z, expected",
    [
        (0.0, 0.5),
        (np.log(3), 0.75),
        (-np.log(3), 0.25),
    ],
)
def test_sigmoid(z, expected):
    assert np.isclose(sigmoid(z), expected)


def test_relu():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([0.0, 0.0, 3.0])

    assert np.array_equal(relu(z), expected)


def test_relu_derivative():
    z = np.array([-2.0, 0.0, 3.0])
    expected = np.array([0.0, 0.0, 1.0])

    assert np.array_equal(relu_derivative(z), expected)


def test_relu_derivative_accepts_a_scalar():
    assert np.isclose(relu_derivative(2.0), 1.0)