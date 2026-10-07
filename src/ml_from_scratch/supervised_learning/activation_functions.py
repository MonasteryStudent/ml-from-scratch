import numpy as np


def sigmoid(z):
    """Compute sigmoid values in the range (0, 1) for the input z."""
    return 1 / (1 + np.exp(-z))


def relu(z):
    """Compute ReLU values for the input z."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Compute ReLU derivative values for the input z."""
    return (np.asarray(z) > 0).astype(float)