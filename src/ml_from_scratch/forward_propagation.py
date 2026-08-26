import numpy as np


def dense_iterative(a_in, W, b, g):
    """ 
    Compute the activation function for a single dense layer using a for-loop.
    
    Args:
        a_in (ndarray (n, )): Input activations for one example.
        W (ndarray (n,j)): Weight matrix with n inputs and j units.
        b (ndarray (j, )): Bias vector with one bias per unit.
        g: Activation function.
    
    Returns:
        a_out (ndarray (j, )): Output activations of the layer.
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:, j]
        z = a_in @ w + b[j]
        a_out[j] = g(z)

    return a_out


def dense_vectorized(A_in, W, B, g):
    """ 
    Compute the activation function for a single dense layer using vectorized
    operations.

    Args:
        A_in (ndarray (m,n)): Input activations for m examples.
        W (ndarray (n,j)): Weight matrix with n inputs and j units.
        B (ndarray (1,j)): Bias matrix with one bias per unit.
        g: Activation function.
    
    Returns:
        A_out (ndarray (m,j)): Output activations of the layer.
    """
    Z = A_in @ W + B # (m,j)
    A_out = g(Z)
    
    return A_out