#!/usr/bin/env python3

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Parameters:
    X (numpy.ndarray): Shape (n, d) where n is the number of data points
                       and d is the number of dimensions in each point.
    ndim (int): The new dimensionality of the transformed X.

    Returns:
    numpy.ndarray: T, shape (n, ndim) containing the transformed
    version of X.
    """
    
    X_centered =  np.mean(X, axis=0, keepdims=True)

    A = X - X_centered

    u, s, v = np.linalg.svd(A)

    W = v.T[:, :ndim]

    T = np.matmul(A, W)

    return T
