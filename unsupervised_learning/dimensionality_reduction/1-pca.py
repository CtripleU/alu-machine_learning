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
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the top ndim eigenvectors
    W = eigenvectors[:, :ndim]

    # Project the data onto the new subspace
    T = np.dot(X_centered, W)

    return T
