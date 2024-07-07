#!/usr/bin/env python3

"""
This module performs Principal Component Analysis (PCA) to reduce the dimensionality
of a dataset while maintaining a specified fraction of the original variance.
"""


import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters:
    X (numpy.ndarray): Shape (n, d) where n is the number of data points
                       and d is the number of dimensions in each point.
    var (float): The fraction of the variance that the PCA transformation should maintain.

    Returns:
    numpy.ndarray: The weights matrix, W, that maintains var fraction of X's original variance.
                   W has shape (d, nd) where nd is the new dimensionality of the transformed X.
    """
    # Compute the covariance matrix
    cov_matrix = np.cov(X.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate the cumulative variance ratio
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

    # Find the number of components that maintain the desired variance
    n_components = np.argmax(cumulative_variance_ratio >= var) + 1

    # Return the weight matrix W
    return eigenvectors[:, :n_components]