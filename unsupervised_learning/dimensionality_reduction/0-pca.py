#!/usr/bin/env python3

"""
This module performs Principal Component Analysis (PCA) to reduce the dimensionality
of a dataset while maintaining a specified fraction of the original variance.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce its dimensionality while maintaining
    a specified fraction of the original variance.

    Parameters:
    X (numpy.ndarray): The dataset, with shape (n, d) where n is the number
                       of data points and d is the number of dimensions.
    var (float): The fraction of the variance to maintain. Defaults to 0.95.

    Returns:
    numpy.ndarray: The weights matrix, W, that maintains the specified
                   fraction of X's original variance. W has shape (d, nd),
                   where nd is the new dimensionality of the transformed X.
    """
    # Compute the covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Calculate the variance explained by each component
    total_variance = sum(eigenvalues)
    variance_explained = [eigenvalue / total_variance for eigenvalue in eigenvalues]
    cumulative_variance_explained = np.cumsum(variance_explained)

    # Determine the number of components to reach desired variance
    num_components = np.where(cumulative_variance_explained >= var)[0][0] + 1

    # Select the top eigenvectors based on the desired variance
    W = eigenvectors[:, :num_components]

    return W