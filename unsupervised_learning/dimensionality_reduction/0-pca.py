#!/usr/bin/env python3

"""
This module contains a function that performs PCA on a dataset
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Arguments:
    X (numpy.ndarray): Shape (n, d) where n is the number of data points
                       and d is the number of dimensions in each point.
    var (float): The fraction of the variance that the PCA should maintain.

    Returns:
    numpy.ndarray: The weights matrix, W, that maintains var fraction of
    X's original variance.
                   W has shape (d, nd) where nd is the new dimensionality
                   of the transformed X.
    """
    # Perform Singular Value Decomposition
    _, s, v = np.linalg.svd(X)

    # Calculate the cumulative variance ratio
    cumulative_variance_ratio = np.cumsum(s) / np.sum(s)

    # Find the number of components that maintain the desired variance
    nd = np.argwhere(cumulative_variance_ratio >= var)[0, 0]

    # Select the appropriate number of components
    W = v.T[:, :(nd + 1)]

    return W
