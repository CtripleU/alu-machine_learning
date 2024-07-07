#!/usr/bin/env python3

"""
This module contain a function that initializes all variables
required to calculate the P affinities in t-SNE
"""

import numpy as np


def P_init(X, perplexity):
    """
    X: numpy.ndarray: (n, d) the dataset
        n - no. of data points
        d - no. of dimentions in each point
    perplexity: the perplexity in all Gaussian distributions

    Returns:
    D: numpy.ndarray: (n, n) the pairwise Euclidean distances
        - calculates square pairwise distances btn two data points
        - initialized to all 0s
        - diagonal set to 0s
    P: numpy.ndarray: (n, n) the P affinities
        - initialized to all 0s
    betas: numpy.ndarray: (n, 1) the beta values
        - initialized to all 1s
        - B = 1 / (2 * sigma^2)
    H: float: the Shannon entropy for perplexity
    """
    n, d = X.shape

    # Calculate the pairwise Euclidean distances
    mult = np.matmul(X, -X.T)
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * mult, sum_X), sum_X.T)

    # Initialize the P affinities
    P = np.zeros((n, n))

    # Initialize the beta values
    betas = np.ones((n, 1))

    # Initialize the Shannon entropy
    H = np.log2(perplexity)

    return D, P, betas, H
