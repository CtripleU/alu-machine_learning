#!/usr/bin/env python3

"""
This module contain a function that initializes all variables
required to calculate the P affinities in t-SNE
"""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables for computing P affinities in t-SNE.

    Parameters:
    - X (numpy.ndarray): Dataset (n, d) with 'n' samples and 'd' features.
    - perplexity (float): Balances local vs global data aspects.

    Returns:
    - D (numpy.ndarray): Squared Euclidean distances (n, n), zeros on diagonal.
    - P (numpy.ndarray): Conditional probabilities (n, n), initially zeros.
    - betas (numpy.ndarray): Precision of Gaussians (n, 1), initially ones.
    - H (float): Shannon entropy from perplexity, for neighbor count.
    """

    n, d = X.shape

    mult = np.matmul(X, -X.T)

    sum_X = np.sum(np.square(X), 1)

    D = np.add(np.add(-2 * mult, sum_X), sum_X.T)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
