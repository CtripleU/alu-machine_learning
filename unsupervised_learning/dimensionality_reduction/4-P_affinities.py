#!/usr/bin/env python3

"""
This module contain a function that calculates
the symmetric p affinities of a dataset
"""

import numpy as np


P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Function that calculates the Symmetric P affinities
    of a dataset

    X: numpy.ndarray: (n, d) the dataset
        n - no. of data points
        d - no. of dimensions in each data point
    perplexity: the perplexity of all Gaussian distributions
    tol: max tolerance for shannon entropy calculation
    returns:
    P: numpy.ndarray: (n, n) the symmetric P affinities
    """
    
    n, d = X.shape

    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        beta_min = -np.inf
        beta_max = np.inf
        Di = np.append(D[i, :i], D[i, i+1:])
        Hi, Pi = HP(Di, betas[i])

        diff = Hi - H

        while np.abs(diff) > tol:
            if diff > 0:
                beta_min = betas[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + beta_max) / 2
            else:
                beta_max = betas[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + beta_min) / 2

            # Recalculate the Shannon entropy and P affinities
            Hi, Pi = HP(Di, betas[i])
            diff = Hi - H

        # Update the P affinities
        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]

    # Make sure P is symmetric
    P = (P + P.T) / (2 * n)

    return P
