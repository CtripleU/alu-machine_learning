#!/usr/bin/env python3

"""
This module contain a function that
computes the Q affinities
"""

import numpy as np


def Q_affinities(Y):
    """
    Function that calculates Q affinities

    Y - numpy.ndarray (n, ndim) - low dimensional
    transformation of X
        n - number of data points
        ndim - new dimensionality of transformed X
    returns:
    Q - numpy.ndarray (n, n) the Q affinities
    num - numpy.ndarray (n,n) numerator of Q affinities
    """
    n, ndim = Y.shape

    # Compute the pairwise euclidean distance
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    np.fill_diagonal(num, 0.)
    # Q = np.maximum(num, 1e-12)
    Q = np.maximum(num, 1e-12)

    return Q, num
