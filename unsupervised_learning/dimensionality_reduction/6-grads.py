#!/usr/bin/env python3

"""
This module contain a function that
calculates gradients of Y
"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Function that calculates gradients of Y

    Y: numpy.ndarray: (n, ndim)
        - containing the low dimensional transformation of X
        n - number of data points
    P: numpy.ndarray: (n, n)
        - containing the P affinities of X
    Returns:
    dY: numpy.ndarray: (n, ndim)
        - containing the gradients of Y
    Q: numpy.ndarray: (n, n)
        - containing the Q affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))
    PQ = P - Q
    for i in range(n):
        dY[i, :] = np.sum(
            np.tile(
                PQ[:, i] * num[:, i],(ndim, 1)).T * (Y[i,] - Y), 0)
    return dY, Q
