#!/usr/bin/env python3

"""
This module contain a function that calculates
the Shannon entropy and P affinities to a datapoint
"""

import numpy as np


def HP(Di, beta):
    """
    Function that calculates the Shannon entropy and
    P affinities to a datapoint

    Di: numpy.ndarray: (n -1,)
        - contains the Euclidean distances from all data points
        to the ith data point
        n - no. of data points
    beta: numpy.ndarray: (1)
        - contains the beta values for Gaussian distribution
    Returns:
    Hi: float: the Shannon entropy of the points
    Pi: numpy.ndarray: (n -1,) the P affinities of the points
    """

    num = np.exp(-Di * beta)

    num[Di == 0] = 0

    den = np.sum(num)

    Pi = num / den

    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
