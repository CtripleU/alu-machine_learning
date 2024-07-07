#!/usr/bin/env python3

"""
This module determines steady state probabilities
of a markov chain"""

import numpy as np


def regular(P):
    """
    determines steady state probabilities
    of a markov chain

    P - square 2D numpy.ndarray: (n, n) -transition matrix
        - P[i, j] - probability of transitioning from
    state i to state j
        - n no. of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) representing
    steady state probabilities, or None on failure
    """

    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, n = P.shape
    if n != P.shape[0]:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.any(P <= 0):
        return None

    evals, evecs = np.linalg.eig(P.T)
    evecs = evecs.T
    for i in range(len(evals)):
        if np.allclose(evals[i], 1):
            return evecs[i] / np.sum(evecs[i])
    return None
