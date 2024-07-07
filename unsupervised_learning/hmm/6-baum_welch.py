#!/usr/bin/env python3

"""
This module performs the Baum-Welch algorithm
for a hidden markov model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N, M = Emission.shape

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = np.sum(alpha[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]])

    return alpha

def backward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N, M = Emission.shape

    beta = np.zeros((N, T))
    beta[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[i, t] = np.sum(Transition[i, :] * Emission[:,
                                                            Observation[t + 1]] * beta[:, t + 1])

    return beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm
    for a hidden markov model

    Observation - numpy.ndarray (T,) that contains
    index of the observation
        - T - number of observations
    Emission - numpy.ndarray (M, M) containing the
    initialized emission probabilities
    Transition - 2D numpy.ndarray (M, M) containing the
    initialized transition probabilities
    Initial - numpy.ndarray (M, 1) containing
    starting probabilities
    Return:
    Transition, Emission, or None, None on failure
        - Transition is the updated transition probabilities
        - Emission is the updated emission probabilities
    """
    if (not isinstance(Observations, np.ndarray) or len(Observations.shape) != 1):
        return None, None
    T = Observations.shape[0]
    if (not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2):
        return None, None
    M, N = Emission.shape
    if (not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2):
        return None, None
    M1, M2 = Transition.shape
    if M1 != M or M2 != M:
        return None, None
    if (not isinstance(Initial, np.ndarray) or Initial.shape != (M, 1)):
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    for _ in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            denom = np.sum(alpha[:, t] * (Transition @ (Emission[:,
                                                                 Observations[t + 1]] * beta[:, t + 1])))

            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = (alpha[i, t] * Transition[i, j] * Emission[j, Observations[t + 1]] * beta[j, t + 1]) / denom

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, axis=2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denom = np.sum(gamma, axis=1)
        for k in range(N):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission = np.divide(Emission, denom.reshape((-1, 1)))

    return Transition, Emission
