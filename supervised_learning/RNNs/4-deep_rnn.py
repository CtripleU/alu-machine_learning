#!/usr/bin/env python3
"""
This module contains the deep_rnn class
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
    rnn_cells -- list of RNNCell instances of length l that will be used
                 for the forward propagation
    X -- data to be used, given as a numpy.ndarray of shape (t, m, i)
    h_0 -- initial hidden state, given as a numpy.ndarray of shape (l, m, h)

    Returns:
    H -- numpy.ndarray containing all of the hidden states
    Y -- numpy.ndarray containing all of the outputs
    """
    # Get the shape of X and h_0
    t, m, i = X.shape
    l, m, h = h_0.shape

    # Get the output dimensionality from the last RNN cell
    o = rnn_cells[-1].Wy.shape[1]

    # Initialize the hidden states and outputs
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    # Set the initial hidden state
    H[0] = h_0

    # Loop over all time steps
    for step in range(t):
        h_prev = X[step]
        # Loop over all layers
        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(H[step, layer], h_prev)
            H[step + 1, layer] = h_next
            h_prev = h_next
        Y[step] = y

    return H, Y
