#!/usr/bin/env python3
"""
contains a function that performs forward propagation for a simple RNN
"""

import numpy as np

def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.by.shape[1]))

    H[0] = h_0

    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
