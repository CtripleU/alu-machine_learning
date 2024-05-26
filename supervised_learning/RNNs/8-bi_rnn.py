#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    Hf[0] = h_0
    Hb[-1] = h_t

    for step in range(1, t):
        Hf[step] = bi_cell.forward(Hf[step - 1], X[step])
        Hb[t - step - 1] = bi_cell.backward(Hb[t - step], X[t - step - 1])

    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
