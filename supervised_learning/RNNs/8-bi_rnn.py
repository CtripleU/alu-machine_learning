#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Parameters:
    bi_cell -- an instance of BidirectinalCell that will be used for the forward propagation
    X -- the data to be used, given as a numpy.ndarray of shape (t, m, i)
    h_0 -- the initial hidden state in the forward direction, given as a numpy.ndarray of shape (m, h)
    h_t -- the initial hidden state in the backward direction, given as a numpy.ndarray of shape (m, h)

    Returns:
    H -- a numpy.ndarray containing all of the concatenated hidden states
    Y -- a numpy.ndarray containing all of the outputs
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

