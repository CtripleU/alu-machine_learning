#!/usr/bin/env python3
"""
This module contains the BidirectionalCell class
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        i -- dimensionality of the data
        h -- dimensionality of the hidden states
        o -- dimensionality of the outputs
        """
        # Weights
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        # Biases
        self.bhf = np.zeros((1, h))  # Forward hidden states
        self.bhb = np.zeros((1, h))  # Backward hidden states
        self.by = np.zeros((1, o))  # Outputs

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for
        one time step

        Parameters:
        h_prev -- numpy.ndarray of shape (m, h) containing
        the previous hidden state
        x_t -- numpy.ndarray of shape (m, i) that contains
        the data input for the cell

        Returns:
        h_next -- the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        for one time step

        Parameters:
        h_next -- numpy.ndarray of shape (m, h) containing
        the next hidden state
        x_t -- numpy.ndarray of shape (m, i) that contains
        the data input for the cell

        Returns:
        h_prev -- the previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev
