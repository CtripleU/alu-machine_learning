#!/usr/bin/env python3
"""
This module contains the GRUCell class
"""

import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU)
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        # Weights
        self.Wz = np.random.normal(size=(i + h, h))  # Update gate
        self.Wr = np.random.normal(size=(i + h, h))  # Reset gate
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))  # Output

        # Biases
        self.bz = np.zeros((1, h))  # Update gate
        self.br = np.zeros((1, h))  # Reset gate
        self.bh = np.zeros((1, h))  # Intermediate hidden state
        self.by = np.zeros((1, o))  # Output

    def softmax(self, x):
        """
        Softmax activation function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        update_gate = self.sigmoid(np.dot(concat, self.Wz) + self.bz)
        reset_gate = self.sigmoid(np.dot(concat, self.Wr) + self.br)

        concat_reset = np.concatenate((reset_gate * h_prev, x_t), axis=1)
        h_intermediate = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        h_next = update_gate * h_intermediate + (1 - update_gate) * h_prev
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))
