#!/usr/bin/env python3
"""
This module contains the LSTMCell class
"""

import numpy as np


class LSTMCell:
    """
    Represents a long short-term memory (LSTM) unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        """
        # Weights
        self.Wf = np.random.normal(size=(i + h, h))  # Forget gate
        self.Wu = np.random.normal(size=(i + h, h))  # Update gate
        self.Wc = np.random.normal(size=(i + h, h))  # Intermediate cell state
        self.Wo = np.random.normal(size=(i + h, h))  # Output gate
        self.Wy = np.random.normal(size=(h, o))  # Output

        # Biases
        self.bf = np.zeros((1, h))  # Forget gate
        self.bu = np.zeros((1, h))  # Update gate
        self.bc = np.zeros((1, h))  # Intermediate cell state
        self.bo = np.zeros((1, h))  # Output gate
        self.by = np.zeros((1, o))  # Output

    def softmax(self, x):
        """
        Softmax activation function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        forget_gate = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        update_gate = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        c_gate = np.tanh(np.dot(concat, self.Wc) + self.bc)

        c_next = forget_gate * c_prev + update_gate * c_gate
        output_gate = self.sigmoid(np.dot(concat, self.Wo) + self.bo)

        h_next = output_gate * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
