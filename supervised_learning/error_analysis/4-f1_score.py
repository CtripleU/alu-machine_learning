#!/usr/bin/env python3
""" F1 score"""

import numpy as np


def f1_score(confusion):
    """ calculates the F1 score of each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
    Returns:
        (classes,): F1 score of each class
    """
    precision = np.diag(confusion) / np.sum(confusion, axis=0)
    recall = np.diag(confusion) / np.sum(confusion, axis=1)
    return 2 * (precision * recall) / (precision + recall)
