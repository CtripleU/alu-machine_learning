#!/usr/bin/env python3
""" Concatenate two matrices """

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Concatenate two matrices
    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.
    """
    return np.concatenate((mat1, mat2), axis=axis)
