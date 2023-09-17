#!/usr/bin/env python3

""" Add two matrices """
import numpy as np


def add_matrices(mat1, mat2):
    """ Adds two matrices
    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    matrix1 = np.array(mat1)
    matrix2 = np.array(mat2)
    if matrix1.shape != matrix2.shape:
        return None
    return matrix1 + matrix2
