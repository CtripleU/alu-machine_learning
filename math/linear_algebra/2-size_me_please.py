#!/usr/bin/env python3
def matrix_shape(mat):
    """ Calculates the shpe of a matrix """
    shape = []
    while isinstance(mat, list):
        matrix_shape.append(len(mat))
        mat = mat[0]
    return matrix_shape
