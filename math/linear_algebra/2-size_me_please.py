#!/usr/bin/env python3
""" Calculates the shape of a matrix """


def matrix_shape(mat):
    """ Calculates the shape of a matrix """
    shape = []
    while isinstance(mat, list):
        shape.append(len(mat))
        mat = mat[0]
    return shape
