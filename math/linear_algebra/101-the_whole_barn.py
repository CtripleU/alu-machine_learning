#!/usr/bin/env python3

""" Add two matrices """


def matrix_shape(matrix):
    """ Matrix shape """

    matrix_shape = []
    while type(matrix) is list:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape


def add_matrices(mat1, mat2):
    """ Add two matrices """

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if len(matrix_shape(mat1)) is 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
