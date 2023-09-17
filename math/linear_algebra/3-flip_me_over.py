#!/usr/bin/env python3
""" Calculates the transpose of a matrix """


def matrix_transpose(matrix):
    """ Calculates the transpose of a matrix """
    transpose = []
    for i in range(len(matrix[0])):
        transpose.append([])
    for row in matrix:
        for i in range(len(row)):
            transpose[i].append(row[i])
    return transpose
