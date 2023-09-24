#!/usr/bin/env python3
""" defines a function that calculates a summation """


def summation_i_squared(n):
    """ summation """
    if type(n) is not int or n < 1:
        return None
    sigma_sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sigma_sum)
