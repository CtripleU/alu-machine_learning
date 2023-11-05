#!/usr/bin/env python3
"""
Defines a function that calculates the likelihood of obtaining the data
given various hypothetical probabilities of developing severe side effects.
"""


import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining the data given various hypothetical
       probabilitiesnof developing severe side effects.

    Args:
        x: The number of patients that develop severe side effects.
        n: The total number of patients observed.
        P: A 1D numpy.ndarray containing the various hypothetical probabilities 
           of developing severe side effects.

    Returns:
        A 1D numpy.ndarray containing the likelihood of obtaining the data, x and n, 
        for each probability in P, respectively.
    """

    # Check if n is a positive integer.
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # # Check if x is a nonnegative integer
    # if not isinstance(x, int) or x < 0:
    #     raise ValueError("x must be a nonnegative integer")

    # # Check if x is greater than n
    # if x > n:
    #     raise ValueError("x cannot be greater than n")
    
    # Check if x is an integer greater than or equal to 0
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    # Check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy array
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in the range [0, 1]
    for value in P:
        if value > 1 or value < 0:
            raise ValueError("All values in P must be in the range [0, 1]")

    # # Check if P has values that start at 0.0 and end at 1.0 and is sorted
    # if P[0] != 0.0 or P[-1] != 1.0 or not np.all(np.diff(P) >= 0):
    #     raise ValueError("P must have values starting at 0.0, ending at
    #     1.0, and be sorted.")

    # Calculate the likelihood for each probability in P
    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
