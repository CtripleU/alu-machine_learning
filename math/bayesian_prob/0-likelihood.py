#!/usr/bin/env python3
"""
Defines a function that calculates the likelihood of obtaining the data
given various hypothetical probabilities of developing severe side effects.
"""


import numpy as np


# def likelihood(x, n, P):
#   """Calculates the likelihood of obtaining the data given various hypothetical probabilities of developing severe side effects.

#   Args:
#     x: The number of patients that develop severe side effects.
#     n: The total number of patients observed.
#     P: A 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects.

#   Returns:
#     A 1D numpy.ndarray containing the likelihood of obtaining the data, x and n, for each probability in P, respectively.
#   """

#   # Check if n is a positive integer.
#   if not isinstance(n, int) or n <= 0:
#     raise ValueError("n must be a positive integer.")

#   # Check if x is an integer that is greater than or equal to 0.

  
# if
 
# not
 
# isinstance(x, int) or x < 0:
#     raise ValueError("x must be an integer that is greater than or equal to 0.")

#   # Check if x is not greater than n.

  
# if x > n:
#     raise ValueError("x cannot be greater than n.")

#   # Check if P is a 1D numpy.ndarray.

  
# if
 
# not
 
# isinstance(P, np.ndarray) or P.ndim != 1:
#     raise TypeError("P must be a 1D numpy.ndarray.")

#   # Check if all values in P are in the range [0, 1].

  
# if np.any(P < 0) or np.any(P > 1):
#     raise ValueError("All values in P must be in the range [0, 1]")

#   # Calculate the likelihood for each probability in P.
#   likelihoods = np.zeros_like(P)
#   for i in
 
# range(len(P)):
#     likelihoods[i] = np.binom.pmf(x, n, P[i])

#   return likelihoods


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining the data given various hypothetical probabilities of developing severe side effects.

    Args:
        x: The number of patients that develop severe side effects.
        n: The total number of patients observed.
        P: A 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects.

    Returns:
        A 1D numpy.ndarray containing the likelihood of obtaining the data, x and n, for each probability in P, respectively.
    """

    # Check if n is a positive integer.
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a nonnegative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be a nonnegative integer")

    # Check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")
    
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
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    
    # Check if P has values that start at 0.0 and end at 1.0 and is sorted
    if P[0] != 0.0 or P[-1] != 1.0 or not np.all(np.diff(P) >= 0):
        raise ValueError("P must have values starting at 0.0, ending at 1.0, and be sorted.")
    
    # Calculate the likelihood for each probability in P
    likelihoods = np.array([np.math.comb(n, x) * (p**x) * ((1-p)**(n-x)) for p in P])
    
    return likelihoods

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    print(likelihood(26, 130, P))