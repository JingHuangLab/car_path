################################################################################
#      pyCosPath: A Python Library for Locating Optimal Reaction Pathways      #
#                 in Simulations with Chain-of-States Methods                  #
#                                                                              #
#                     Copyright (c) 2022 the Authors                           #
#                                                                              #
################################################################################
#                                                                              #
# Authors: Zilin Song                                                          #
# Contributors:                                                                #
#                                                                              #
################################################################################

import numpy 

def row_wise_proj(A: numpy.ndarray, 
                  B: numpy.ndarray
                  ) -> numpy.ndarray:
    """Perform vector projection of each row of the array B onto the correspond-
    ing row of the array A. A and B must both be 2D arrays of the same shapes. 

    Parameters
    ----------
    A: numpy.ndarray
        The input array as the reference.

    B: numpy.ndarray
        The input array to project onto the reference.

    Returns
    -------
    proj_a_B: numpy.ndarray
        The projected array.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D array.")

    if A.shape != B.shape:
        raise ValueError("A and B should have identical shapes.")

    proj_b_A = (numpy.sum(A*B, axis=1) / numpy.sum(A**2, axis=1))[:, None] * A
    return proj_b_A

def row_wise_cos(A: numpy.ndarray, 
                    B: numpy.ndarray
                    ) -> numpy.ndarray:
    """Computes the cosine value b/w each row of B and the corresponding row of 
    A. A and B must both be 2D arrays of the same shapes. 

    Parameters
    ----------
    A: numpy.ndarray
        The input array. 

    B: numpy.ndarray
        The other input array.

    Returns
    -------
    cosine_b_A: numpy.ndarray
        The array that carries the cosine values.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D array.")

    if A.shape != B.shape:
        raise ValueError("A and B should have identical shapes.")

    cosine_b_A =   numpy.sum(A*B, axis=1)                                      \
                 / numpy.linalg.norm(A, axis=1)                                \
                 / numpy.linalg.norm(B, axis=1)
    return cosine_b_A
