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
from typing import Tuple

def rigid_fit(A_raw: numpy.ndarray,
              B_raw: numpy.ndarray
              ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Implements the Kabsch algorithm to perform the rigid best-fit operations 
    b/w the 1D arrays A_raw and B_raw. A_raw and B_raw are firstly reshaped to 
    3D arrays, translation and rotation are sequentially done to B in order to 
    fit it onto B, which minimizes the RMS distances b/w the two arrays. 
    Note that A_raw and B_raw should be equal-sized arrays which represent the 
    rep_vecs. 

    Parameters
    ----------
    A_raw: numpy.ndarray of shape (n_dofs, )
        The 1D numpy.ndarray as the reference structure to fit on.
    
    B_raw: numpy.ndarray of shape (n_dofs, )
        The 1D numpy.ndarray as the structure to be translated ans rotated to 
        fit onto A.

    Returns
    -------
    (B_new, ro, tr): (numpy.ndarray of shape (n_dofs, ),
                      numpy.ndarray of shape (3, 3),
                      numpy.ndarray of shape (3,))
        B_new: The best-fitted B flattend;
        ro:    The rotational matrix;
        tr:    The translational matrix.
    """
    # If A_raw or B_raw is not 1D array.
    if A_raw.ndim != 1 or B_raw.ndim != 1:
        raise ValueError("A_raw and B_raw must be flattened 1D arrays.")
    
    # If shapes of A and B are compatible.
    if A_raw.shape != B_raw.shape:
        raise ValueError("A_raw and B_raw should have identical shapes.")

    # Matrix is reshaped to (x, y, z) coordinates for an atom.
    A = A_raw.reshape(-1, 3) # shape (n_atoms, 3)
    B = B_raw.reshape(-1, 3)

    # Centroids of A, B.
    A_cen = numpy.mean(A, axis=0)

    # Translate to origin.
    A_ori = A - A_cen
    B_ori = B - numpy.mean(B, axis=0)

    # H = B.T @ A, so that B is fitted onto A
    H = B_ori.T @ A_ori

    # U @ diag{Sigma} @ V^{T} = SVD(H) 
    V, S, W = numpy.linalg.svd(H)

    # If determinant of V @ U^{T} is negative, R would produce a mirror fit,
    # in this case, the sign of the last row of R is inverted by multiplying -1.
    V[:, -1] = numpy.sign((numpy.linalg.det(V @ W))) * V[:, -1]
    
    
    # Translational matrix
    ro = V @ W

    # Best-fitted B.
    B_bestfit: numpy.ndarray = B_ori @ ro + A_cen[None, :]

    # Flatten for a new copy that is compatible with the input shape. 
    B_raw_bestfit = B_bestfit.flatten()

    return B_raw_bestfit, ro, A_cen
    