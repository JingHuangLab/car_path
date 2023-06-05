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
from typing import Tuple, Union

def rms(A:            numpy.ndarray, 
        wscale:       numpy.ndarray, 
        redund:       int, 
        return_grads: bool
        )  -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
    """Computes the mass-weighted root-mean-square (RMS) distances b/w the 
    adjacent rows of the A: 
            ||r_{i} - r_{i+1}||_{2} ;
    and the gradients of RMS distances:
            ||r_{i} - r_{i+1}||_{2} w.r.t. r_{i}, if return_grads is True.
    Note that A and the wscale factor could be degrees-of-freedom(dof)-wise and 
    the resulting RMS should be computed atom-wise, the \'redund\' parameter is 
    use to convert the n_dofs / n_atoms ratio. 

    Parameters
    ----------
    A: numpy.ndarray of shape (n_reps, n_dofs)
        The chain vector whose rows are the replica vectors.

    wscale: numpy.ndarray of shape (ndofs, )
        The weighting factor on each dof. 

    redund: int
        The redundancy of the wscale: accounts for the n_dofs / n_atoms ratio.
    
    return_grads: bool
        If the gradients of RMS distances should be returned. 

    Returns
    -------
    rms: numpy.ndarray of shape (n_reps-1, )
        The RMS distances b/w adjacent rows of A (rep_vecs). 
    
    (rms, rms_grad): (numpy.ndarray of shape (n_reps-1, ), 
                      numpy.ndarray of shape (n_reps-1, n_dofs))
        The RMS distances b/w adjacent rows of A and their gradients w.r.t. 
        r_{i}. Returns only if get_grads = True. 
    """
    A_diff = A[0:-1, :] - A[1:  , :]
    
    wscale_norm = numpy.sum(wscale) / redund # normalization factor for wscale.

    rms = (numpy.sum(A_diff**2 * wscale[None, :], axis=1) / wscale_norm) ** .5

    if return_grads == True:    # Return RMS and its positional gradients. 
        rms_grad = A_diff * wscale[None, :] / wscale_norm / rms[:, None] 
        return rms, rms_grad

    else:                    # Return RMS only.
        return rms
        