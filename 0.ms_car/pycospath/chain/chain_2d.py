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
from typing import List, Tuple, Union

from pycospath.chain        import ChainBase
from pycospath.comms.twodim import Rep2D
from pycospath.utils        import rms

class Chain2D(ChainBase):
    """The chain of 2D replicas."""

    def __init__(self,
                 replicas: List[Rep2D]):
        """Create a chain of 2D replicas.

        Parameter
        ---------
        replicas: List[Rep2D]
            A list of 2D replica objects that makes up the chain.
        """
        ChainBase.__init__(self, replicas)

        # Fill the chain_vec with the rep_vec from each replica.
        chain_vec = numpy.zeros((self._n_reps, self._n_dofs))

        for i_rep in range(self._n_reps):
            if not isinstance(self._rep_list[i_rep], Rep2D):
                raise TypeError("Chain2D must be initialized with List[Rep2D].")

            rep: Rep2D = self._rep_list[i_rep]
            chain_vec[i_rep, :] = rep.get_rep_vec()
        
        self._chain_vec = chain_vec
    
    def get_wscale_vec(self) -> numpy.ndarray:
        """Get the copy of the weighting scaling vector (self._wscale_vec.copy()
        ) for the scaling of the rows of the chain vector (self._chain_vec) 
        degrees-of-freedoms.

        Raises a NotImplementedError if called. 
        """
        raise NotImplementedError("No weight-scaling in Chain2D objects.")

    def get_chain_vec(self) -> numpy.ndarray:
        """Get a copy of the chain vector (self._chain_vec.copy()).

        Returns
        -------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            A copy of the chain vector (self._chain_vec.copy()).
        """
        return self._chain_vec.copy()
    
    def set_chain_vec(self, 
                      chain_vec: numpy.ndarray
                      ) -> None:
        """Set the value of the chain vector (self._chain_vec) with a copy of 
        chain_vec (chain_vec.copy()). 
        
        Also update the replica vector (rep_vec) for all replica objects on this 
        chain. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The vector used to set the value of chain vector (self._chain_vec).
        """
        self._chain_vec = chain_vec.copy()

        # Also update the rep_vec for each Rep2D.
        c_vec_val = self._chain_vec.copy() # such that the mem alloc differs. 

        for r in range(self._n_reps):
            self._rep_list[r].set_rep_vec(c_vec_val[r, :])
            
    def get_eners_grads(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the energies vector and the gradients vector of the chain vector 
        (self._chain_vec).

        Returns
        -------
        (eners_vec, grads_vec): (numpy.ndarray of shape (n_reps, ), 
                                 numpy.ndarray of shape (n_reps, n_dofs))
            eners_vec: The energies  vector of the chain vector;
            grads_vec: The gradients vector of the chain vector. 
        """
        eners_vec = numpy.zeros(shape=(self._n_reps, ))
        grads_vec = numpy.zeros(shape=(self._n_reps, self._n_dofs))

        for r in range(self._n_reps):
            eners_vec[r], grads_vec[r, :] = self._rep_list[r].get_ener_grad()

        return eners_vec, grads_vec
    
    def get_rms(self, 
                chain_vec: numpy.ndarray,
                return_grads: bool = False
                ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """Computes the mass-weighted root-mean-square (RMS) distances b/w the 
        adjacent rows of the chain vector (self._chain_Vec): 
            ||r_{i} - r_{i+1}||_{2} ;
        and the gradients of RMS distances:
            ||r_{i} - r_{i+1}||_{2} w.r.t. r_{i}, if return_grads is true.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector along which the RMS b/w adjacent replica vectors is
            to be computed. 
        
        return_grads: bool
            If the gradients of RMS distances should also be returned. 
            
        Returns
        -------
        rms: numpy.ndarray of shape (n_reps-1, )
            The RMS distances b/w adjacent rows of the chain vector. 
        
        (rms, rms_grad): (numpy.ndarray of shape (n_reps-1, ), 
                          numpy.ndarray of shape (n_reps-1, n_dofs))
            The RMS distances b/w adjacent rows of the chain vector and their 
            gradients w.r.t. r_{i}. Returns only if return_grads = True. 
        """
        wscale = numpy.ones((2, ))
        redund = 2      # 2 dof weights per 2D atom/replica.

        return rms(chain_vec, wscale, redund, return_grads)
