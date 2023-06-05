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

# Imports
import numpy, abc
from typing import List, Tuple, Union

from pycospath.comms import RepBase

class ChainBase(abc.ABC):
    """The base class of chain."""

    def __init__(self, 
                 replicas: List[RepBase]):
        """Create a chain object.

        Parameters
        ----------
        replicas: List[RepBase]
            A list of replica objects that makes up the chain.
        """
        self._n_dofs   = replicas[0].get_num_dofs() # n_col in chain vector.
        self._n_reps   = len(replicas)              # n_row in chain vector.
        self._rep_list = replicas                   # The list of replicas.

    def get_rep_list(self) -> List[RepBase]:
        """Get the list of replica objects that makes up the chain.

        NOTE: 
        The returned list shares the same memory allocation with self._rep_list. 

        Returns
        -------
        rep_list: List[RepBase]
            The list of replica objects that makes up the chain.
        """
        return self._rep_list

    def get_num_reps(self) -> int:
        """Get the no. replicas on the chain. 

        Returns
        -------
        n_reps: int
            The number of replica objects on the chain.
        """
        return self._n_reps

    def get_num_dofs(self) -> int:
        """Get the no. degrees-of-freedom in the chain vector (self._chain_vec).

        Returns
        -------
        n_dofs: int
            The number of degrees-of-freedom in the replica vectors.
        """
        return self._n_dofs

    @abc.abstractmethod
    def get_wscale_vec(self) -> numpy.ndarray:
        """Get the copy of the weighting scaling vector (self._wscale_vec.copy()
        ) for the scaling of the rows of the chain vector (self._chain_vec) 
        degrees-of-freedoms.
        
        Returns
        -------
        wscale_vec: numpy.ndarray of shape (n_dofs, )
            The copy of the weighting scaling vector (self._wscale_vec.copy()) 
            for the scaling of the chain vector (self._chain_vec) degrees-of-
            freedoms.
        """

    @abc.abstractmethod
    def get_chain_vec(self) -> numpy.ndarray:
        """Get a copy of the chain vector (self._chain_vec.copy()).

        Returns
        -------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            A copy of the chain vector (self._chain_vec.copy()).
        """
    
    @abc.abstractmethod
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

    @abc.abstractmethod
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
        
    @abc.abstractmethod
    def get_rms(self, 
                chain_vec:    numpy.ndarray,
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
        