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
import abc, numpy
from typing import Tuple

from pycospath.chain import ChainBase

class CosBase(abc.ABC):
    """The base class of chain-of-states methods."""

    def __init__(self,
                 chain:   ChainBase, 
                 verbose: bool = True):
        """Create a chain-of-states calculation. 

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If infomation related to the calculation should be verbosed. 
        """
        self._chain   = chain   # chain instance.
        self._verbose = verbose # if output info during calc.

    def is_verbose(self) -> bool:
        """Get if infomation related to the calculation should be verbosed.
        
        Returns
        -------
        is_verbose: bool
            If infomation related to the calculation should be verbosed. 
        """
        return self._verbose

    def get_chain(self) -> ChainBase:
        """Get the chain instance.

        Returns
        -------
        chain: ChainBase
            The chain instance for the chain-of-states calculation.
        """
        return self._chain
    
    def _get_path_tan(self, 
                      chain_vec: numpy.ndarray
                      ) -> numpy.ndarray:
        """Get the path tangent vector at each replica along the chain. 

        Here: linear tangent.
            r_{i} - r_{i+1} at replica i
            for all replicas along the path (except the last replica).
            The tangent of the last replica is manually set to be the same as the
            second last replica.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector on which the path tangent vectors are computed.
        
        eners_vec: numpy.ndarray of shape (n_reps, )
            The energies vector of the chain vector on which the path tangent 
            vectors are computed. 
            
        Returns
        -------
        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs).
            The path tangent vector at each replica along the chain. 
        """
        path_tan_vec = numpy.zeros(chain_vec.shape)
        path_tan_vec[0:-1, :] = chain_vec[0:-1, :] - chain_vec[1:  , :]
        path_tan_vec[  -1, :] = path_tan_vec[-2, :] # last = sec last.
        return path_tan_vec

    @abc.abstractmethod
    def impose(self, 
               chain_vec: numpy.ndarray
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Impose the chain-of-states (cos) condition on the chain vector and 
        return the chain vector, the energies vector, and the gradients vector. 

        Will also update the chain vector in the chain instance. 

        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the cos condition. 

        Returns
        -------
        (chain_vec, eners_vec, grads_vec): 
            (numpy.ndarray of shape (n_reps, n_dofs), 
             numpy.ndarray of shape (n_reps, ), 
             numpy.ndarray of shape (n_reps, n_dofs))
            chain_vec: The chain vector imposed with the cos condition;
            eners_vec: The energies vector of the chain vector imposed with the 
                       cos condition;
            grads_vec: The gradients vector of the chain vector imposed with the
                       cos condition;
        """
