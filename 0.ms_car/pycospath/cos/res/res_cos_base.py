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
from typing import Tuple

from pycospath.chain import ChainBase
from pycospath.cos   import CosBase

class ResCosBase(CosBase):
    """The base class of restraint-based chain-of-states methods."""

    def __init__(self,
                 chain:   ChainBase,
                 verbose: bool = True):
        """Create a restraint-based chain-of-states calculation.
        
        Parameters
        ----------
        chain: ChainBase
            The chain of replicas.
        
        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        """
        CosBase.__init__(self, chain, verbose)

        # If applying the restraining force on the gradient vector.
        self._apply_res_grads_vec = True

    def _enable_res_grad_vec(self) -> None:
        """Enable the restraint force on the gradient vectors."""
        self._apply_res_grads_vec = True

    def _disable_res_grad_vec(self) -> None:
        """Disable the restraint force on the gradient vectors."""
        self._apply_res_grads_vec = False
    
    @abc.abstractmethod
    def _get_res_grads_vec(self,
                           chain_vec: numpy.ndarray
                           ) -> numpy.ndarray:
        """Determine and return the restraint gradient vector that acts on the  
        chain vector.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the constraint condition.

        Returns
        -------
        restr_grads_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The gradient vector that gives the restraint forces. 
        """

    def impose(self,
               chain_vec: numpy.ndarray
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Impose the chain-of-states (cos) condition on the gradient vector and
        return the chain vector, the energies vector, and the gradients vector. 
        
        Will NOT update the chain vector in the chain instance.

        Here: Impose the restraint force on the gradient vector. 

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
            chain_vec: The chain vector;
            eners_vec: The energies vector;
            grads_vec: The gradients vector imposed with the cos condition;
        """
        chain_vec            = self._chain.get_chain_vec()
        eners_vec, grads_vec = self._chain.get_eners_grads()

        # Impose the gradient restraints.
        if self._apply_res_grads_vec == True:
            restr_grads_vec = self._get_res_grads_vec(chain_vec)
            grads_vec = grads_vec + restr_grads_vec # NOTE: gradients = -force 
        
        return chain_vec, eners_vec, grads_vec
            