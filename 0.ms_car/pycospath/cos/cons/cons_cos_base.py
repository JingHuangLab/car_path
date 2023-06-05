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

class ConsCosBase(CosBase):
    """The base class of constraint-based chain-of-States methods."""

    def __init__(self,
                 chain:   ChainBase, 
                 verbose: bool = True):
        """Create a constraint-based chain-of-states calculation. 

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        """
        CosBase.__init__(self, chain, verbose)

        # If applying the constraint condition on the chain vector. 
        self._apply_cons_chain_vec = True 

    def _enable_cons_chain_vec(self) -> None:
        """Enable the constraint condition on the chain vectors."""
        self._apply_cons_chain_vec = True

    def _disable_cons_chain_vec(self) -> None:
        """Disable the constraint condition on the chain vectors."""
        self._apply_cons_chain_vec = False

    @abc.abstractmethod
    def _get_cons_chain_vec(self, 
                            chain_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Determine and return the chain vector that satisfies the constraint
        condition. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the constraint condition.

        Returns
        -------
        constr_chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector that satisfies the constraint condition.
        """

    def impose(self, 
               chain_vec: numpy.ndarray
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Impose the chain-of-states (cos) condition on the chain vector and 
        return the chain vector, the energies vector, and the gradients vector. 

        Will also update the chain vector in the chain instance. 

        Here: Impose the coordinate constraints on the chain vector. 

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
        # Impose the coordinate constraints. 
        if self._apply_cons_chain_vec == True:
            chain_vec = self._get_cons_chain_vec(chain_vec)
            self._chain.set_chain_vec(chain_vec)
        
        chain_vec            = self._chain.get_chain_vec()
        eners_vec, grads_vec = self._chain.get_eners_grads()

        return chain_vec, eners_vec, grads_vec
        