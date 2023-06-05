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

from pycospath.cos import CimgBase, ConsCosBase

class ConsCimgBase(CimgBase):
    """The base class of constraint-based climbing image (ci) chain-of-states 
    (cos) methods.
    """

    def __init__(self, 
                 cos: ConsCosBase):
        """Create a constraint-based climbing image (ci) chain-of-states (cos) 
        calculation.

        Parameters
        ----------
        cos: ConsCosBase
            The instance of the constraint-based cos method to perform the ci 
            protocol. 
        """
        if not isinstance(cos, ConsCosBase):
            raise TypeError("Constraint-based Climbing Image decorator must be "
                            "initialized on instances of the constraint-based "
                            "Chain-of-States method. ")

        CimgBase.__init__(self, cos)

        # If applying the constraint condition on the chain vector. 
        self._apply_cons_chain_vec = True 

    ##################################################
    # NOTE: Belows are decorated ConsCosBase methods.#
    ##################################################

    def _enable_cons_chain_vec(self) -> None:
        """Decorated ConsCosBase class member. 
        Enable the constraint condition on the chain vectors.
        """
        self._apply_cons_chain_vec = True
    
    def _disable_cons_chain_vec(self) -> None:
        """Decorated ConsCosBase class member. 
        Disable the constraint condition on the chain vectors.
        """
        self._apply_cons_chain_vec = False

    @abc.abstractmethod
    def _get_cons_chain_vec(self, 
                            chain_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Decorated ConsCosBase class member. 
        Determine and return the chain vector that satisfies the constraint
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

        Here: Determine the index of the climbing image replica via the replica 
              energy values. 
              Impose the coordinate constraints on the chain vector;
              Impose the climbing image gradient projection on the highest
              energy replica.

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
        # NOTE: Initialize the index of the climbing image (ci) replica for the 
        #       ci protocols, no grads_vec projection performed for the initial
        #       determination of the ci replica index. 
        if self._init_cimg_rep == False:
            self.get_chain().set_chain_vec(chain_vec)
            eners_vec, grads_vec = self.get_chain().get_eners_grads()

            _ = self._get_cimg_grads_vec(chain_vec, eners_vec, grads_vec)
            
        # Impose the coordinate constraints. 
        if self._apply_cons_chain_vec == True:
            chain_vec = self._get_cons_chain_vec(chain_vec)
            self.get_chain().set_chain_vec(chain_vec)

        chain_vec            = self.get_chain().get_chain_vec()
        eners_vec, grads_vec = self.get_chain().get_eners_grads()

        # Impose the ci gradient projection on the highest energy replica. 
        if self._apply_cimg_grad_vec == True:
            grads_vec = self._get_cimg_grads_vec(chain_vec, 
                                                 eners_vec, 
                                                 grads_vec)
            
        return chain_vec, eners_vec, grads_vec
