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

# Implementation Note I.
# The CimgBase class follows the Decorator/Wrapper Design Pattern: it implements
# the same method as in the CosBase class, by wrapping the CosBase class so that 
# the descending of the chain can be altered. The CimgBase does not subclass the 
# CosBase in order to bypass a very tricky "diamond inheritance" scenario. 

# Imports
import abc, numpy
from typing import Tuple

from pycospath.cos   import CosBase
from pycospath.chain import ChainBase
from pycospath.utils import row_wise_proj

class CimgBase(abc.ABC):
    """The base class of climbing image (ci) chain-of-states (cos) methods."""

    def __init__(self, 
                 cos: CosBase):
        """Create a climbing image (ci) chain-of-states (cos) calculation. 

        Parameters
        ----------
        cos: CosBase
            The instance of the cos method to perform the ci protocol. 
        """        
        self._cos = cos         # chain-of-states calculation instance.
        
        # If applying the ci protocol on the gradients vector. 
        self._apply_cimg_grad_vec = True

        # If the ci replica has been determined for at least once.
        self._init_cimg_rep = False
        # The index of the ci replica. 
        self._i_cimg_rep = None

    def get_cos(self):
        """Get the chain-of-states calculation instance decorated by self. 

        Returns
        -------
        cos: CosBase
            The chain-of-states calculation instance decorated by self. 
        """
        return self._cos

    def _enable_cimg_grads_vec(self) -> None:
        """Enable the climbing image protocol on the highest energy replica."""
        self._apply_cimg_grad_vec = True
    
    def _disable_cimg_grads_vec(self) -> None:
        """Disable the climbing image protocol on the highest energy replica."""
        self._apply_cimg_grad_vec = False
    
    def get_i_cimg_rep(self) -> int:
        """Get the index of the climbing image replica. 
        
        Raises a RuntimeError if the ci replica index has not been determined 
        for at least once. 

        Returns
        -------
        i_rep_cimg: int
            The index (starting from zero) of the climbing image replica as the 
            approximated transition state.
        """
        if self._init_cimg_rep == False:
            raise RuntimeError("The climbing image replica has not yet been "
                               "determined.")

        return self._i_cimg_rep

    def _get_cimg_grads_vec(self, 
                            chain_vec: numpy.ndarray, 
                            eners_vec: numpy.ndarray,
                            grads_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Get the gradients vector imposed with the climbing image protocol, 
        also determines the index of the climbing image replica. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the climbing image condition. 

        eners_vec: numpy.ndarray of shape (n_reps, )
            The energies vector of the chain vector to be imposed with the 
            climbing image condition.
            
        grads_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The gradients vector of the chain vector to be imposed with the 
            climbing image condition. 

        Returns
        -------
        cimg_grads_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The gradients vector imposed with the climbing image condition. 
        """
        # Determine the ci replica from the ener_vec of the first step. 
        if self._init_cimg_rep == False:
            self._init_cimg_rep = True
            self._i_cimg_rep = numpy.argmax(eners_vec)

        # Sanity check on the determined ci replica: don't climb the side ones.
        n_reps = self.get_chain().get_num_reps()

        if self._i_cimg_rep in [0, 1, n_reps-2, n_reps-1]:
            raise ValueError(f"Climbing the {self._i_cimg_rep}-th replica is "
                             f"ambiguous on a chain of {n_reps} replicas.")

        # Tangent vector and the gradient vector on the CI replica.
        tanv_ci = self._get_path_tan(chain_vec)[self._i_cimg_rep, :]
        grad_ci = grads_vec[self._i_cimg_rep, :]

        # CI protocol.
        # 1. Get the parallel components of the grad on the CI replica and; 
        grad_ci_parallel = row_wise_proj(tanv_ci[None, :], grad_ci[None, :])

        # 2. Reverse the parallel components of the the grad on the CI replica. 
        grads_vec[self._i_cimg_rep, :] = grad_ci - 2 * grad_ci_parallel

        return grads_vec

    ##############################################
    # NOTE: Belows are decorated CosBase methods.#
    ##############################################

    def is_verbose(self) -> bool:
        """Decorated CosBase class member. 
        Get if information related to the calculation should be verbose.
        
        Returns
        -------
        is_verbose: bool
            If information related to the calculation should be verbose. 
        """
        return self._cos.is_verbose()

    def get_chain(self) -> ChainBase:
        """Decorated CosBase class member. 
        Get the chain instance.

        Returns
        -------
        chain: ChainBase
            The chain instance for the chain-of-states calculation.
        """
        return self._cos.get_chain()

    def _get_path_tan(self, 
                      chain_vec: numpy.ndarray
                      ) -> numpy.ndarray:
        """Decorated CosBase class member. 
        Get the path tangent vector at each replica along the chain. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector on which the path tangent vectors are computed.

        Returns
        -------
        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs).
            The path tangent vector at each replica along the chain.
        """
        return self._cos._get_path_tan(chain_vec)

    @abc.abstractmethod
    def impose(self, 
               chain_vec: numpy.ndarray
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Decorated CosBase class member. 
        Impose the chain-of-states (cos) condition on the chain vector and 
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
