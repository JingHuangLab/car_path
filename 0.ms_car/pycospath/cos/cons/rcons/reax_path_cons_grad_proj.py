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
import numpy
from typing import Tuple
from io import TextIOWrapper

from pycospath.cos   import ReaxPathCons
from pycospath.chain import ChainBase
from pycospath.utils import row_wise_proj

class ReaxPathConsGradProj(ReaxPathCons):
    """The RPwHC method with path tangent estimations and gradient projections. 

    """

    def __init__(self, 
                 chain:           ChainBase, 
                 verbose:         bool = True, 
                 cons_thresh:     float = 1e-8, 
                 cons_maxitr:     int = 200, 
                 cons_log_writer: TextIOWrapper = None):
        """Create a RPwHC calculation with path tangent estimations and gradient
        projections.

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        
        cons_thresh: float, default=1e-8
            The threshold under which the holonomic constraints on equal RMS 
            distances b/w adjacent replicas are considered converged. 

        cons_maxitr: int, default=200
            The maximum no. iterations that the Lagrangian multiplier solver is
            allowed to loop to search for the constrained chain vector. 

        cons_log_writer: TextIOWrapper, default=None
            A TextIOWrapper object to log the Lagrangian update steps, it must
            implement a write() method for echoing information. None means no 
            logger. 
        """
        ReaxPathCons.__init__(self, chain, verbose, 
                              cons_thresh, cons_maxitr, 
                              cons_log_writer)
    
        # If applying the gradient projections on the gradient vector.
        self._apply_proj_grads_vec = True
        
    def _get_path_tan(self,
                      chain_vec: numpy.ndarray
                      ) -> numpy.ndarray:
        """Get the path tangent vector at each replica along the chain. 

        Here: Approximated tangent. 
            r_{i-1} - r_{i+1} at replica i
            for all replicas along the path (except end-point replicas).
            The tangent of the first and the last replica is manualy set to be 
            the linear tangent.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector on which the path tangent vectors are computed.

        Returns
        -------
        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs).
            The path tangent vector at each replica along the chain.
        """
        path_tan_vec = ReaxPathCons._get_path_tan(self, chain_vec)
        path_tan_vec[1:-1] = chain_vec[0:-2] - chain_vec[2:  ]
        return path_tan_vec

    def _enable_proj_grads_vec(self):
        """Enable the gradient projections for the gradients vector."""
        self._apply_proj_grads_vec = True

    def _disable_proj_grads_vec(self):
        """Disable the gradient projections for the gradients vector."""
        self._apply_proj_grads_vec = False
    
    def _get_proj_grads_vec(self, 
                            chain_vec: numpy.ndarray, 
                            grads_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Determine the gradients vector projected perpendicular to the path
        and return the projected gradients.

        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector. 

        grads_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The gradients vector of the chain vector.

        Returns
        -------
        proj_grads_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The projected gradients vector of the chain vector.
        """
        # Path tangent vectors.
        path_tan_vec = self._get_path_tan(chain_vec)

        # Project potential gradients of intermediate replicas perpendicular to 
        # the path tangent vector (except the first and the last replica). 
        proj_grads_vec = numpy.copy(grads_vec)
        proj_grads_vec[1:-1] =   grads_vec[1:-1]                               \
                               - row_wise_proj(path_tan_vec[1:-1], 
                                               grads_vec[1:-1])
        
        return proj_grads_vec

    def impose(self, 
               chain_vec: numpy.ndarray
               ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Impose the chain-of-states (cos) condition on the chain vector and 
        return the chain vector, the energies vector, and the gradients vector. 

        Will also update the chain vector in the chain instance. 

        Here: Impose the coordinate constraints on the chain vector;
              Impose the gradient projections on the gradients vector.

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

        # Impose the gradients projections. 
        if self._apply_proj_grads_vec == True:
            grads_vec = self._get_proj_grads_vec(chain_vec, grads_vec)
            
        return chain_vec, eners_vec, grads_vec