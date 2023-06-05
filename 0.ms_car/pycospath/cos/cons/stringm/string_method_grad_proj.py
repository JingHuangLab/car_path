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
from typing import Tuple, Union

from pycospath.cos   import StringMethod
from pycospath.chain import ChainBase
from pycospath.utils import IntpolBase, row_wise_proj

class StringMethodGradProj(StringMethod):
    """The String Method (original version):

    String method for the study of rare events.
    W. E, W. Ren, E. Vanden-Eijnden, Phys. Rev. B, 2002, 66, 052301.  
    10.1103/PhysRevB.66.052301

    This is the original string method. It differs from its simplified version 
    by that it projects the gradients perpendicularly to the string (the path)
    b/f the path descending. 
    """

    def __init__(self, 
                 chain:   ChainBase, 
                 verbose: bool = False, 
                 intpol:  Union[str, IntpolBase] = 'cspline'):
        """Create a String Method calculation. 

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        
        intpol: Union[str, IntpolBase], default='cspline'
            An instance of the type of interpolator to fit for the path. 
        """
        StringMethod.__init__(self, chain, verbose, intpol)

        # If applying the gradient projections on the gradient vector.
        self._apply_proj_grads_vec = True

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
        # NOTE: Initialize the interpolators for path tangent calculations, no
        #       chain_vec update performed for the initial re-parametrization. 
        if self._init_reparam == False: 
            _ = self._get_cons_chain_vec(chain_vec)

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
