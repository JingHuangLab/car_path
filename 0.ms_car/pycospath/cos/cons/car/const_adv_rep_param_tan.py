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
import numpy, copy
from typing import Union
from io import TextIOWrapper

from pycospath.cos   import ConstAdvRep
from pycospath.chain import ChainBase
from pycospath.utils import IntpolBase, CubicSplineIntpol

class ConstAdvRepParamTan(ConstAdvRep):
    """The Constant Advancement Replicas method with Parametrized Tangent."""
    
    def __init__(self, 
                 chain:                 ChainBase, 
                 verbose:               bool = True, 
                 cons_thresh:           float = 1e-8, 
                 cons_maxitr:           int = 10, 
                 cons_curvreglr_thresh: int = 30,
                 cons_stepreglr_thresh: float = 1.25, 
                 cons_stepreglr_dnscal: float = .8, 
                 cons_log_writer:       TextIOWrapper = None, 
                 intpol:                Union[str, IntpolBase] = 'cspline'):
        """Create a Constant Advancement Replicas with Parametrized Tangent 
        calculation.

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 

        cons_thresh: float, default=1e-8
            The threshold under which the holonomic constraints on equal RMS 
            distances b/w adjacent replicas are considered converged. 

        cons_maxitr: int, default=10
            The maximum no. iterations that the Lagrangian multiplier solver is
            allowed to loop to search for the constrained chain vector. 
        
        cons_curvreglr_thresh: int, default=30
            The threshold (in degrees) above which the curvature regularization
            becomes effective for the chain vector updates. Allowed values are 
            (0, 45)
            It explicitly controls the maximum angle allowed b/w the two vectors
            r_{i-1}-r_{i+1} and r{i-1}-r{i} of any three neighboring replicas on 
            the path. 
        
        cons_stepreglr_thresh: float, default=1.25 
            The threshold above which the regularization protocol for preventing 
            overstepping of Lagrangian multipliers becomes effective for the 
            chain vector updates. Allowed values are (1., 1.8].
            This regularization protocol explicitly determines if the Lagrangian 
            multiplier updates have overstepped from references to the previous 
            step and applies a Line-Search protocol to determine a set of
            Lagrangian multipliers with reasonable/regularized step size:
            If the mean-RMS of the chain-of-replicas after the coordinate update
            is higher by {cons_stepreglr_thresh} times mean-RMS before the
            coordinate update, the current Lagrangian multipliers are determined 
            to be overstepping, and is uniformly scaled down by a factor of
            {cons_stepreglr_dnscal}. 
            
        cons_stepreglr_dnscal: float, default=.8
            The down-scaling factor for the step size regularizing Line-Search. 
            The Lagrangian multipliers were scaled down by this factor until a
            proper step size is determined. Allowed values are (0., 1.).

        cons_log_writer: TextIOWrapper, default=None
            A TextIOWrapper object to log the Lagrangian update steps, it must
            implement a write() method for echoing information. None means no 
            logger. 

        intpol: Union[str, IntpolBase], default='cspline'
            An instance of the type of interpolator to fit for the path. 
        """
        ConstAdvRep.__init__(self, chain, verbose, 
                             cons_thresh, cons_maxitr, 
                             cons_curvreglr_thresh, 
                             cons_stepreglr_thresh, 
                             cons_stepreglr_dnscal, 
                             cons_log_writer)
        
        # Interpolators.
        if isinstance(intpol, IntpolBase):
            self._intpol_cls = intpol
        
        elif isinstance(intpol, str):
            
            if not intpol in ['cspline']:
                raise ValueError(f"Unknown interpolator spec: {intpol}")

            self._intpol_cls = CubicSplineIntpol()

        else:
            raise TypeError("The interpolator must be an IntpolBase object.")
        
    def _get_path_tan(self,
                      chain_vec: numpy.ndarray
                      ) -> numpy.ndarray:
        """Get the path tangent vector at each replica along the chain. 

        Here: Parametrized tangent. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector on which the path tangent vectors are computed.

        Returns
        -------
        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs).
            The path tangent vector at each replica along the chain.
        """
        # Compute the total arclength as RMS distances along the chain.
        rms     = numpy.zeros((chain_vec.shape[0], ))
        rms[1:] = self.get_chain().get_rms(chain_vec)

        alpha_vec = numpy.cumsum(rms) / numpy.sum(rms)

        # Compute path tangent vector. 
        str_tangent_vec = numpy.zeros(chain_vec.shape)

        # Parametrize the string and get its tangent. 
        for i_dof in range(chain_vec.shape[1]):
            intpol = copy.deepcopy(self._intpol_cls)
            intpol.fit(alpha_vec, chain_vec[:, i_dof])
            str_tangent_vec[:, i_dof] = intpol.get_grad(alpha_vec)

        return str_tangent_vec
    