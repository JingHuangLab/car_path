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
from io import TextIOWrapper
import numpy
from typing import Union

from pycospath.cos import ConsCosBase, ConsCimgBase
from pycospath.opt import ConsOptBase

class ConsGradientDescent(ConsOptBase):
    """The Gradient Descent optimizer for constraint-based chain-of-states."""

    def __init__(self,
                 cos:         Union[ConsCosBase, ConsCimgBase],
                 fix_mode:    str = 'none',
                 dy_max:      float = 5,
                 ener_logger: TextIOWrapper = None,
                 gd_eta:      float = .01,
                 gd_eta_scal: float = 1.):
        """Create a gradient descent optimizer for constraint-based chain-of-
        states calculations.
        
        Parameters
        ----------
        cos: Union[ConsCosBase, ConsCimgBase]
            The constraint-based chain-of-states calculation to be optimized.

        fix_mode: str, default = 'none'
            Specifies if certain replicas coordinates should be fixed during the 
            chain vector optimization. 
            Allowed values are: 'none', 'both', 'head', and 'tail. 
            'none': Coordinates of all replicas will be optimized; 
            'both': Coordinates of all replicas except the two end-point replicas
                    (the first and the last) will be optimized;
            'head': Coordinates of all replicas except the first replica will be
                    optimized;
            'tail': Coordinates of all replicas except the last replica will be 
                    optimized. 
        
        dy_max: float, default=5
            The maximum step size (the gradient vector of the object function) 
            the optimizer is allowed to take for each descending step. If the 
            maximum value of the element in the step size (the gradient) vector 
            is higher than this value, the whole vector is scaled down to have 
            its maximum of dy_max. 
            Units in: kcal/mol/Angstrom. 
        
        gd_eta: float, default=.01
            The descending rate of the optimizer. 

        gd_eta_ada: float, default=1. 
            The scaling factor of the descending rate after each descent step. 
            Recommended values are b/w 0.90 ~ 1.00.
            gd_eta_ada > 1., increasing descending rates;
            gd_eta_ada = 1., constant descending rates;
            gd_eta_ada < 1., decreasing descending rates. 
        """
        ConsOptBase.__init__(self, cos, fix_mode, dy_max, ener_logger)

        self._gd_eta      = gd_eta
        self._gd_eta_scal = gd_eta_scal

    def _descend(self,
                 x:  numpy.ndarray,
                 y:  numpy.ndarray,
                 dy: numpy.ndarray
                 ) -> numpy.ndarray:
        """Execute one optimizer descending step. 
        
        Parameters
        ----------
        x: numpy.ndarray of shape (n_reps, n_dofs)
            The input chain_vec.

        y: numpy.ndarray of shape (n_reps)
            The input eners_vec.

        dy: numpy.ndarray of shape (n_reps, n_dofs)
            The input grads_vec.

        Returns
        -------
        x_dscn: numpy.ndarray of shape (n_reps, n_dofs)
            The descended x. 
        """
        if self._ener_is_log == True:
            self._ener_logger.write(f'{y}\n')
            self._ener_logger.flush()

        # Scale down the gradient vector if too large. 
        dy_max = numpy.max(numpy.abs(dy))

        if dy_max > self._dy_max:
            dy *= (self._dy_max / dy_max)

        x_dscn = x - self._gd_eta * dy

        # optionally adaptive descending rate.
        self._gd_eta *= self._gd_eta_scal
        
        return x_dscn
    