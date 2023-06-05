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
from typing import Union

from pycospath.cos import ConsCosBase, ConsCimgBase
from pycospath.opt import ConsOptBase

class ConsAdaptiveMomentum(ConsOptBase):
    """The Adaptive Momentum optimizer for constraint-based chain-of-states."""

    def __init__(self,
                 cos:      Union[ConsCosBase, ConsCimgBase],
                 fix_mode: str = 'none',
                 dy_max:   float = 5,
                 am_eta:   float = .01,
                 am_b1:    float = .9, 
                 am_b2:    float = .999, 
                 am_e:     float = 1e-8):
        """Create a Adaptive Momentum optimizer for constraint-based chain-of-
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

        am_eta: float, default=.01
            The descending rate of the optimizer. 

        am_b1: float, default=0.9
            The AdaM beta_1.

        am_b2: float, default=0.999
            The AdaM beta_2.

        am_e: float, default 1e-8
            The AdaM epsilon.
        """
        ConsOptBase.__init__(self, cos, fix_mode, dy_max)

        self._am_eta = am_eta
        self._am_b1  = am_b1
        self._am_b2  = am_b2
        self._am_e   = am_e
        self._am_t   = 0          # Adam step counter.

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
        # Scale down the gradient vector if too large. 
        dy_max = numpy.max(numpy.abs(dy))

        if dy_max > self._dy_max:
            dy *= (self._dy_max / dy_max)

        # Descend.
        if self._am_t == 0:  # initialize M and V.
            self._am_M = 0
            self._am_V = 0

        self._am_t += 1

        # step1: 
        # M^{t+1} = beta_1 * M^{t} + (1-beta_1) * grad;
        self._am_M = self._am_b1 * self._am_M + (1-self._am_b1) * dy

        # step2: 
        # V^{t+1} = beta_2 * V^{t} + (1-beta_2) * grad**2;
        self._am_V = self._am_b2 * self._am_V + (1-self._am_b2) * dy**2

        # step3: 
        # M^{hat,t+1} = M^{t+1} / (1 - beta_1**t);
        adam_Mh = self._am_M / (1 - self._am_b1**self._am_t)

        # step4: 
        # V^{hat,t+1} = V^{t+1} / (1 - beta_2**t);
        adam_Vh = self._am_V / (1 - self._am_b2**self._am_t)

        # step5: 
        # x^(t+1) = x^{t} - eta * M^{hat,t+1} / (V^{hat,t+1} ** 0.5  + epsilon);
        x_dscn = x - self._am_eta * adam_Mh / (adam_Vh**0.5 + self._am_e)

        return x_dscn
