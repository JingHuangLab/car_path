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

import numpy, abc
from typing import Union
from io import TextIOWrapper

from pycospath.opt import OptBase
from pycospath.cos import ConsCosBase, ConsCimgBase

class ConsOptBase(OptBase):
    """The base class of constraint-based cos optimizers."""

    def __init__(self,
                 cos:      Union[ConsCosBase, ConsCimgBase],
                 fix_mode: str = 'none',
                 dy_max:   float = 5, 
                 ener_logger: TextIOWrapper = None):
        """Create a constraint-based cos optimizer.
        
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
        """
        if not isinstance(cos, (ConsCosBase, ConsCimgBase)):
            raise TypeError("Constraint-based optimizers must take constraint-"
                            "based chain-of-states calculations. ")

        OptBase.__init__(self, cos, fix_mode)

        self._dy_max = dy_max

        self._init_step = False # If self.step has been executed at least once.
        self._x_pv      = None  # NOTE: first descend then constrain requires a
        self._y_pv      = None  #       memory of the last descend step. 
        self._dy_pv     = None  #       see self.step(). 
        
        # Logger for lagrangian multiplier calculations.
        if ener_logger is None:
            self._ener_is_log = False
            self._ener_logger = None
        else:
            self._ener_is_log = True
            self._ener_logger = ener_logger

    def step(self) -> None:
        """Execute one optimization step."""

        if self._init_step == False:
            self._x_pv              = self._cos.get_chain().get_chain_vec()
            self._y_pv, self._dy_pv = self._cos.get_chain().get_eners_grads()

            self._init_step = True
        
        x = self._descend(self._x_pv, self._y_pv, self._dy_pv)

        # fix_mode processing.
        if self._fix_mode == 'both':
            x[0]  = numpy.copy(self._x_pv[0])
            x[-1] = numpy.copy(self._x_pv[-1])
        
        elif self._fix_mode == 'head':
            x[0]  = numpy.copy(self._x_pv[0])
        
        elif self._fix_mode == 'tail':
            x[-1] = numpy.copy(self._x_pv[-1])
        
        else:
            pass

        self._x_pv, self._y_pv, self._dy_pv = self._cos.impose(x)

    @abc.abstractmethod
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
