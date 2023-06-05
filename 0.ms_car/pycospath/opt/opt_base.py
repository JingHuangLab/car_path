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
import abc

from typing import Union
from pycospath.cos import CosBase, CimgBase

class OptBase(abc.ABC):
    """The base class for optimizers."""

    def __init__(self,
                 cos:      Union[CosBase, CimgBase],
                 fix_mode: str = 'none'):
        """Create an optimizer.
        
        Parameters
        ----------
        cos: Union[CosBase, CimgBase]
            The chain-of-states calculation to be optimized.

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
        """
        self._cos = cos
        self._loggers = []

        # The scheme for fixing replicas on the chain. 
        if not fix_mode.lower() in ['none', 'both', 'head', 'tail']:
            raise KeyError("Invalid value for 'fix_mode', allowed values are "
                           "'none', 'both', 'head', 'tail'. ")
        else:
            self._fix_mode = fix_mode
    
    def get_cos(self) -> Union[CosBase, CimgBase]: 
        """Get the chain-of-states calculation.
        
        Returns
        -------
        cos: Union[CosBase, CimgBase]
            The chain-of-states calculation.
        """
        return self._cos

    def append_logger(self, 
                      logger):
        """Append a logger to the optimizers. Loggers will log at each step() 
        call.
        
        Parameters
        ----------
        logger: 
        
        """

    @abc.abstractmethod
    def step(self,
             n_steps: int = 10
             ) -> None:
        """Execute descent optimization for {n_steps} steps. 
        
        Parameters
        ----------
        n_steps: int, default=10
            The number of descending steps.
        """
    