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
from typing import Tuple

class PotBase(abc.ABC):
    """The base class for a potential function."""

    @abc.abstractmethod
    def get_ener(self, 
                 coor: object
                 ) -> object:
        """Get the energy of the coor object.
        
        Parameters
        ----------
        coor: object
            The input system coordinate object.

        Returns
        -------
        ener: object
            The energy object.
        """
    
    @abc.abstractmethod
    def get_ener_grad(self, 
                      coor: object
                      ) -> Tuple[object, object]:
        """Get the energy and gradients of the coor object.
        
        Parameters
        ----------
        coor: object
            The input system coordinate object.

        Returns
        -------
        (ener, grad): (object, object)
            ener: The energy object;
            grad: The gradient object.
        """
