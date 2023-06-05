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

from pycospath.comms        import RepBase
from pycospath.comms.twodim import Pot2DBase

class Rep2D(RepBase):
    """Implements the 2D replica."""

    def __init__(self, 
                 coor: numpy.ndarray, 
                 pot:  Pot2DBase):
        """Create a 2D replica object. 

        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The 2D replica coordinate numpy.ndarray. 
        
        pot: Pot2DBase
            The 2D replica potential object.
        """
        if not isinstance(pot, Pot2DBase):
            raise TypeError("Rep2D must be initialized on Pot2DBase.")
        
        if coor.shape != (2, ):
            raise ValueError("Non-2D coordinates on 2D potential.")

        RepBase.__init__(self, coor, pot)
        
    def coor_to_rep_vec(self, 
                        coor: numpy.ndarray
                        ) -> numpy.ndarray:
        """Convert the replica coordinate numpy.ndarray (coor) to the 
        representation of the replica vector. 
        
        Parameters
        ----------
        coor: numpy.ndarray
            The replica coordinate numpy.ndarray. 

        Returns
        -------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            A vector converted from the replica coordinate numpy.ndarray 
            (self._coor).
        """
        # copy for a new numpy.ndarray
        rep_vec = coor.copy()
        return rep_vec

    def rep_vec_to_coor(self, 
                        rep_vec: numpy.ndarray
                        ) -> numpy.ndarray:
        """Convert the representation of the replica vector (rep_vec) to a new 
        replica coordinate numpy.ndarray.

        Parameters
        ----------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            The representation of the replica vector.

        Returns
        -------
        coor: numpy.ndarray
            A new coordinate numpy.ndarray converted from the representation of 
            the replica vector (rep_vec).
        """
        # copy for a new numpy.ndarray
        coor = rep_vec.copy()
        return coor

    def get_ener_grad(self) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the replica vector (self._rep_vec). 
        
        Returns
        -------
        (ener, grad): (float, numpy.ndarray of shape (n_dofs, ) )
            ener: the energy of the replica vector;
            grad: the gradient of the replica vector.
        """
        pot:  Pot2DBase     = self._pot
        coor: numpy.ndarray = self._coor

        ener, grad = pot.get_ener_grad(coor)
        
        return ener, grad
    