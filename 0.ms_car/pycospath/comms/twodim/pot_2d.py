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
import numpy, abc
from typing import Tuple

from pycospath.comms import PotBase

class Pot2DBase(PotBase):
    """The base class for a 2D potential."""

    @abc.abstractmethod
    def get_ener(self, 
                 coor: numpy.ndarray
                 ) -> float:
        """Get the energy of the coor numpy.ndarray.
        
        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The input system coordinate numpy.ndarray.

        Returns
        -------
        ener: float
            The energy value.
        """
        if coor.shape != (2, ):
            raise ValueError("Non-2D coordinates on 2D potential.")

    @abc.abstractmethod
    def get_ener_grad(self, 
                      coor: numpy.ndarray
                      ) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the coor numpy.ndarray.
        
        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The input system coordinate numpy.ndarray.

        Returns
        -------
        (ener, grad): (float, numpy.ndarray}
            ener: The energy value;
            grad: The gradient numpy.ndarray.
        """
        if coor.shape != (2, ):
            raise ValueError("Non-2D coordinates on 2D potential.")
    
    def get_pes(self, 
                xmin: float, 
                xmax: float, 
                ymin: float,
                ymax: float,
                grid: int,
                ecut: float
                ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Get the energies of the 2D potential within a specified region.

        Parameters
        ----------
        xmin: float
            The minimal of x axis.
        
        xmax: float
            The maximal of x axis. 

        ymin: float
            The minimal of y axis.
        
        ymax: float
            The maximal of y axis. 
        
        grid: int
            The number of grid points on each dimension. Must be positive.

        ecut: float
            Energy values higher than ecut is set to ecut for ploting purpose.

        Returns
        -------
        (x_coor, y_coor, v)
            A tuple that could be plotted directly with matplotlib:
            ```
            ax.contourf(x_coor, y_coor, v)
            ```
        """        
        # build the x-y grid dimensions
        x_coor = numpy.linspace(xmin, xmax, num=grid)
        y_coor = numpy.linspace(ymin, ymax, num=grid)
        
        xx, yy = numpy.mgrid[xmin:xmax:grid*1j, ymin:ymax:grid*1j]

        # coor.shape=(grid^2, 2)
        coor   = numpy.vstack([xx.ravel(), yy.ravel()]).T 

        # compute potential at each coor. 
        v = numpy.zeros((coor.shape[0], ))
        
        for pt in range(coor.shape[0]):
            v[pt] = self.get_ener(coor[pt])

        # upper bound of V
        v = numpy.where(v<=ecut, v, ecut)

        v = v.reshape((grid, grid)).T

        return x_coor, y_coor, v

class PotMuller(Pot2DBase):
    """The 2D sMuller-Brown potential."""
    
    def __init__(self):
        """Create a Muller-Brown potential function."""
        Pot2DBase.__init__(self)

        # M-B pot constants.
        self._AA = [-200., -100.,  -170.,  15., ]
        self._aa = [  -1.,   -1.,   -6.5,  0.7, ]
        self._bb = [   0.,    0.,    11.,  0.6, ]
        self._cc = [ -10.,  -10.,   -6.5,  0.7, ] 
        self._xx = [   1.,    0.,   -0.5,  -1., ]
        self._yy = [   0.,   0.5,    1.5,   1., ]

    def get_ener(self, 
                 coor: numpy.ndarray
                 ) -> float:
        """Get the energy of the coor numpy.ndarray.
        
        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The input system coordinate numpy.ndarray.

        Returns
        -------
        ener: float
            The energy value.
        """
        Pot2DBase.get_ener(self, coor)

        ener = 0
        
        for i in range(4):
            ener += self._AA[i] * numpy.exp(
                          self._aa[i] * (coor[0] - self._xx[i])**2
                        + self._bb[i] * (coor[0] - self._xx[i])
                                      * (coor[1] - self._yy[i])
                        + self._cc[i] * (coor[1] - self._yy[i])**2
                    )
        return ener

    def get_ener_grad(self, 
                      coor: numpy.ndarray
                      ) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the coor numpy.ndarray.
        
        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The input system coordinate numpy.ndarray.

        Returns
        -------
        (ener, grad): (float, numpy.ndarray}
            ener: The energy value;
            grad: The gradient numpy.ndarray.
        """
        Pot2DBase.get_ener_grad(self, coor)

        ener = self.get_ener(coor)

        grad = numpy.zeros(coor.shape)
        for i in range(4):
            u = self._AA[i] * numpy.exp(
                          self._aa[i] * (coor[0] - self._xx[i])**2
                        + self._bb[i] * (coor[0] - self._xx[i])
                                      * (coor[1] - self._yy[i])
                        + self._cc[i] * (coor[1] - self._yy[i])**2
                    )

            grad[0] += u * (2. * self._aa[i]*(coor[0] - self._xx[i])
                               + self._bb[i]*(coor[1] - self._yy[i]))
            grad[1] += u * (2. * self._cc[i]*(coor[1] - self._yy[i])
                               + self._bb[i]*(coor[0] - self._xx[i]))

        return ener, grad
    
    def get_pes(self, 
                xmin: float = -1.5, 
                xmax: float =  1.3, 
                ymin: float = -0.5,
                ymax: float =  2.3,
                grid: int   =  101, 
                ecut: float =  200,
                ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Get the energies of the 2D potential within a specified region.

        Parameters
        ----------
        xmin: float
            The minimal of x axis.
        
        xmax: float
            The maximal of x axis. 

        ymin: float
            The minimal of y axis.
        
        ymax: float
            The maximal of y axis. 
        
        grid: int
            The number of grid points on each dimension. Must be positive.

        ecut: float
            Energy values higher than ecut is set to ecut for ploting purpose.

        Returns
        -------
        (x_coor, y_coor, v)
            A tuple that could be plotted directly with matplotlib:
            ```
            ax.contourf(x_coor, y_coor, v)
            ```
        """  
        return Pot2DBase.get_pes(self, xmin, xmax, ymin, ymax, grid, ecut)
    
class PotSymDoubleWell(Pot2DBase):
    """The Symmetric Double Well potential."""

    def __init__(self):
        """Create a Symmetric Double Well potential function."""
        Pot2DBase.__init__(self)

        # sym-DW pot constants.
        self._a =  30. # Total width of x. 
        self._b =  75. # Max energy difference on MEP.
        self._c =   0. # Asymmetric x minimum energies. 
        self._d =  60. # Total width on y. 
        self._e = -80. # energy offset (to fit for PotMuller magnitude) = E_TS

    def get_ener(self, 
                 coor: numpy.ndarray
                 ) -> float:
        """Get the energy of the coor numpy.ndarray.
        
        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The input system coordinate numpy.ndarray.

        Returns
        -------
        ener: float
            The energy value.
        """
        Pot2DBase.get_ener(self, coor)

        ener =   self._a * coor[0]**4                                          \
               - self._b * coor[0]**2                                          \
               + self._c * coor[0]                                             \
               + self._d * coor[1]**2                                          \
               + self._e
        
        return ener
    
    def get_ener_grad(self, 
                      coor: numpy.ndarray
                      ) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the coor numpy.ndarray.
        
        Parameters
        ----------
        coor: numpy.ndarray of shape (2, )
            The input system coordinate numpy.ndarray.

        Returns
        -------
        (ener, grad): (float, numpy.ndarray}
            ener: The energy value;
            grad: The gradient numpy.ndarray.
        """
        Pot2DBase.get_ener_grad(self, coor)

        ener = self.get_ener(coor)

        grad = numpy.zeros(coor.shape)

        grad[0] = 4. * self._a * coor[0]**3 - 2. * self._b * coor[0] + self._c
        grad[1] = 2. * self._d * coor[1]

        return ener, grad
    
    def get_pes(self, 
                xmin: float = -2., 
                xmax: float =  2., 
                ymin: float = -2.,
                ymax: float =  2.,
                grid: int   = 201,
                ecut: float = 200 
                ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Get the energies of the 2D potential within a specified region.

        Parameters
        ----------
        xmin: float
            The minimal of x axis.
        
        xmax: float
            The maximal of x axis. 

        ymin: float
            The minimal of y axis.
        
        ymax: float
            The maximal of y axis. 
        
        grid: int
            The number of grid points on each dimension. Must be positive.

        ecut: float
            Energy values higher than ecut is set to ecut for ploting purpose.

        Returns
        -------
        (x_coor, y_coor, v)
            A tuple that could be plotted directly with matplotlib:
            ```
            ax.contourf(x_coor, y_coor, v)
            ```
        """  
        return Pot2DBase.get_pes(self, xmin, xmax, ymin, ymax, grid, ecut)
    
class PotAsymDoubleWell(PotSymDoubleWell):
    """The Asymmetric Double Well potential."""

    def __init__(self):
        """Create an Asymmetric Double Well potential function."""
        PotSymDoubleWell.__init__(self)

        self._c = 20.   # Asymmetric x minimum energies. 