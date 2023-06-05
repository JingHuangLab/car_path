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

from scipy.interpolate import CubicSpline

class IntpolBase(abc.ABC):
    """The base class of interpolators for the parametrization of the path."""
    
    def __init__(self):
        """Create an interpolator."""
        self._is_fitted = False

    @abc.abstractmethod
    def fit(self, 
            x: numpy.ndarray, 
            y: numpy.ndarray
            ) -> None:
        """Fit the interpolator for y=intpol(x).
        
        Parameters
        ----------
        x: numpy.ndarray
            1-D numpy array giving values of x.

        y: numpy.ndarray
            1-D numpy array giving values of y.
        """
        self._is_fitted = True

    @abc.abstractmethod
    def transform(self, 
                  x: numpy.ndarray
                  ) -> numpy.ndarray:
        """Get the y values from the interpolated spline at x. 

        Parameters
        ----------
        x: numpy.ndarray
            1-D numpy array giving values of x.
        """
        if self._is_fitted == False:
            raise RuntimeError("transform: interpolator has not been fitted.")
        
    @abc.abstractmethod
    def get_grad(self, 
                      x: numpy.ndarray
                      ) -> numpy.ndarray:
        """Get the gradients/tangents from the interpolated line at x.

        Parameters
        ----------
        x: numpy.ndarray
            1-D numpy array giving values of x.
        """
        if self._is_fitted == False:
            raise RuntimeError("get_grad: interpolator has not been fitted.")
        
class CubicSplineIntpol(IntpolBase):
    """The cubic spline interpolator for the String Method."""

    def __init__(self):
        """Create a cubic spline interpolator."""
        IntpolBase.__init__(self)

        self._intpol = None

    def fit(self, 
            x: numpy.ndarray, 
            y: numpy.ndarray
            ) -> None:
        """Fit the interpolator as y=intpol(x).
        
        Parameters
        ----------
        x: numpy.ndarray
            1-D numpy array giving values of x.

        y: numpy.ndarray
            1-D numpy array giving values of y.

        Returns
        -------
        self: CubicSplineInterpolator
            The fitted CubicSplineInterpolator object.
        """
        IntpolBase.fit(self, x, y)

        self._intpol = CubicSpline(x, y)

    def transform(self, 
                  x: numpy.ndarray
                  ) -> numpy.ndarray:
        """Get new interpolated values from the interpolator. fit(x, y) must be 
        called b/f using this method.
        
        Parameters
        ----------
        x: numpy.ndarray
            1-D values on the x axis to be evaluated. 

        Returns
        -------
        y: numpy.ndarray
            The interpolated values at x.
        """
        IntpolBase.transform(self, x)

        y = self._intpol(x, nu=0)
        return y

    def get_grad(self, 
                 x: numpy.ndarray
                 ) -> numpy.ndarray:
        """Get the gradients (string tangents) from the interpolated line at x.
        fit(x, y) must be called b/f this method.
        
        Parameters
        ----------
        x: numpy.ndarray, 
            1-D values on the x axis where the tangents are to be evaluated. 

        Returns
        -------
        grads: numpy.ndarray
            The gradients of the interpolated spline at x.
        """
        IntpolBase.get_grad(self, x)
        
        grads = self._intpol(x, nu=1)
        return grads
        