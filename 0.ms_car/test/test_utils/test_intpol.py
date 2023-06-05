# Unit test the interpolators. 
# Zilin Song, 17 Jun 2022

import pytest

import numpy
from random import randint

from pycospath.utils import CubicSplineIntpol

def test_cubic_spline_intpol_states():
    """Test if various Exception are raised with different class states."""
    
    # Do get_grad and transform without fitting the intpol.
    # Note that cubic_spline_intpol is implemented in scipy, the correctness of 
    # the interpolation algorithm need not to be tested in here.

    # Unfitted.
    cspline_intpol = CubicSplineIntpol()
    
    n_rep = randint(5, 100)
    x = (numpy.arange(n_rep) + numpy.random.random_sample((n_rep, )) ) / n_rep

    with pytest.raises(RuntimeError,
                       match="transform: interpolator has not been fitted."):
        cspline_intpol.transform(x)

    with pytest.raises(RuntimeError,
                       match="get_grad: interpolator has not been fitted."):
        cspline_intpol.get_grad(x)

    # Fit intpol
    y = (numpy.random.random_sample((n_rep, )) - .5) * randint(2, 10)

    cspline_intpol.fit(x, y)

    try:
        cspline_intpol.transform(numpy.arange(n_rep+1) / n_rep)
        cspline_intpol.get_grad( numpy.arange(n_rep+1) / n_rep)
    except Exception as e:
        assert False,                                                          \
            f"Fitted interpolators should be raise {e} when transform/get_grad."