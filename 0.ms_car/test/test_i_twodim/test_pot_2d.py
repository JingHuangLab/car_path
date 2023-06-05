# Unit test for 2D potentials.
# Zilin Song, 18 Jun 2022

import pytest

import numpy
from random import randint
import matplotlib.pyplot as plt

from pycospath.comms.twodim import PotAsymDoubleWell, PotSymDoubleWell, PotMuller

def test_constructor():
    """Test if __init__ works correctly."""
    
    # Input any 3D array to see if pot raises error.
    a = numpy.random.random_sample(randint(3, 9))

    muller = PotMuller()
    dw_sym = PotSymDoubleWell()
    dw_asy = PotAsymDoubleWell()

    # For get_ener()
    with pytest.raises(ValueError,
                    match="Non-2D coordinates on 2D potential."):
        muller.get_ener(a)
    with pytest.raises(ValueError,
                    match="Non-2D coordinates on 2D potential."):
        dw_sym.get_ener(a)
    with pytest.raises(ValueError,
                    match="Non-2D coordinates on 2D potential."):
        dw_asy.get_ener(a)

    # For get_grad.
    with pytest.raises(ValueError,
                    match="Non-2D coordinates on 2D potential."):
        muller.get_ener_grad(a)
    with pytest.raises(ValueError,
                    match="Non-2D coordinates on 2D potential."):
        dw_sym.get_ener_grad(a)
    with pytest.raises(ValueError,
                    match="Non-2D coordinates on 2D potential."):
        dw_asy.get_ener_grad(a)

    # test_good_shape_input()
    a = numpy.random.random_sample(2)
    
    try:
        muller.get_ener(a)
        dw_sym.get_ener(a)
        dw_asy.get_ener(a)
        muller.get_ener_grad(a)
        dw_sym.get_ener_grad(a)
        dw_asy.get_ener_grad(a)
    except Exception as e:
        assert False, f"1D arrays {a} should not riase {e}."

def test_ener_grad():
    """Test if energy and gradients are correct."""

    # If energies returned from get_ener and get_grad are the same.
    a = numpy.random.random_sample(2)

    muller = PotMuller()
    e_ener = muller.get_ener(a)
    e_grad = muller.get_ener_grad(a)[0]
    assert e_ener == e_grad,                                                   \
        f"PotMuller: energies from get_ener() and get_grad should be "         \
        f"the same. get_ener: {e_ener}; get_grad: {e_grad}."

    dw_sym = PotSymDoubleWell()
    e_ener = dw_sym.get_ener(a)
    e_grad = dw_sym.get_ener_grad(a)[0]
    assert e_ener == e_grad,                                                   \
        f"PotSymDoubleWell: energies from get_ener() and get_grad should be "  \
        f"the same. get_ener: {e_ener}; get_grad: {e_grad}."

    dw_asy = PotAsymDoubleWell()
    e_ener = dw_asy.get_ener(a)
    e_grad = dw_asy.get_ener_grad(a)[0]
    assert e_ener == e_grad,                                                   \
        f"PotAsymDoubleWell: energies from get_ener() and get_grad should be " \
        f"the same. get_ener: {e_ener}; get_grad: {e_grad}."

def test_pes():
    """Not a test, plot all the PES figures to examples."""

    # Plot Muller potential.
    muller  = PotMuller()
    x, y, v = muller.get_pes()

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))

    ax.contourf(x, y, v, 200)
    
    ax.set_title('Muller-Brown PES')
    ax.set_xlim(-1.5, 1.3)
    ax.set_ylim(-0.5, 2.3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./examples/2d_muller/pes/muller_pes.png')

    # Plot Symmetric Double Well potential.
    dw_sym  = PotSymDoubleWell()
    x, y, v = dw_sym.get_pes()

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))

    ax.contourf(x, y, v, 200)

    ax.set_title('Symmetric Double Well PES')
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-2., 2.)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./examples/2d_double_well/pes/symdw_pes.png')

    # Plot Asymmetric Double Well potential.
    dw_asy  = PotAsymDoubleWell()
    x, y, v = dw_asy.get_pes()

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))

    ax.contourf(x, y, v, 200)
    ax.set_title('Asymmetric Double Well PES')
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-2., 2.)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./examples/2d_double_well/pes/asym_pes.png')

