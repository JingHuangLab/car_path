# Unit test for Rep2D objects.
# Zilin Song, 19 Jun 2022

import pytest

import numpy
from random import randint

from openmm import VerletIntegrator, unit, Context as OmmContext,              \
                   System as OmmSystem
from openmm.app import CharmmPsfFile, CharmmParameterSet, NoCutoff

from pycospath.comms.twodim import PotMuller, PotAsymDoubleWell,                \
                                   PotSymDoubleWell, Rep2D
from pycospath.comms.openmm import PotOpenMM

def test_constructor():
    """Test if __init__ works correctly."""

    good_pot = PotMuller()
    good_coor = numpy.random.random_sample(2)

    # test_bad_coor()
    bad_coor  = numpy.random.random_sample(randint(3, 9))

    with pytest.raises(ValueError,
                       match="Non-2D coordinates on 2D potential."):
        Rep2D(bad_coor, good_pot)

    # test_bad_pot()
    ffs = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm")
    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")
    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_cnt = OmmContext(omm_sys, VerletIntegrator(1.*unit.picosecond))
    bad_pot = PotOpenMM(omm_cnt)

    with pytest.raises(TypeError,
                       match="Rep2D must be initialized on Pot2DBase."):
        Rep2D(good_coor, bad_pot)

    # test_good_input()
    try:
        Rep2D(good_coor, good_pot)
    except Exception as e:
        assert False, f"Rep2D({good_coor}, {good_pot}) should not raise {e}."

def test_members():
    """Test various class members in Rep2D."""
    
    pot = PotMuller()
    coor = numpy.random.random_sample(2)

    rep = Rep2D(coor, pot)

    """Test if coor and rep_vec are binded correctly."""

    # Rep2D.get_coor()
    coor_return = rep.get_coor()
    assert (coor == coor_return).all(),                                        \
        f"Rep2D.get_coor: incorrect coor-rep_vec binding: __init__ "           \
        f"coor: {coor}; Rep2D.get_coor: {coor_return}"
    
    # Rep2D.get_rep_vec()
    rep_vec_return = rep.get_rep_vec()
    assert (coor == rep_vec_return).all(),                                     \
        f"Rep2D.get_rep_vec: incorrect coor-rep_vec binding: __init__ "        \
        f"rep_vec: {coor}; Rep2D.get_rep_vec: {rep_vec_return}"
    
    # Rep2D.set_coor()
    coor_new = numpy.random.random_sample(2)
    rep.set_coor(coor_new)

    coor_new_return = rep.get_coor()
    assert (coor_new_return == coor_new).all(),                                \
        f"Rep2D.set_coor: incorrect coor-rep_vec binding: set_coor "           \
        f"coor: {coor_new}; Rep2D.get_coor: {coor_new_return}"

    rep_vec_new_return = rep.get_rep_vec()
    assert (rep_vec_new_return == coor_new).all(),                             \
        f"Rep2D.set_coor: incorrect coor-rep_vec binding: set_coor "           \
        f"rep_vec: {coor_new}; Rep2D.get_rep_vec: {rep_vec_new_return}"
    
    # Rep2D.set_rep_vec()
    rep_vec_new = numpy.random.random_sample(2)
    rep.set_rep_vec(rep_vec_new)

    coor_new_return = rep.get_coor()
    assert (coor_new_return == rep_vec_new).all(),                             \
        f"Rep2D.set_rep_vec: incorrect coor-rep_vec binding: set_rep_vec "     \
        f"coor: {rep_vec_new}; Rep2D.get_coor: {coor_new_return}"

    rep_vec_new_return = rep.get_rep_vec()
    assert (rep_vec_new_return == rep_vec_new).all(),                          \
        f"Rep2D.set_rep_vec: incorrect coor-rep_vec binding: set_rep_vec "     \
        f"rep_vec: {rep_vec_new}; Rep2D.get_rep_vec: {rep_vec_new_return}"

def test_ener_grad():
    """Test if the energy and gradients are correct."""

    # Muller pot.
    muller_coor = numpy.random.random_sample(2)*2.8 - numpy.asarray([1.5, .5])
    muller_pot  = PotMuller()
    muller_rep  = Rep2D(muller_coor, muller_pot)
    
    pot_ener, pot_grad = muller_pot.get_ener_grad(muller_coor)
    rep_ener, rep_grad = muller_rep.get_ener_grad()

    assert rep_ener == pot_ener,                                               \
        f"Rep2D on PotMuller: rep_get_pot_grad()[0] and pot.get_grad()[0] "    \
        f"should be the same. pot_ener: {pot_ener}; rep_ener: {rep_ener}. "

    assert (rep_grad == pot_grad).all(),                                       \
        f"Rep2D on PotMuller: rep_get_pot_grad()[1] and pot.get_grad()[1] "    \
        f"should be the same. pot_ener: {pot_grad}; rep_ener: {rep_grad}. "

    # Symmetric Double Well pot.
    dw_sym_coor = numpy.random.random_sample(2)*4. - 2
    dw_sym_pot  = PotSymDoubleWell()
    dw_sym_rep  = Rep2D(dw_sym_coor, dw_sym_pot)
    
    pot_ener, pot_grad = dw_sym_pot.get_ener_grad(dw_sym_coor)
    rep_ener, rep_grad = dw_sym_rep.get_ener_grad()

    assert rep_ener == pot_ener,                                               \
        f"Rep2D on PotSymDw: rep_get_pot_grad()[0] and pot.get_grad()[0] "     \
        f"should be the same. pot_ener: {pot_ener}; rep_ener: {rep_ener}. "

    assert (rep_grad == pot_grad).all(),                                       \
        f"Rep2D on PotSymDW: rep_get_pot_grad()[1] and pot.get_grad()[1] "     \
        f"should be the same. pot_ener: {pot_grad}; rep_ener: {rep_grad}. "

    # Asymmetric Double Well pot.
    dw_asy_coor = numpy.random.random_sample(2)*4.  - 2
    dw_asy_pot  = PotAsymDoubleWell()
    dw_asy_rep  = Rep2D(dw_asy_coor, dw_asy_pot)
    
    pot_ener, pot_grad = dw_asy_pot.get_ener_grad(dw_asy_coor)
    rep_ener, rep_grad = dw_asy_rep.get_ener_grad()

    assert rep_ener == pot_ener,                                               \
        f"Rep2D on PotAsymDw: rep_get_pot_grad()[0] and pot.get_grad()[0] "    \
        f"should be the same. pot_ener: {pot_ener}; rep_ener: {rep_ener}. "

    assert (rep_grad == pot_grad).all(),                                       \
        f"Rep2D on PotAsymDw: rep_get_pot_grad()[1] and pot.get_grad()[1] "    \
        f"should be the same. pot_ener: {pot_grad}; rep_ener: {rep_grad}. "
