# Unit test for RepOpenMM objects.
# Zilin Song, 19 Jun 2022

from random import randint
import pytest

import numpy

from openmm import VerletIntegrator, unit, Context as OmmContext,              \
                   System as OmmSystem, State as OmmState
from openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet,       \
                       PDBFile, NoCutoff
from openmm.unit import Quantity as OmmQuantity

from pycospath.comms.twodim import PotMuller
from pycospath.comms.openmm import RepOpenMM, PotOpenMM

def _get_omm_context() -> OmmContext:
    # Data files.
    ffs = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm")
    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")

    # Set PotOpenMM
    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_int = VerletIntegrator(1.*unit.picosecond) # requires a new integrator.
    omm_cnt = OmmContext(omm_sys, omm_int)

    return omm_cnt

def test_constructor():
    """Test if __init__ works correctly."""

    # Inputs
    omm_cnt = _get_omm_context()
    good_pot = PotOpenMM(omm_cnt)

    cor = CharmmCrdFile("./examples/omm_c36m_diala/diala.cor")
    good_coor: OmmQuantity = cor.positions

    n_atoms = omm_cnt.getSystem().getNumParticles()
    good_mscale = numpy.random.random_sample((n_atoms))
    
    # test_bad_pot()
    with pytest.raises(TypeError,
                       match="RepOpenMM must be initialized on PotOpenMM."):
        RepOpenMM(good_coor, PotMuller(), mscale_vec=good_mscale)

    # test_bad_coor_type()
    bad_coor_type = numpy.random.random_sample(20*3).reshape(20, 3)

    with pytest.raises(TypeError,
                       match="coor must be a openmm.unit.Quantity object."):
        RepOpenMM(bad_coor_type, good_pot, mscale_vec=good_mscale)

    # test_bad_coor_shape[0]()
    start = randint(1, n_atoms-2)
    end   = randint(start, n_atoms-1)
    bad_coor_shape = good_coor[start:end]
    with pytest.raises(ValueError, 
                       match="Incompatible shape of coor to openmm.system."):
        RepOpenMM(bad_coor_shape, good_pot, mscale_vec=good_mscale)

    # test_bad_mass_scaling_type()
    bad_mscale = 1.
    with pytest.raises(TypeError, 
                    match="Incompatible type of mscale_vec to numpy.ndarray."):
        RepOpenMM(good_coor, good_pot, mscale_vec=bad_mscale)

    # test_bad_mass_scaling_shape()
    start = randint(1, n_atoms-2)
    end   = randint(start, n_atoms-1)
    bad_mscale = good_mscale[start:end]
    with pytest.raises(ValueError, 
                       match="Incompatible shape of mscale_vec to the number "
                             "of atoms in openmm.system."):
        RepOpenMM(good_coor, good_pot, mscale_vec=bad_mscale)
    
    # test_good_mass_scaling_shape()
    good_mscale = numpy.asarray(numpy.random.random_sample(n_atoms))

    try:
        RepOpenMM(good_coor, good_pot, mscale_vec=good_mscale)
    except Exception as e:
        assert False,                                                          \
            f"RepOpenMM({good_coor}, {good_pot}, mass_scaling={good_mscale})"  \
            f" should not raise {e}"

def test_members():
    """Test various class members in RepOpenMM."""

    # Set PotOpenMM
    omm_cnt: OmmContext = _get_omm_context()
    pot = PotOpenMM(omm_cnt)

    cor = CharmmCrdFile("./examples/omm_c36m_diala/diala.cor")
    coor: OmmQuantity = cor.positions

    rep = RepOpenMM(coor, pot, is_mass_weighted = False)
    # the number of atoms. 
    n_atoms = omm_cnt.getSystem().getNumParticles()
    
    """Test get_num_dofs."""

    # RepOpenMM.get_num_dofs()
    n_dofs = omm_cnt.getSystem().getNumParticles() * 3
    assert rep.get_num_dofs() == n_dofs,                                       \
        f"RepOpenMM.get_num_dofs: incorrect n_dofs indicator: get_num_dofs " \
        f"Truth: {n_dofs}; RepOpenMM.get_num_dofs(): {rep.get_num_dofs()}."

    """Test if coor and rep_vec are binded correctly."""
    
    rep_vec = numpy.asarray(coor.value_in_unit(unit.angstrom)).flatten()

    # RepOpenMM.get_coor()
    coor_return = rep.get_coor() # returns the same object. 
    assert coor_return == coor,                                                \
        f"RepOpenMM.get_coor: incorrect coor-rep_vec binding: __init__ "       \
        f"coor: {coor}; RepOpenMM.get_coor: {coor_return}"

    # RepOpenMM.get_rep_vec()
    rep_vec_return = rep.get_rep_vec()
    assert (rep_vec_return == rep_vec).all(),                                  \
        f"RepOpenMM.get_rep_vec: incorrect coor-rep_vec binding: __init__ "    \
        f"rep_vec: {rep_vec}; RepOpenMM.get_rep_vec: {rep_vec_return}"

    # RepOpenMM.set_coor()
    coor_val = (numpy.random.random_sample((n_atoms, 3)) - 0.5) * 3
    coor_new = OmmQuantity(value=coor_val, unit=unit.nanometers)
    rep_vec_new = (coor_val*10).flatten()
    rep.set_coor(coor_new)

    coor_new_return = rep.get_coor()
    assert (coor_new_return == coor_new).all(),                                \
        f"RepOpenMM.set_coor: incorrect coor-rep_vec binding: set_coor "       \
        f"coor: {coor_new}; RepOpenMM.get_coor: {coor_new_return}"

    rep_vec_new_return = rep.get_rep_vec() # unit.Angstrom
    assert (rep_vec_new_return == rep_vec_new).all(),                          \
        f"RepOpenMM.set_coor: incorrect coor-rep_vec binding: set_coor "       \
        f"rep_vec: {rep_vec_new}; RepOpenMM.get_rep_vec: {rep_vec_new_return}"

    # RepOpenMM.set_rep_vec()
    rep_vec_new = (numpy.random.random_sample(n_atoms*3) - 0.5) * 3
    coor_new = OmmQuantity(value=rep_vec_new.reshape(-1, 3), unit=unit.angstrom)
    rep.set_rep_vec(rep_vec_new)

    coor_new_return = rep.get_coor()
    assert (coor_new_return == coor_new).all(),                                \
        f"RepOpenMM.set_rep_vec: incorrect coor-rep_vec binding: set_rep_vec " \
        f"coor: {coor_new}; RepOpenMM.get_coor: {coor_new_return}"

    rep_vec_new_return = rep.get_rep_vec()
    assert (rep_vec_new_return == rep_vec_new).all(),                          \
        f"RepOpenMM.set_rep_vec: incorrect coor-rep_vec binding: set_rep_vec " \
        f"rep_vec: {rep_vec_new}; RepOpenMM.get_rep_vec: {rep_vec_new_return}"

    """Test if mass_weighting and scaling are handled correctly."""
    omm_sys: OmmSystem = omm_cnt.getSystem()
    masses = [omm_sys.getParticleMass(i).value_in_unit(unit.atomic_mass_unit)  \
              for i in range(n_atoms)]
    masses = numpy.asarray(masses)
    wscale = numpy.random.random_sample(n_atoms)*10
    
    # For the following three cases, 
    # test RepOpenMM.is_mass_weighted() and RepOpenMM.get_dof_weighting_vec()

    # scaling on weight. 
    rep = RepOpenMM(coor, pot, is_mass_weighted=False, mscale_vec=wscale)
    assert rep.is_mass_weighted() == False,                                    \
        f"RepOpenMM.is_mass_weighted: incorrect is_mass_weighted indicator. "  \
        f"Truth: False; rep.is_mass_weighted() {rep.is_mass_weighted()}"

    rep_wscale_truth = numpy.repeat(wscale, 3)
    rep_wscale = rep.get_wscale_vec()
    assert (rep_wscale == rep_wscale_truth).all(),                             \
        f"RepOpenMM.get_dof_weighting_vec: incorrect dof weighting vector. "   \
        f"Applied weight-scaling only. Truth: {rep_wscale_truth}; "            \
        f"rep.get_dof_weighting_vec() {rep_wscale}"
    
    # scaling on mass.
    rep = RepOpenMM(coor, pot, is_mass_weighted=True, mscale_vec=None)
    assert rep.is_mass_weighted() == True,                                     \
        f"RepOpenMM.is_mass_weighted: incorrect is_mass_weighted indicator. "  \
        f"Truth: False; rep.is_mass_weighted() {rep.is_mass_weighted()}"

    rep_wscale_truth = numpy.repeat(masses, 3)
    rep_wscale = rep.get_wscale_vec()
    assert (rep_wscale == rep_wscale_truth).all(),                             \
        f"RepOpenMM.get_dof_weighting_vec: incorrect dof weighting vector. "   \
        f"Applied mass-weighting only. Truth: {rep_wscale_truth}; "            \
        f"rep.get_dof_weighting_vec() {rep_wscale}"

    # scaling on mass and weighting.
    rep = RepOpenMM(coor, pot, is_mass_weighted=True, mscale_vec=wscale)
    assert rep.is_mass_weighted() == True,                                     \
        f"RepOpenMM.is_mass_weighted: incorrect is_mass_weighted indicator. "  \
        f"Truth: False; rep.is_mass_weighted() {rep.is_mass_weighted()}"

    rep_wscale_truth = numpy.repeat(masses * wscale, 3)
    rep_wscale = rep.get_wscale_vec()
    assert (rep_wscale == rep_wscale_truth).all(),                             \
        f"RepOpenMM.get_dof_weighting_vec: incorrect dof weighting vector. "   \
        f"Applied mass-weighting & weight-scaling. Truth: {rep_wscale_truth}; "\
        f"rep.get_dof_weighting_vec() {rep_wscale}"
    
def test_ener_grad():
    """Test if energy and gradients are correct."""

    # Set reference OpenMM objects.
    omm_cnt = _get_omm_context()
    
    cor = CharmmCrdFile("./examples/omm_c36m_diala/diala.cor")
    omm_cnt.setPositions(cor.positions)

    # True ener and grad.
    omm_stt: OmmState = omm_cnt.getState(getEnergy=True, getForces=True)
    omm_ener: OmmQuantity = omm_stt.getPotentialEnergy()
    omm_ener = omm_ener.value_in_unit(unit.kilocalorie_per_mole)
    omm_grad: OmmQuantity = -1. * omm_stt.getForces()
    omm_grad = omm_grad.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    omm_grad = numpy.asarray(omm_grad).flatten()

    # Set RepOpenMM
    omm_cnt = _get_omm_context()
    pot = PotOpenMM(omm_cnt)
    rep = RepOpenMM(cor.positions, pot)

    # Energy and gradient from get_ener_grad() should be the same with OpenMM.
    rep_ener, rep_grad = rep.get_ener_grad()

    assert rep_ener == omm_ener,                                               \
        f"RepOpenMM.get_ener_grad: incorrect energy value. "                   \
        f"Truth: {omm_ener} kcal/mol; rep.get_ener_grad() {rep_ener}"

    assert numpy.allclose(rep_grad, omm_grad, atol=1e-4),                      \
        f"RepOpenMM.get_ener_grad: incorrect gradient value. "                 \
        f"Truth: {omm_grad}; rep.get_ener_grad() {rep_grad}"

    """Change coordinates and see if the ener and grad are correct.
    """
    
    # Set reference OpenMM objects.
    omm_cnt = _get_omm_context()
    pdb = PDBFile("./examples/omm_c36m_diala/c36m_c_7eq.pdb")
    omm_cnt.setPositions(pdb.positions)
    # True ener and grad.
    omm_stt: OmmState = omm_cnt.getState(getEnergy=True, getForces=True)
    omm_ener: OmmQuantity = omm_stt.getPotentialEnergy()
    omm_ener = omm_ener.value_in_unit(unit.kilocalorie_per_mole)
    omm_grad: OmmQuantity = -1. * omm_stt.getForces()
    omm_grad = omm_grad.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom)
    omm_grad = numpy.asarray(omm_grad).flatten()

    # Change cooredinates
    rep.set_coor(pdb.positions)
    rep_ener, rep_grad = rep.get_ener_grad()

    assert rep_ener == omm_ener,                                               \
        f"RepOpenMM.get_ener_grad: incorrect energy value after coor change. " \
        f"Truth: {omm_ener} kcal/mol; rep.get_ener_grad() {rep_ener}"

    assert numpy.allclose(rep_grad, omm_grad, atol=1e-4),                      \
        f"RepOpenMM.get_ener_grad: incorrect gradient value after coor change."\
        f" Truth: {omm_grad}; rep.get_ener_grad() {rep_grad}"
