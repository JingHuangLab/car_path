# Unit test for OpenMM potentials.
# Zilin Song, 18 Jun 2022

import pytest
from pycospath.comms.openmm import PotOpenMM

from openmm import VerletIntegrator, unit, Context as OmmContext,              \
                   System as OmmSystem, State as OmmState
from openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet,       \
                       NoCutoff, Simulation as OmmSimulation
from openmm.unit import Quantity as OmmQuantity

def _get_omm_context() -> OmmContext:
    # Data files.
    ffs = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm")
    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")

    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_int = VerletIntegrator(1.*unit.picosecond) # requires a new integrator.
    omm_cnt = OmmContext(omm_sys, omm_int)
    return omm_cnt

def _get_omm_simulation() -> OmmSimulation:
    # Data files.
    ffs = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm")
    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")

    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_int = VerletIntegrator(1.*unit.picosecond)
    omm_sim = OmmSimulation(psf.topology, omm_sys, omm_int)
    return omm_sim

def test_constructor():
    """Test if __init__ works correctly."""

    omm_cnt = _get_omm_context()
    omm_sys: OmmSystem = omm_cnt.getSystem()
    
    with pytest.raises(TypeError,                                              \
                       match="omm_context must be openmm.Context type."):
        PotOpenMM(omm_sys)

    omm_sim: OmmSimulation = _get_omm_simulation()

    with pytest.raises(TypeError,                                              \
                       match="omm_context must be openmm.Context type."):
        PotOpenMM(omm_sim)

    # Correct object type passed.
    omm_cnt = _get_omm_context()

    try:
        PotOpenMM(omm_cnt)
    except Exception as e:
        assert False,                                                          \
            "openmm.Context object passed to PotOpenMM constructor. Should not"\
            f"raise an error {e}"

def test_members():
    """Test various class members in PotOpenMM."""
    omm_cnt = _get_omm_context()
    pot_omm = PotOpenMM(omm_cnt)

    # PotOpenMM.get_omm_context()
    returned_cnt = pot_omm.get_omm_context()

    assert isinstance(returned_cnt, OmmContext),                               \
        "Incorrect return type by PotOpenMM.get_omm_context(). Expected type: "\
        f"openmm.openmm.Context; Returned type {type(returned_cnt)}"

    # PotOpenMM.get_omm_context()
    returned_sys = pot_omm.get_omm_system()

    assert isinstance(returned_sys, OmmSystem),                                \
        "Incorrect return type by PotOpenMM.get_omm_system(). Expected type: " \
        f"openmm.openmm.System; Returned type {type(returned_sys)}"

def test_ener_grad():
    """Test if energy and gradients are correct."""

    # Set reference OpenMM objects. 
    omm_cnt = _get_omm_context()
    cor = CharmmCrdFile("./examples/omm_c36m_diala/diala.cor")
    omm_cnt.setPositions(cor.positions)

    omm_stt: OmmState = omm_cnt.getState(getEnergy=True, getForces=True)
    omm_ener: OmmQuantity = omm_stt.getPotentialEnergy()
    omm_grad: OmmQuantity = -1. * omm_stt.getForces()

    # Set PotOpenMM
    pot_omm = PotOpenMM(omm_cnt)

    # Energy from get_ener() should be the same with OpenMM.
    pot_ener = pot_omm.get_ener(cor.positions)

    assert pot_ener == omm_ener,                                               \
        f"Inconsistent energy output from PotOpenMM.get_ener().\n"             \
        f"pot_omm: {pot_ener}; omm_sim: {omm_ener}"

    # Energy and gradient from get_grad() should be the same with OpenMM.
    pot_ener, pot_grad = pot_omm.get_ener_grad(cor.positions)

    assert pot_ener == omm_ener,                                               \
        f"Inconsistent energy output from PotOpenMM.get_grad().\n"             \
        f"pot_omm: {pot_ener}; omm_sim: {omm_ener}"

    diff_grad = [pot_grad[i] - omm_grad[i] for i in range(len(omm_grad))]

    import numpy
    assert numpy.sum(numpy.abs(numpy.asarray(diff_grad))
            ).value_in_unit(unit.kilocalorie_per_mole/unit.angstrom) <= 1e-4,  \
        f"Inconsistent gradient output from PotOpenMM.get_grad().\n"           \
        f"pot_omm: {pot_grad}; omm_sim: {omm_grad}"

    # Energies from get_ener() and get_grad()[0] should be the same.
    pot_e_ener = pot_omm.get_ener(cor.positions)
    pot_e_grad = pot_omm.get_ener_grad(cor.positions)[0]
    assert pot_e_ener == pot_e_grad,                                           \
        f"PotOpenMM: energies from get_ener() and get_grad should be the same."\
        f"get_ener: {pot_e_ener}; get_grad: {pot_e_grad}."
