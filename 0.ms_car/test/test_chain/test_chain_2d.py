# Unit test for Chain2D objects.
# Zilin Song, 22 Jun 2022

import pytest

import numpy
from random import randint
from pycospath.chain import Chain2D

from pycospath.comms.twodim import PotMuller, PotAsymDoubleWell, Rep2D
from pycospath.comms.openmm import PotOpenMM, RepOpenMM

from openmm import VerletIntegrator, unit, Context as OmmContext,              \
                   System as OmmSystem
from openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet,       \
                       NoCutoff

def _get_omm_rep():
    # Make a bad replica.
    ffs = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm")
    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")
    cor = CharmmCrdFile("./examples/omm_c36m_diala/diala.cor")
    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_cnt = OmmContext(omm_sys, VerletIntegrator(1.*unit.picosecond))
    omm_pot = PotOpenMM(omm_cnt)
    omm_rep = RepOpenMM(cor.positions, omm_pot)
    return omm_rep

def _get_2d_coor():
    return (numpy.random.random_sample(2)-0.5)*5

def test_constructor():
    """Test if __init__ works correctly."""

    # test_all_good_rep()
    # Make list of good reps.
    rep_list = []

    for _ in range(randint(4,10)):
        rep = Rep2D(_get_2d_coor(), PotAsymDoubleWell())
        rep_list.append(rep)

    try:
        Chain2D(rep_list)
    except Exception as e:
        assert False, f"Chain2D({rep_list}) should not raise {e}."

    # test_one_bad_rep()
    rep_list.append(_get_omm_rep())

    with pytest.raises(TypeError,
                       match="Chain2D must be initialized with List\[Rep2D\]."):
        Chain2D(rep_list)

def test_members():
    """Test various class members in Chain2D."""

    # Make Chain2D.
    n_reps_true = randint(4, 10)
    rep_list_true = []
    coors_true = numpy.zeros((n_reps_true, 2))
    
    for _ in range(n_reps_true):
        coors_true[_, :] = _get_2d_coor()
        rep = Rep2D(coors_true[_, :], PotAsymDoubleWell())
        rep_list_true.append(rep)

    chain = Chain2D(rep_list_true)

    # Chain2D.get_rep_list()
    rep_list_return = chain.get_rep_list()
    assert rep_list_return == rep_list_true,                                   \
        f"Chain2D.get_rep_list: incorrect replica list object."

    # Chain2D.get_num_reps()
    n_reps_return = chain.get_num_reps()
    assert n_reps_return == n_reps_true,                                       \
        f"Chain2D.get_num_reps: incorrect n_reps, "                            \
        f"Truth: {n_reps_true}, Chain2D.get_num_reps: {n_reps_return}"

    # Chain2D.get_num_dofs()
    ndofs_return = chain.get_num_dofs()
    assert ndofs_return == 2,                                                  \
        f"Chain2D.get_num_dofs: incorrect ndofs, "                             \
        f"Truth: 2 Chain2D.get_num_dofs: {ndofs_return}"

    # Chain2D.get_rep_vec_weighting_vec()
    with pytest.raises(NotImplementedError, 
                       match="No weight-scaling in Chain2D objects."):
        chain.get_wscale_vec()
        
    # Chain2D.get_chain_vec()
    chain_vec_return = chain.get_chain_vec()
    assert (chain_vec_return == coors_true).all(),                             \
        f"Chain2D.get_chain_vec: incorrect get_chain_vec, "                    \
        f"Truth: {coors_true}; Chain2D.get_rep_vec: {chain_vec_return}"

    # Chain2D.set_chain_vec()
    chain_vec_new = (numpy.random.random_sample((n_reps_true, 2))-0.5) * 9
    chain.set_chain_vec(chain_vec_new)

    chain_vec_new_return = chain.get_chain_vec()
    assert (chain_vec_new_return == chain_vec_new).all(),                      \
        f"Chain2D.set_chain_vec: incorrect chain_vec update, "                 \
        f"Truth: {chain_vec_new}; "                                            \
        f"Chain2D.set_chain_vec: {chain_vec_new_return}"
    
    # Chain2D.set_chain_vec(): check rep_vec update.
    rep_list_true = chain.get_rep_list()
    rep_vecs_new_return = [rep_list_true[i].get_rep_vec()                      \
                           for i in range(n_reps_true)]

    assert (numpy.asarray(rep_vecs_new_return) == chain_vec_new).all(),        \
        f"Chain2D.set_chain_vec: incorrect rep_vec binding update, "           \
        f"Truth: {chain_vec_new}; "                                            \
        f"Chain2D.set_chain_vec.rep_vecs: {rep_vecs_new_return}"
    
def test_ener_grad():
    """Test if the energy and gradients are correct."""

    n_reps = randint(4, 10)
    coors = numpy.zeros((n_reps, 2))
    rep_list = []

    for _ in range(n_reps):
        coors[_, :] = _get_2d_coor()
        rep = Rep2D(coors[_, :], PotMuller())
        rep_list.append(rep)

    chain = Chain2D(rep_list)

    # determine the true energies and gradients. 
    e_true = numpy.zeros((n_reps,))
    g_true = numpy.zeros((n_reps, 2))

    for _ in range(n_reps):
        rep: Rep2D= rep_list[_]
        e_true[_], g_true[_, :] = rep.get_ener_grad()
    
    # chain energies and gradients.
    e_chain, g_chain = chain.get_eners_grads()

    assert (e_true == e_chain).all(),                                          \
        f"Chain2D.get_pot_grads: incorrect energies. "                         \
        f"Truth: {e_true}; Chain2D.get_pot_grads()[0]: {e_chain}"
    
    assert (g_true == g_chain).all(),                                          \
        f"Chain2D.get_pot_grads: incorrect gradients. "                        \
        f"Truth: {g_true}; Chain2D.get_pot_grads()[1]: {g_chain}"
    
    # determine the true energies and gradients. 
    e_true = numpy.zeros((n_reps,))
    g_true = numpy.zeros((n_reps, 2))

    for _ in range(n_reps):
        coors[_, :] = _get_2d_coor()

        rep: Rep2D= rep_list[_]
        rep.set_coor(coors[_, :])
        e_true[_], g_true[_, :] = rep.get_ener_grad()

    # chain energies and gradients.
    chain.set_chain_vec(coors)
    e_chain, g_chain = chain.get_eners_grads()

    assert (e_true == e_chain).all(),                                          \
        f"Chain2D.get_pot_grads: incorrect energies after set_chain_vec(). "   \
        f"Truth: {e_true}; Chain2D.get_pot_grads()[0]: {e_chain}"
    
    assert (g_true == g_chain).all(),                                          \
        f"Chain2D.get_pot_grads: incorrect gradients after set_chain_vec(). "  \
        f"Truth: {g_true}; Chain2D.get_pot_grads()[0]: {g_chain}"

def test_rms():
    """Test if the RMS distances and gradients are correct."""
    # Init 2D coors.
    n_reps = randint(4, 10)
    coors = numpy.zeros((n_reps, 2))

    for _ in range(n_reps):
        coors[_, :] = _get_2d_coor()

    # Calculate true rms and rms_grad.
    coor_diff = (coors[0:-1, :] - coors[1:  , :])
    rms_true = numpy.sum(coor_diff**2, axis=1) ** .5
    rms_grad_true = coor_diff / rms_true[:, None]

    # Make Chain2D.
    rep_list = []

    for _ in range(n_reps):
        rep = Rep2D(coors[_, :], PotMuller())
        rep_list.append(rep)
    
    chain = Chain2D(rep_list)
    chain_vec = chain.get_chain_vec()

    rms_chain = chain.get_rms(chain_vec)

    assert (rms_chain == rms_true).all(),                                      \
        f"Chain2D.get_rms(get_grads=False): Incorrect RMS values."             \
        f"Truth: {rms_true}; Chain2D.get_rms(): {rms_chain}"
    
    rms_chain, rms_grad_chain = chain.get_rms(chain_vec, return_grads=True)

    assert (rms_chain == rms_true).all(),                                      \
        f"Chain2D.get_rms(): Incorrect RMS values. "                           \
        f"Truth: {rms_true}; Chain2D.get_rms()[0]: {rms_chain}"

    assert (rms_grad_chain == rms_grad_true).all(),                            \
        f"Chain2D.get_rms(get_grads=True): Incorrect RMS gradients."           \
        f"Truth: {rms_grad_true}; Chain2D.get_rms()[1]: {rms_grad_chain}"
