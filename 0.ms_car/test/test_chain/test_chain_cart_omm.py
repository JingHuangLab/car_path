# Unit test for Chain2D objects.
# Zilin Song, 22 Jun 2022

from typing import List
import pytest

import numpy
from random import randint
from pycospath.chain import ChainCart

from pycospath.comms.twodim import PotSymDoubleWell, Rep2D
from pycospath.comms.openmm import PotOpenMM, RepOpenMM
from pycospath.utils        import rigid_fit

from openmm import VerletIntegrator, unit, Context as OmmContext,              \
                   System as OmmSystem
from openmm.unit import Quantity as OmmQuantity
from openmm.app import CharmmPsfFile, PDBFile, CharmmParameterSet, NoCutoff

def _get_omm_rep_list(n_reps, 
                      is_mass_weighted=False, 
                      mass_scaling_vec=None
                      ) -> List[RepOpenMM]:

    ffs = CharmmParameterSet(
                        "./examples/toppar/top_all36_prot.rtf",
                        "./examples/toppar/par_all36m_prot.prm")
    psf = CharmmPsfFile("./examples/omm_c36m_diala/diala.psf")

    omm_sys: OmmSystem = psf.createSystem(ffs, nonbondedMethod=NoCutoff)
    omm_cnt = OmmContext(omm_sys, VerletIntegrator(1.*unit.picosecond))
    omm_pot = PotOpenMM(omm_cnt)

    # Init an initial series of coordinates.

    rep_list = []

    pdb_ref = PDBFile(f"./examples/omm_c36m_diala/init_guess/r0.pdb")
    rep_vec_ref = numpy.asarray(pdb_ref.positions.value_in_unit(unit.angstrom)).flatten()

    for i_rep in range(n_reps):
        pdb = PDBFile(f"./examples/omm_c36m_diala/init_guess/r{i_rep}.pdb")
        rep_vec= numpy.asarray(pdb.positions.value_in_unit(unit.angstrom)).flatten()
        rep_coor_align, _, _ = rigid_fit(rep_vec_ref, rep_vec)
        rep_coor = OmmQuantity(rep_coor_align.reshape(-1, 3), unit=unit.angstrom)
        rep_list.append(RepOpenMM(rep_coor, omm_pot, 
                                  is_mass_weighted=is_mass_weighted, 
                                  mscale_vec=mass_scaling_vec))
    
    return rep_list

def test_constructor():
    """Test if __init__ works correctly."""

    # test_all_good_rep()
    n_reps = randint(5, 12) 
    rep_list = _get_omm_rep_list(n_reps)

    try:
        ChainCart(rep_list, is_rigid_fit=False)
    except Exception as e:
        assert False, f"ChainCart({rep_list}) should not raise {e}."

    # test_one_bad_rep()
    # Make a bad rep
    bad_rep = Rep2D((numpy.random.random_sample(2)-0.5)*5, PotSymDoubleWell())
    rep_list.append(bad_rep)

    with pytest.raises(TypeError, 
                       match="ChainCart must be initialized with "
                             "List\[RepCartBase\]."):
        ChainCart(rep_list, is_rigid_fit=False)
    
    # test_init_best_fit()
    n_reps = randint(5, 12) 
    rep_list = _get_omm_rep_list(n_reps)

    chain = ChainCart(rep_list, is_rigid_fit=True)

    rvecs = numpy.asarray([rep_list[i].get_rep_vec() for i in range(n_reps)])

    for i_rep in range(1, n_reps):
        # See if best-fit is performed correctly. 
        rvecs[i_rep, :], _, _ = rigid_fit(rvecs[i_rep-1, :], rvecs[i_rep, :])
        true_rvec = rvecs[i_rep, :]
        
        # Chain vectors (best-fitted)
        return_rvec = chain.get_chain_vec()[i_rep, :]

        diff = numpy.sum(numpy.abs(true_rvec - return_rvec))

        assert diff <= 1e-8,                                                   \
            f"Imcompatible best-fitted chain vector to raw replica vectors at "\
            f"the {i_rep}-th replica. True chain vec: {true_rvec}, "           \
            f"chain.get_chain_vec()[i_rep, :]: {return_rvec}. "                \
            f"Deviation: {diff}"

def test_members():
    """Test various members in ChainCart."""
    n_reps = randint(5, 12)
    rep_list = _get_omm_rep_list(n_reps)
    n_dofs = rep_list[1].get_num_dofs()
    
    # ChainCart.get_rep_list()
    chain = ChainCart(rep_list, is_rigid_fit=True)
    rep_list_return = chain.get_rep_list()
    assert rep_list_return == rep_list,                                        \
        f"ChainCart.get_rep_list: incorrect replica list object."

    # ChainCart.is_best_fit()
    best_fit_true = True if randint(0,1) == 0 else False
    chain = ChainCart(rep_list, is_rigid_fit=best_fit_true)
    best_fit_return = chain.is_rigid_fit()
    assert chain.is_rigid_fit() == best_fit_true,                               \
        f"ChainCart.is_best_fit: incorrect best-fit state, "                   \
        f"Truth: {best_fit_true}, ChainCart.is_best_fit: {best_fit_return}"

    # ChainCart.get_num_reps()
    chain = ChainCart(rep_list, is_rigid_fit=True)
    n_reps_return = chain.get_num_reps()
    assert n_reps_return == n_reps,                                            \
        f"ChainCart.get_num_reps: incorrect n_reps, "                          \
        f"Truth: {n_reps}, ChainCart.get_num_reps: {n_reps_return}"

    # ChainCart.get_num_dofs()
    n_dofs_return = chain.get_num_dofs()
    assert n_dofs_return == n_dofs,                                            \
        f"ChainCart.get_num_dofs: incorrect ndofs, "                           \
        f"Truth: {n_dofs} ChainCart.get_num_dofs: {n_dofs_return}"
        
    # ChainCart.get_rep_vec_weighting_vec()
    rep_vec_weighting_vec_return = chain.get_wscale_vec()
    rep_vec_weighting_vec_true = rep_list[0].get_wscale_vec()
    assert (rep_vec_weighting_vec_return == rep_vec_weighting_vec_true).all(), \
        f"Chain2D.n_rep_vec_weighting: incorrect rep_vec weighting vec, "      \
        f"Truth: {rep_vec_weighting_vec_true}; "                               \
        f"Chain2D.get_rep_vec_weighting_vec: {rep_vec_weighting_vec_return}"

    # ChainCart.get_chain_vec()

    # Test chain_vec without best-fit.
    # True chain_vec values w/o best-fit
    chain_vec_true = numpy.zeros((n_reps, n_dofs))
    for i_rep in range(n_reps):
        chain_vec_true[i_rep, :] = rep_list[i_rep].get_rep_vec()

    chain = ChainCart(rep_list, is_rigid_fit=False)
    chain_vec_return = chain.get_chain_vec()

    assert (chain_vec_return == chain_vec_true).all(),                         \
        f"ChainCart.get_chain_vec: incorrect non-best-fit chain_vecs, "        \
        f"Truth: {chain_vec_true}; ChainCart.get_chain_vec: {chain_vec_return}"

    # Test chain_vec with best-fit.
    # True chain_vec values w/ best-fit: chain_vec_raw
    for i_rep in range(1, n_reps):
        chain_vec_true[i_rep, :], _, _ = rigid_fit(chain_vec_true[i_rep-1, :], 
                                                   chain_vec_true[i_rep,   :])
    
    chain = ChainCart(rep_list, is_rigid_fit=True)
    chain_vec_return = chain.get_chain_vec()
    rep_vec_diff = numpy.sum(numpy.abs(chain_vec_return - chain_vec_true))

    assert rep_vec_diff <= 1e-8,                                               \
        f"ChainCart.get_chain_vec: incorrect best-fit chain_vecs, "            \
        f"Truth: {chain_vec_true}; "                                           \
        f"ChainCart.get_chain_vec: {chain_vec_return}, "                       \
        f"Deviation: {chain_vec_true -chain_vec_return}"

    # ChainCart.set_chain_vec()

    # Test ChainCart w/o best-fit.
    chain = ChainCart(rep_list, is_rigid_fit=False)

    # ChainCart.set_chain_vec: check chain_vec update.
    chain_vec_new = (numpy.random.random_sample((n_reps, n_dofs))-.5) * 15
    chain.set_chain_vec(chain_vec_new)
    chain_vec_new_return = chain.get_chain_vec()

    assert (chain_vec_new_return == chain_vec_new).all(),                      \
        f"ChainCart.set_chain_vec: inccorect chain_vec update w/o best fit, "  \
        f"Truth: {chain_vec_new}, "                                            \
        f"ChainCart.set_chain_vec: {chain_vec_new_return}"
    
    # ChainCart.set_chain_vec: check rep_vec update.
    rep_list_new_return = chain.get_rep_list()
    rep_vecs_new_return = [rep_list_new_return[i].get_rep_vec()                \
                           for i in range(n_reps)]
    assert (numpy.asarray(rep_vecs_new_return) == chain_vec_new).all(),        \
        f"ChainCart.set_chain_vec: incorrect rep_vec binding w/o best fit, "   \
        f"Truth: {chain_vec_new}; "                                            \
        f"ChainCart.set_chain_vec.rep_vecs: {rep_vecs_new_return}"
    
    # Test ChainCart w/ best-fit.
    chain = ChainCart(rep_list_new_return, is_rigid_fit=True)

    # ChainCart.set_chain_vec: check chain_vec update. 
    # Ground truth best fitted chain_vec.
    chain_vec_temp = (numpy.random.random_sample((n_reps, n_dofs))-.5) * 15
    # Note that each row of chain_vec is best-fitted onto the its preceding row. 
    chain_vec_new = numpy.copy(chain_vec_temp)

    # NOTE: should chain vector be rigid-fitted before set_chain_vec?
    # for i_rep in range(1, chain_vec_temp.shape[0]):
    #     chain_vec_new[i_rep, :], _, _ = rigid_fit(chain_vec_new[0,     :], 
    #                                               chain_vec_new[i_rep, :])

    chain.set_chain_vec(chain_vec_temp)
    chain_vec_new_return = chain.get_chain_vec()

    assert (chain_vec_new_return == chain_vec_new).all(),                      \
        f"ChainCart.set_chain_vec: inccorect chain_vec update w/ best fit, "   \
        f"Truth: {chain_vec_new}, "                                            \
        f"ChainCart.set_chain_vec: {chain_vec_new_return}"
    
    # ChainCart.set_chain_vec: check rep_vec update.
    # Note that each row of chain_vec is best-fitted onto the a prior rep_vec. 
    rep_vecs_new_true = numpy.zeros((n_reps, n_dofs))
    for i_rep in range(n_reps):
        rep = rep_list[i_rep]
        rep_vecs_new_true[i_rep, :], _, _ = rigid_fit(rep.get_rep_vec(),
                                                      chain_vec_new[i_rep, :])

    rep_list_new_return = chain.get_rep_list()
    rep_vecs_new_return = [rep_list_new_return[i].get_rep_vec()                \
                           for i in range(n_reps)]

    rep_vec_diff = numpy.sum(numpy.abs(rep_vecs_new_return - rep_vecs_new_true))
    assert rep_vec_diff <= 1e-8,                                               \
        f"ChainCart.set_chain_vec: incorrect rep_vec binding w/ best fit, "    \
        f"Truth: {rep_vecs_new_true}; "                                        \
        f"ChainCart.set_chain_vec.rep_vecs: {rep_list_new_return}; "           \
        f"Deviation: {rep_vec_diff}"

def test_pot_grad():
    """Test if the energy and gradients are correct."""
    
    # Init an initial series of coordinates.
    n_reps = randint(5, 12)   # No. reps. 
    rep_list = _get_omm_rep_list(n_reps)
    n_dofs = rep_list[2].get_num_dofs()

    # Test the energies and gradients w/o best-fit.
    chain = ChainCart(rep_list, is_rigid_fit=False)

    # determine the true energies and gradients. 
    e_true = numpy.zeros((n_reps, ))
    g_true = numpy.zeros((n_reps, n_dofs))

    for i_rep in range(n_reps):
        e_true[i_rep], g_true[i_rep, :] = rep_list[i_rep].get_ener_grad()

    # chain energies and gradients. 
    e_chain, g_chain = chain.get_eners_grads()

    assert numpy.allclose(e_chain, e_true, atol=1e-4),                         \
        f"ChainCart.get_pot_grads: incorrect energies w/o best-fit. "          \
        f"Truth: {e_true}; ChainCart.get_pot_grads()[0]: {e_chain}"
    
    assert numpy.allclose(g_chain, g_true, atol=1e-4),                         \
        f"ChainCart.get_pot_grads: incorrect gradients w/o best-fit. "         \
        f"Truth: {g_true[1]}; ChainCart.get_pot_grads()[1]: {g_chain[1]}"

    # Test the energies and gradients w/ best-fit. 

    # Init chain and determine the true energies and gradients.
    chain = ChainCart(rep_list, is_rigid_fit=True)

    e_true = numpy.zeros((n_reps, ))
    g_true = numpy.zeros((n_reps, n_dofs))

    for i_rep in range(n_reps):
        e_true[i_rep], g_rep = rep_list[i_rep].get_ener_grad()

        # Gradients on rep_vec is rotated to be gradients on chain_vec[i_rep]. 
        ref_rep_vec = chain.get_chain_vec()[i_rep, :]
        _, ro, _ = rigid_fit(ref_rep_vec, rep_list[i_rep].get_rep_vec())
        g_true[i_rep, :] = (ro @ g_rep.reshape(-1, 3).T).T.flatten()

    # update chain_vec
    e_chain, g_chain = chain.get_eners_grads()
    
    e_diff = numpy.sum(numpy.abs(e_chain - e_true))
    g_diff = numpy.sum(numpy.abs(g_chain - g_true))

    assert numpy.allclose(e_chain, e_true, atol=1e-4),                         \
        f"ChainCart.get_pot_grads: incorrect energies w/ best-fit. "           \
        f"Truth: {e_true}; ChainCart.get_pot_grads()[0]: {e_chain}"
    
    assert numpy.allclose(g_chain, g_true, atol=1e-4),                         \
        f"ChainCart.get_pot_grads: incorrect gradients w/ best-fit. "          \
        f"Truth: {g_true}; ChainCart.get_pot_grads()[1]: {g_chain}; "          \
        f"Deviation {g_diff}"

def test_rms():
    """Test if the RMS distances and gradients are correct."""
    ####################################
    # No mass-weighted, no best-fitted #
    ####################################

    # Init chain and chain_vec.
    n_reps = randint(5, 12)
    rep_list = _get_omm_rep_list(n_reps)
    n_dofs = rep_list[3].get_num_dofs()
    
    chain = ChainCart(rep_list, is_rigid_fit=False)
    chain_vec = chain.get_chain_vec()

    # Compute the true values.
    r_diff = chain_vec[0:-1, :] - chain_vec[1:  , :]
    rms_true = (numpy.sum(r_diff ** 2., axis=1) / n_dofs * 3) ** .5
    rms_grad_true = r_diff / rms_true[:, None] / n_dofs * 3

    rms_chain, rms_grad_chain = chain.get_rms(chain_vec, return_grads=True)

    rms_diff = numpy.sum(numpy.abs(rms_chain - rms_true))
    assert rms_diff <= 1e-8,                                                   \
        f"Mode: no best-fit, no weight-scaling. "                              \
        f"ChainCart.get_rms(): Incorrect RMS. "                                \
        f"Truth: {rms_true}; ChainCart.get_rms(): {rms_chain}"
    
    rms_grad_diff = numpy.sum(numpy.abs(rms_grad_chain - rms_grad_true))
    assert rms_grad_diff <= 1e-8,                                              \
        f"Mode: no best-fit, no weight-scaling. "                              \
        f"ChainCart.get_rms(get_grads=True): Incorrect RMS gradients. "        \
        f"Truth: {rms_grad_true}; ChainCart.get_rms()[1]: {rms_grad_chain}"

    ##################################
    # mass-weighted, no best-fitted. #
    ##################################

    # Init chain and chain_vec. 
    rep_list = _get_omm_rep_list(n_reps, 
                                 is_mass_weighted=True
                                 )
        
    chain = ChainCart(rep_list, is_rigid_fit=False)
    chain_vec = chain.get_chain_vec()

    # Compute the true values.
    r_diff = chain_vec[0:-1, :] - chain_vec[1:  , :]
    w = chain.get_wscale_vec()
    rms_true = (numpy.sum(r_diff**2 * w[None, :], axis=1) / numpy.sum(w)*3)**.5
    rms_grad_true = r_diff * w[None, :] / numpy.sum(w) * 3 / rms_true[:, None]

    rms_chain, rms_grad_chain = chain.get_rms(chain_vec, return_grads=True)

    rms_diff = numpy.sum(numpy.abs(rms_chain - rms_true))
    assert rms_diff <= 1e-8,                                                   \
        f"Mode: no best-fit, with mass-weighting. "                            \
        f"ChainCart.get_rms(): Incorrect RMS. "                                \
        f"Truth: {rms_true}; "                                                 \
        f"ChainCart.get_rms(): {rms_chain}" 
    
    rms_grad_diff = numpy.sum(numpy.abs(rms_grad_chain - rms_grad_true))
    assert rms_grad_diff <= 1e-8,                                              \
        f"Mode: no best-fit, with mass-weighting. "                            \
        f"ChainCart.get_rms(get_grads=True): Incorrect RMS gradients. "        \
        f"Truth: {rms_grad_true}; "                                            \
        f"ChainCart.get_rms()[1]: {rms_grad_chain}"

    ##################################
    # weight-scaled, no best-fitted. #
    ##################################
    
    # Init chain and chain_vec. 
    rep_list = _get_omm_rep_list(n_reps, 
                                 is_mass_weighted=True, 
                                 mass_scaling_vec=numpy.arange(n_dofs / 3)
                                 )

    chain = ChainCart(rep_list, is_rigid_fit=False)

    chain_vec = chain.get_chain_vec()

    # Compute the true values.
    r_diff = chain_vec[0:-1, :] - chain_vec[1:  , :]
    w = chain.get_wscale_vec()
    rms_true = (numpy.sum(r_diff**2 * w[None, :], axis=1) / numpy.sum(w)*3)**.5
    rms_grad_true = r_diff * w[None, :] / numpy.sum(w) * 3 / rms_true[:, None]

    rms_chain, rms_grad_chain = chain.get_rms(chain_vec, return_grads=True)

    rms_diff = numpy.sum(numpy.abs(rms_chain - rms_true))
    assert rms_diff <= 1e-8,                                                   \
        f"Mode: no best-fit, with weight-scaling. "                            \
        f"ChainCart.get_rms(): Incorrect RMS. "                                \
        f"Truth: {rms_true}; "                                                 \
        f"ChainCart.get_rms(): {rms_chain}" 
    
    rms_grad_diff = numpy.sum(numpy.abs(rms_grad_chain - rms_grad_true))
    assert rms_grad_diff <= 1e-8,                                              \
        f"Mode: no best-fit, with weight-scaling. "                            \
        f"ChainCart.get_rms(get_grads=True): Incorrect RMS gradients. "        \
        f"Truth: {rms_grad_true}; "                                            \
        f"ChainCart.get_rms()[1]: {rms_grad_chain}"

    ##################################
    # weight-scaled and best-fitted. #
    ##################################

    # Init chain and chain_vec. 
    rep_list = _get_omm_rep_list(n_reps,
                                 is_mass_weighted=True, 
                                 mass_scaling_vec=numpy.arange(n_dofs/3)
                                 )

    chain = ChainCart(rep_list, is_rigid_fit=True)
    chain_vec = chain.get_chain_vec()

    chain_vec_fit = numpy.zeros((n_reps, n_dofs))
    for i_rep in range(n_reps-1, 0, -1):
        chain_vec_fit[i_rep], _, _ = rigid_fit(chain_vec[i_rep-1, :], 
                                               chain_vec[i_rep,   :])

    # Compute the true values.
    w = chain.get_wscale_vec()
    r_diff = chain_vec[0:-1, :] - chain_vec_fit[1:, :]

    rms_true = (numpy.sum(r_diff**2 * w[None, :], axis=1) / numpy.sum(w)*3)**.5
    rms_grad_true = r_diff * w[None, :] / numpy.sum(w) * 3 / rms_true[:, None]

    rms_chain, rms_grad_chain = chain.get_rms(chain_vec, return_grads=True)

    rms_diff = numpy.sum(numpy.abs(rms_chain - rms_true))
    assert rms_diff <= 1e-8,                                                   \
        f"Mode: with best-fit, with weight-scaling. "                          \
        f"ChainCart.get_rms(): Incorrect RMS. "                                \
        f"Truth: {rms_true}; "                                                 \
        f"ChainCart.get_rms(): {rms_chain}"
    
    rms_grad_diff = numpy.sum(numpy.abs(rms_grad_chain - rms_grad_true))
    assert rms_grad_diff <= 1e-8,                                              \
        f"Mode: with best-fit, with weight-scaling. "                          \
        f"ChainCart.get_rms(get_grads=True): Incorrect RMS gradients. "        \
        f"Truth: {rms_grad_true}; "                                            \
        f"ChainCart.get_rms()[1]: {rms_grad_chain}"
