# Unit test for RepPySCF objects.
# Zilin Song, 25 Aug 2022

from re import X
import pytest

import numpy

from pycospath.comms.pyscf import RepPySCF

from pyscf.gto.mole import Mole as PyscfMole
from pyscf.scf.hf   import RHF  as PyscfHF
from pyscf.dft.rks  import RKS  as PyscfKS

from pyscf.gto.mole import atom_mass_list

from pycospath.utils.units import akma_dist_to_au, au_dist_to_akma,            \
                                  au_ener_to_akma, au_grad_to_akma

def test_constructor():
    """Test if __init__ works correctly."""
    # Inputs
    # Test kwarg: pyscf_xc                 
    # xc    = 'to_test'
    atoms = 'O 0 0 0; H 0 1 0; H 0 0 1' 
    basis = '3-21g'
    spin  = 0
    chrg  = 0

    pyscf_rep = RepPySCF('hf', atoms, basis, spin, chrg)
    runner = pyscf_rep._get_pyscf_runner(pyscf_rep._mol) 
    assert isinstance(runner, PyscfHF)
    
    basis = 'ccpvtz'
    pyscf_rep = RepPySCF('b3lyp', atoms, basis, spin, chrg)
    runner = pyscf_rep._get_pyscf_runner(pyscf_rep._mol)
    assert isinstance(runner, PyscfKS)

    basis = 'def2-tzvp'
    pyscf_rep = RepPySCF('bp86', atoms, basis, spin, chrg)
    runner = pyscf_rep._get_pyscf_runner(pyscf_rep._mol)
    assert isinstance(runner, PyscfKS)
    
    with pytest.raises(KeyError):
        pyscf_rep = RepPySCF('b1ply', atoms, basis, spin, chrg)  # wrong key
    with pytest.raises(KeyError):
        pyscf_rep = RepPySCF('b2plyp', atoms, basis, spin, chrg) # DH needs mp2.
    
    # test_bad_mass_scaling_type()
    bad_mscale = 1.
    with pytest.raises(TypeError, 
                    match="Incompatible type of mscale_vec to numpy.ndarray."):
        RepPySCF('b3lyp', atoms, basis, spin, chrg, mscale_vec=bad_mscale)
    
    # test_bad_mass_scaling_shape()
    bad_mscale = numpy.asarray([1., 2.])
    with pytest.raises(ValueError, 
                       match="Incompatible shape of mscale_vec to the number "
                             "of atoms in openmm.system."):
        RepPySCF('b3lyp', atoms, basis, spin, chrg, mscale_vec=bad_mscale)
    
    # test_good_input()
    good_mscale = numpy.asarray(numpy.random.random_sample(3))

    try: 
        RepPySCF('hf', atoms, basis, spin, chrg, mscale_vec=good_mscale)
    except Exception as e:
        assert False,                                                          \
            f"RepPySCF(..., mass_scaling={good_mscale}) should not raise {e}"

def test_members():
    """Test various class members in RepPySCF."""
    
    # Inputs
    # Test kwarg: pyscf_xc
    xc    = 'b3lyp'
    atoms = 'O 0 0 0; H 0 1 0; H 0 0 1; H 1 0 0' 
    basis = 'ccpvtz'
    spin  = 0
    chrg  = 1

    mol = PyscfMole()
    mol.atom   = atoms
    mol.basis  = basis
    mol.spin   = spin
    mol.charge = chrg
    mol.build()

    rep = RepPySCF(pyscf_xc=xc, pyscf_mole_atoms=atoms, pyscf_mole_basis=basis,
                   pyscf_mole_spin=spin, pyscf_mole_chrg=chrg)

    rep_vec = numpy.asarray([0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.])
    rep_vec_bohr = akma_dist_to_au(rep_vec)

    """Test get_num_dofs."""
    n_dofs = rep.get_num_dofs()
    n_atoms = int(n_dofs/3)
    assert n_dofs == 12,                                                       \
        f"RepPySCF.get_num_dofs: incorrect n_dofs indicator: get_num_dofs "    \
        f"Truth: 12; RepPySCF.get_num_dofs(): {rep.get_num_dofs()}."

    """Test if coor and rep_vec are binded correctly. """

    # RepPySCF.get_coor()
    coor_return = numpy.asarray([c[1] for c in rep.get_coor()])
    coor_diff = coor_return-rep_vec_bohr.reshape(-1, 3)

    assert numpy.sum(numpy.abs(coor_diff)) <= 1e-6,                            \
        f"RepPySCF.get_coor: incorrect coor-rep_vec binding: __init__ "        \
        f"coor: {rep_vec_bohr.reshape(-1, 3)}; "                               \
        f"RepPySCF.get_coor: {coor_return}"

    # RepPySCF.get_rep_vec()
    rep_vec_return = rep.get_rep_vec()
    rep_vec_diff = rep_vec - rep_vec_return
    assert numpy.sum(numpy.abs(rep_vec_diff)) <= 1e-6,                         \
        f"RepPySCF.get_rep_vec: incorrect coor-rep_vec binding: __init__ "     \
        f"rep_vec: {rep_vec}; RepPySCF.get_rep_vec: {rep_vec_return}"

    # RepPySCF.set_coor()
    coor_val = (numpy.random.random_sample((n_atoms, 3)) - 0.5) * 3. 
    atm = [a[0] for a in mol.atom] # atom names. 

    coor_new = [[atm[i], coor_val[i]] for i in range(n_atoms)] # Bohr
    rep_vec_new = au_dist_to_akma(coor_val).flatten()          # Angstrom

    rep.set_coor(coor_new)

    coor_new_return = rep.get_coor()

    test_atom_names = []
    for i_atom in range(n_atoms):
        test_atom_names.append(
            coor_new[i_atom][0] == coor_new_return[i_atom][0]
        )
    assert not (False in test_atom_names),                                     \
        f"RepPySCF.set_coor: incorrect coor atom name binding: set_coor "      \
        f"coor: {coor_new}; RepPySCF.get_coor: {coor_new_return}"

    test_atom_coors = []
    for i_atom in range(n_atoms):
        test_atom_coors.append(
            numpy.sum(numpy.abs(
                coor_new[i_atom][1] - coor_new_return[i_atom][1])) <= 1e-8
        )
    assert not (False in test_atom_coors),                                     \
        f"RepPySCF.set_coor: incorrect coor-rep_vec binding: set_coor "        \
        f"coor: {coor_new}; RepPySCF.get_coor: {coor_new_return}"

    rep_vec_new_return = rep.get_rep_vec()

    assert numpy.sum(numpy.abs(rep_vec_new - rep_vec_new_return)) <= 1e-6,     \
        f"RepPySCF.set_coor: incorrect coor-rep_vec binding: set_coor "        \
        f"rep_vec: {rep_vec_new}; RepPySCF.get_rep_vec: {rep_vec_new_return}"
    
    # RepPySCF.set_rep_vec()
    coor_val = (numpy.random.random_sample((n_atoms, 3)) - 0.5) * 3. 
    atm = [a[0] for a in mol.atom] # atom names. 

    coor_new = [[atm[i], coor_val[i]] for i in range(n_atoms)] # Bohr
    rep_vec_new = au_dist_to_akma(coor_val).flatten()          # Angstrom

    rep.set_rep_vec(rep_vec_new)

    coor_new_return = rep.get_coor()

    test_atom_names = []
    for i_atom in range(n_atoms):
        test_atom_names.append(
            coor_new[i_atom][0] == coor_new_return[i_atom][0]
        )
    assert not (False in test_atom_names),                                     \
        f"RepPySCF.set_rep_vec: incorrect coor atom name binding: set_rep_vec "\
        f"coor: {coor_new}; RepPySCF.get_coor: {coor_new_return}"
    
    test_atom_coors = []
    for i_atom in range(n_atoms):
        test_atom_coors.append(
            numpy.sum(numpy.abs(
                coor_new[i_atom][1] - coor_new_return[i_atom][1])) <= 1e-8
        )
    assert not (False in test_atom_coors),                                     \
        f"RepPySCF.set_rep_vec: incorrect coor-rep_vec binding: set_rep_vec "  \
        f"coor: {coor_new}; RepPySCF.get_coor: {coor_new_return}"

    rep_vec_new_return = rep.get_rep_vec()

    assert numpy.sum(numpy.abs(rep_vec_new - rep_vec_new_return)) <= 1e-6,     \
        f"RepPySCF.set_rep_vec: incorrect coor-rep_vec binding: set_rep_vec "  \
        f"rep_vec: {rep_vec_new}; RepPySCF.get_rep_vec: {rep_vec_new_return}"

    """Test if mass_weighting and scaling are handled correctly."""
    masses = atom_mass_list(mol, isotope_avg=True)
    wscale = numpy.random.random_sample(n_atoms) * 10.

    # For the following three cases, 
    # test RepPySCF.is_mass_weighted() and RepPySCF.get_dof_weighting_vec()
    
    # scaling on weight. 
    rep = RepPySCF(pyscf_xc=xc, pyscf_mole_atoms=atoms, pyscf_mole_basis=basis,
                   pyscf_mole_spin=spin, pyscf_mole_chrg=chrg, 
                   is_mass_weighted=False, mscale_vec=wscale)

    assert rep.is_mass_weighted() == False,                                    \
        f"RepPySCF.is_mass_weighted: incorrect is_mass_weighted indicator. "   \
        f"Truth: False; rep.is_mass_weighted() {rep.is_mass_weighted()}"

    rep_wscale_truth = numpy.repeat(wscale, 3)
    rep_wscale       = rep.get_wscale_vec()
    assert (rep_wscale == rep_wscale_truth).all(),                             \
        f"RepPySCF.get_dof_weighting_vec: incorrect dof weighting vector. "    \
        f"Applied weight-scaling only. Truth: {rep_wscale_truth}; "            \
        f"rep.get_dof_weighting_vec() {rep_wscale}"
    
    # scaling on mass.
    rep = RepPySCF(pyscf_xc=xc, pyscf_mole_atoms=atoms, pyscf_mole_basis=basis,
                   pyscf_mole_spin=spin, pyscf_mole_chrg=chrg, 
                   is_mass_weighted=True, mscale_vec=None)

    assert rep.is_mass_weighted() == True,                                     \
        f"RepPySCF.is_mass_weighted: incorrect is_mass_weighted indicator. "  \
        f"Truth: False; rep.is_mass_weighted() {rep.is_mass_weighted()}"

    rep_wscale_truth = numpy.repeat(masses, 3)
    rep_wscale = rep.get_wscale_vec()
    assert (rep_wscale == rep_wscale_truth).all(),                             \
        f"RepPySCF.get_dof_weighting_vec: incorrect dof weighting vector. "   \
        f"Applied mass-weighting only. Truth: {rep_wscale_truth}; "            \
        f"rep.get_dof_weighting_vec() {rep_wscale}"

    # scaling on mass and weighting.
    rep = RepPySCF(pyscf_xc=xc, pyscf_mole_atoms=atoms, pyscf_mole_basis=basis,
                   pyscf_mole_spin=spin, pyscf_mole_chrg=chrg, 
                   is_mass_weighted=True, mscale_vec=wscale)
    assert rep.is_mass_weighted() == True,                                     \
        f"RepPySCF.is_mass_weighted: incorrect is_mass_weighted indicator. "  \
        f"Truth: False; rep.is_mass_weighted() {rep.is_mass_weighted()}"

    rep_wscale_truth = numpy.repeat(masses * wscale, 3)
    rep_wscale = rep.get_wscale_vec()
    assert (rep_wscale == rep_wscale_truth).all(),                             \
        f"RepPySCF.get_dof_weighting_vec: incorrect dof weighting vector. "   \
        f"Applied mass-weighting & weight-scaling. Truth: {rep_wscale_truth}; "\
        f"rep.get_dof_weighting_vec() {rep_wscale}"
    
def test_ener_grad():
    """Test if energy and gradient are correct. """
    # Set reference PySCF calculations. 
    s, c, a, u, b = 0, 0, 'O 0 0 0; H 0 1 0; H 0 0 1', 'Ang', '6-31g'
    x = 'b3lyp'

    mol = PyscfMole()
    mol.spin   = s
    mol.charge = c
    mol.atom   = a
    mol.unit   = u
    mol.basis  = b
    mol.build()

    pyscf_ref = PyscfKS(mol, xc = x)

    e_truth = pyscf_ref.kernel()                    # energies  in Hartree
    g_truth = pyscf_ref.nuc_grad_method().kernel()  # gradients in Hartree/Bohr

    e_truth = au_ener_to_akma(e_truth)              # to kcal/mol
    g_truth = au_grad_to_akma(g_truth).flatten()    # to kcal/mol/A
    
    pyscf_rep = RepPySCF(pyscf_xc=x, pyscf_mole_atoms=a, pyscf_mole_basis=b, 
                         pyscf_mole_spin=s, pyscf_mole_chrg=c)

    e, g = pyscf_rep.get_ener_grad() # e in kcal/mol, g in kcal/mol/A

    assert numpy.abs(e - e_truth) <= 1e-8,                                     \
        f"RepPySCF.get_ener_grad: incorrect energy value. "                    \
        f"Truth: {e_truth} kcal/mol; rep.get_ener_grad() {e}"
    
    assert numpy.sum(numpy.abs(g - g_truth)) <=1e-6,                           \
        f"RepPySCF.get_ener_grad: incorrect gradient value. "                  \
        f"Truth: {g_truth} kcal/mol/A; rep.get_ener_grad() {g}"

    """Change coordinates and see if the ener and grad are correct.
    """
    
    new_a = 'O 0 0 0; H 0 0.9527 0; H 0 0 0.9527'
    mol.atom = new_a
    mol.unit = 'bohr'
    mol.build()

    new_coor =[ ['O', [0., 0.,     0.    ]],  # In the unit of 'Bohr
                ['H', [0., 0.9527, 0.    ]], 
                ['H', [0., 0.,     0.9527]]
                ]
                
    pyscf_ref = PyscfKS(mol, xc = x)

    e_truth = pyscf_ref.kernel()                    # energies  in Hartree
    g_truth = pyscf_ref.nuc_grad_method().kernel()  # gradients in Hartree/Bohr

    e_truth = au_ener_to_akma(e_truth)
    g_truth = au_grad_to_akma(g_truth).flatten()

    pyscf_rep.set_coor(new_coor)
    e, g = pyscf_rep.get_ener_grad() # e in kcal/mol, g in kcal/mol/A

    assert numpy.abs(e - e_truth) <= 1e-8,                                     \
        f"RepPySCF.get_ener_grad: incorrect energy value after coor change. "  \
        f"Truth: {e_truth} kcal/mol; rep.get_ener_grad() {e}"
        
    assert numpy.sum(numpy.abs(g - g_truth)) <= 1e-6,                          \
        f"RepPySCF.get_ener_grad: incorrect gradient value after coor change." \
        f" Truth: {g_truth} kcal/mol/A; rep.get_ener_grad() {g}"
    