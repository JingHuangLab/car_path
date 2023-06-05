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
import numpy
from typing import List, Tuple, Union

from pycospath.comms import RepCartBase
from pycospath.utils.units import akma_dist_to_au, au_dist_to_akma,            \
                                  au_ener_to_akma, au_grad_to_akma

# pyscf utils
from pyscf.gto.mole import Mole as PyscfMole
from pyscf.scf.hf   import RHF  as PyscfHF
from pyscf.dft.rks  import RKS  as PyscfKS

# pyscf helpers
from pyscf.gto.mole  import atom_mass_list, format_atom
from pyscf.dft.libxc import parse_xc_name

class RepPySCF(RepCartBase):
    """The PySCF potential."""

    def __init__(self, 
                 pyscf_xc:         str,                     # pot
                 pyscf_mole_atoms: Union[str, List, Tuple], # pot
                 pyscf_mole_basis: Union[str, dict],        # pot
                 pyscf_mole_spin:  int,                     # pot
                 pyscf_mole_chrg:  int,                     # pot
                 is_mass_weighted: bool = True,
                 mscale_vec:       numpy.ndarray = None):
        """Create a PySCF potential function.
        NOTE: 
        Currently allow only RHF and RKS calculations. PySCF spin and charge are
        not checked for consistency. 
        
        NOTE:
        There are two types of coordinate sets in this class: 
        1. self._rep_vec: the row of chain_vec,       unit: Angstrom; 
        2. self._coor:    the PySCF internal  format, unit: Bohr. 
                        PySCF internal format: 
                            | atom = [[atom1, (x, y, z)],
                            |         [atom2, (x, y, z)],
                            |         ...
                            |         [atomN, (x, y, z)]]

        Parameters
        ----------
        pyscf_xc: str
            The density functional used to treat the electron x-c energies. It 
            is passed exactly to the `xc` attribute of the pyscf.dft.RKS() 
            object. A special case is that, if 'hf' is passed, a pyscf.scf.RHF()
            object is created instead. 
            ```
            mol = gto.M(atom = 'O 0 0 0; H 0 0 1; O 0 1 0')
            
            if pyscf_xc == 'hf':
                from pyscf import gto, scf
                rhf = scf.RHF(mol)

            else:
                from pyscf import gto, dft
                rks = dft.RKS(mol)
                rks.xc = pyscf_xc         
            ```
        
        pyscf_mole_atoms: Union[str, List, Tuple]
            The atoms and their coordinates of the pyscf.gto.Mole() object. Any
            object that can be accepted by pyscf.gto.Mole() could be specified 
            in here. 
            NOTE: the unit of the atomic coordinates defined by pyscf_mole_atoms
                  is forced to be Angstrom.

        pyscf_mole_basis: Union[str, dict]
            The basis set of the pyscf.gto.Mole() object. Any object that can be
            accepted by pyscf.gto.Mole() could be specified in here.

        pyscf_mole_spin: int
            The total spin of the pyscf.gto.Mole() object. 
            Note that this value refers to (spin_up - spin_down) in PySCF.

        pyscf_mole_chrg: int
            The total charge of the pyscf.gto.Mole() object.

        is_mass_weighted: bool, default=True
            If the replica vector should be mass-weighted for the calculations 
            of the RMS. 
            Note that mass-weighting can be applied together with mass-scaling. 

        mscale_vec: numpy.ndarray of shape (n_atoms, ), default=None
            If the replica vector should be mass-scaled for the calculations of
            the RMS.
            Note that mass-weighting can be applied together with mass-scaling. 
            If None, no scaling (or scale by one) will be used. 
        """
        if not isinstance(pyscf_xc, str):
            raise TypeError("pyscf_xc must be a string type.")

        # check if specified xc name exists. 
        if pyscf_xc != 'hf': # If not 'hf', it's some DFT. 

            try:
                parse_xc_name(pyscf_xc)

            except Exception as e:
                raise e

        self._pyscf_xc = pyscf_xc

        # check if mole.atom is given correctly. 
        try:
            # NOTE: 
            # pyscf.gto.mole.format_atom() always Returns coordinates in Bohr,
            # The kwarg 'unit' specifies the unit of the input pyscf_mole_atoms.
            coor = format_atom(pyscf_mole_atoms, unit='Ang')

        except Exception as e:
            raise e

        # check if pyscf.gto.Mole() objects can be built correctly. 
        try:
            self._mol = PyscfMole()
            self._mol.charge   = pyscf_mole_chrg
            self._mol.spin     = pyscf_mole_spin
            self._mol.symmetry = False
            self._mol.atom     = coor   # units in Bohr, see the NOTE above
            self._mol.unit     = 'Bohr' # yep, coor.unit = 'Bohr'
            self._mol.basis    = pyscf_mole_basis

            self._mol.build()   # Mole.build() to apply changes on Mole.attrs. 
    
        except Exception as e:
            raise e

        # Init self._pot, self._coor, self._n_atoms, and self._wscale_vec.
        # PySCF comms does not implement self._pot
        RepCartBase.__init__(self, coor, None, is_mass_weighted, mscale_vec)

    def get_num_atoms(self) -> int:
        """Get the number of atoms from the self._mol object."""
        return self._mol.natm

    def get_atom_mass_vec(self) -> numpy.ndarray:
        """Get the copy of the atomic mass vector.
       
        Returns
        -------
        mass_vec: numpy.ndarray of shape (n_atoms, )
            The copy of the mass vector.
        """
        mass_vec = numpy.asarray(atom_mass_list(self._mol, isotope_avg=True))
        return mass_vec
        
    def coor_to_rep_vec(self, 
                        coor: List
                        ) -> numpy.ndarray:
        """Convert the replica coordinate PySCF internal coordinate to the 
        representation of the replica vector. 

        Parameters
        ----------
        coor: List 
            The replica coordinate PySCF internal coordinate format, 
            unit in Bohr. 
            PySCF internal format: 
                | atom = [[atom1, (x, y, z)],
                |         [atom2, (x, y, z)],
                |         ...
                |         [atomN, (x, y, z)]]

        Returns
        -------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            A vector converted from the replica coordinate PySCF internal 
            coordinate format (self._coor).
        """
        # flatten() returns a copy.
        # coor is in Bohr, need to convert to Angstrom before setting rep_vec. 
        rep_vec = au_dist_to_akma(numpy.asarray([c[1] for c in coor])).flatten()
        return rep_vec

    def rep_vec_to_coor(self,
                        rep_vec: numpy.ndarray
                        ) -> List:
        """Convert the representation of the replica vector (rep_vec) to a new
        replica coordinate List. 

        Parameters
        ----------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            The representation of the replica vector.

        Returns
        -------
        coor: List
            A new coordinate List converted from the representation of the 
            replica vector (rep_vec).
        """
        # Convert rep_vec values to Bohr (coor is always in Bohr). 
        rep_vec_bohr = akma_dist_to_au(rep_vec).reshape(self._n_atoms, 3)
        coor = []
        for i_atom in range(self._n_atoms):
            coor.append([   self._coor[i_atom][0], 
                            ( rep_vec_bohr[i_atom][0], 
                              rep_vec_bohr[i_atom][1], 
                              rep_vec_bohr[i_atom][2]  
                            )
                        ]
                    )
        return coor
    
    def _get_pyscf_runner(self, 
                          mol: PyscfMole
                          ) -> Union[PyscfHF, PyscfKS]:
        """Get the PySCF class that gives the energy and gradient calls. 

        Parameters
        ----------
        mol: pyscf.gto.Mole()
            The PySCF Mole object. 

        Returns
        -------
        pyscf_runner: pyscf.scf.RHF() or pyscf.dft.RKS()
            The PySCF object that implements the kernel() method which returns 
            the energy and the nuc_grad_method() method that returns the
            pyscf.grad.Gradients object (depends on what pyscf_runner is) that 
            implments the kernel() method which returns the gradients. 
        """
        # check if energy and gradient providers can be made correctly. 
        pyscf_runner = PyscfHF(mol) if self._pyscf_xc == 'hf' else  \
                       PyscfKS(mol, xc = self._pyscf_xc)
        
        return pyscf_runner

    def get_ener_grad(self) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the replica vector (self._rep_vec). 
        
        Returns
        -------
        (ener, grad): (float, numpy.ndarray of shape (n_dofs, ) )
            ener: the energy of the replica vector;
            grad: the gradient of the replica vector.
        """
        self._mol.atom = self._coor
        self._mol.unit = 'Bohr'     # self._coor is always in Bohr
        self._mol.build()

        pyscf_runner = self._get_pyscf_runner(self._mol)

        ener = pyscf_runner.kernel()                   # unit in Hartree
        grad = pyscf_runner.nuc_grad_method().kernel() # unit in Hartree/Bohr

        ener = au_ener_to_akma(numpy.asarray(ener))    # unit in kcal/mol
        grad = au_grad_to_akma(numpy.asarray(grad))    # unit in kcal/mol/A
        grad = grad.flatten()

        return ener, grad
