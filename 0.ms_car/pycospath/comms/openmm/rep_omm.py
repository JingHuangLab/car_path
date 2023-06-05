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
from typing import Tuple

from pycospath.comms        import RepCartBase
from pycospath.comms.openmm import PotOpenMM

from openmm.unit import Quantity as OmmQuantity
from openmm      import unit

class RepOpenMM(RepCartBase):
    """The OpenMM replica.
    rep_vec: units in Angstrom;
    ener:    units in kcal/mol;
    grad:    units in kcal/mol/Angstrom;
    mass:    units in AMU.
    """

    def __init__(self,
                 coor:             OmmQuantity,
                 pot:              PotOpenMM,
                 is_mass_weighted: bool = True, 
                 mscale_vec:       numpy.ndarray = None):
        """Create an OpenMM replica object.

        Parameters
        ----------
        coor: openmm.unit.Quantity
            The replica coordinate openmm.unit.Quantity. 

        pot: PotOpenMM
            The OpenMM potential PotOpenMM.
        
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
        # Sanity checks.
        if not isinstance(pot, PotOpenMM):
            raise TypeError("RepOpenMM must be initialized on PotOpenMM.")
            
        if not isinstance(coor, OmmQuantity):
            raise TypeError("coor must be a openmm.unit.Quantity object.")

        # Init self._pot, self._coor, self._n_atoms, and self._wscale_vec.
        RepCartBase.__init__(self, coor, pot, is_mass_weighted, mscale_vec)
        
        # Sanity check on coor
        coor_array = numpy.asarray(coor.value_in_unit(unit.angstrom))

        if coor_array.shape != (self._n_atoms, 3):
            raise ValueError("Incompatible shape of coor to openmm.system.")

    def get_num_atoms(self) -> int:
        """Get the number of atoms from the PotOpenMM object. """
        pot: PotOpenMM = self._pot
        return pot.get_omm_system().getNumParticles()

    def get_atom_mass_vec(self) -> numpy.ndarray:
        """Get the copy of the atomic mass vector.
       
        Returns
        -------
        mass_vec: numpy.ndarray of shape (n_atoms, )
            The copy of the mass vector.
        """
        pot: PotOpenMM = self._pot
        omm_sys = pot.get_omm_system()

        mass_vec = numpy.ones((self._n_atoms, )) # This is copied. 

        for i_atom in range(self._n_atoms):
            i_mass = omm_sys.getParticleMass(i_atom)
            mass_vec[i_atom] = i_mass.value_in_unit(unit.atomic_mass_unit)
        
        return mass_vec
        
    def coor_to_rep_vec(self,
                        coor: OmmQuantity
                        ) -> numpy.ndarray:
        """Convert the replica coordinate openmm.unit.Quantity (coor) to the 
        representation of the replica vector. 
        
        Parameters
        ----------
        coor: openmm.unit.Quantity
            The replica coordinate openmm.unit.Quantity. 

        Returns
        -------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            A vector converted from the replica coordinate openmm.unit.Quantity
            (self._coor).
        """
        # flatten() returns a copy.
        rep_vec = numpy.asarray(coor.value_in_unit(unit.angstrom)).flatten()
        return rep_vec

    def rep_vec_to_coor(self,
                        rep_vec: numpy.ndarray
                        ) -> OmmQuantity:
        """Convert the representation of the replica vector (rep_vec) to a new 
        replica coordinate openmm.unit.Quantity.

        Parameters
        ----------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            The representation of the replica vector.

        Returns
        -------
        coor: openmm.unit.Quantity
            A new coordinate openmm.unit.Quantity converted from the 
            representation of the replica vector (rep_vec).
        """
        # create a new coor openmm.unit.Quantity -> copy not needed. 
        coor = OmmQuantity(rep_vec.reshape((self._n_atoms, 3)), unit.angstrom)
        return coor

    def get_ener_grad(self) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the replica vector (self._rep_vec). 
        
        Returns
        -------
        (ener, grad): (float, numpy.ndarray of shape (n_dofs, ) )
            ener: the energy of the replica vector;
            grad: the gradient of the replica vector.
        """
        pot:  PotOpenMM   = self._pot
        coor: OmmQuantity = self._coor

        ener, grad = pot.get_ener_grad(coor)
        ener = ener.value_in_unit(unit.kilocalorie_per_mole)

        grad = grad.value_in_unit(unit.kilocalorie_per_mole/unit.angstrom)
        grad = numpy.asarray(grad).flatten()

        return ener, grad
