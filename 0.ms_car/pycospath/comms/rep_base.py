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
import abc, numpy
from typing import Tuple

from pycospath.comms import PotBase

# NOTE: get_rep_vec returns a copy of self._rep_vec.
#       set_rep_vec and coor_to_rep_vec modify self._rep_vec with a copy of the 
#       passed array. 
#       rep_vec_to_coor makes a new coor object with self._rep_vec. 

class RepBase(abc.ABC):
    """The base class of replica."""

    def __init__(self, 
                 coor: object,
                 pot:  PotBase = None):
        """Create a replica object. 

        Parameters
        ----------
        coor: object
            The replica coordinate object. 
        
        pot: PotBase, default: None
            The replica potential object.
            Note that not all replica sub-class requires a pot object, in the
            case that a potential function is handled directly at the level of 
            RepBase, self._pot always has None value.  
        """
        self._pot     = pot
        self._coor    = coor
        self._rep_vec = self.coor_to_rep_vec(coor)

    def get_pot(self) -> PotBase:
        """Get the replica potential object (self._pot). 

        NOTE: 
        The returned object shares the same memory allocation with self._pot.
        
        Returns
        -------
        potential: PotBase
            The replica potential object (self._pot).
        """
        if self._pot is None:
            raise NotImplementedError("This replica class does not implement " 
                                      "potential objects.")
        return self._pot

    def get_coor(self) -> object:
        """Get the replica coordinate object (self._coor). 

        NOTE: 
        The returned object shares the same memory allocation with self._coor.
        
        Returns
        -------
        coor: object
            The replica coordinate object (self._coor). 
        """
        return self._coor

    def set_coor(self, 
                 coor: object
                 ) -> None: 
        """Set the replica coordinate object (self._coor). 

        NOTE: 
        The returned object shares the same memory allocation with self._coor.
        
        This method is intended for advanced uses and routine calculations do 
        not concern this method. 
        
        Also updates the value of the replica vector (self._rep_vec).
        See self.coor_to_rep_vec(coor)

        Parameters
        ----------
        coor: object
            The object used to set the replica coordinate object (self._coor). 
        """
        self._coor = coor

        # Also update the replica vector.
        self._rep_vec = self.coor_to_rep_vec(coor)

    def get_rep_vec(self) -> numpy.ndarray:
        """Get a copy of the replica vector (self._rep_vec.copy()).

        Returns
        -------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            A copy of the replica vector (self._rep_vec.copy()).
        """
        return self._rep_vec.copy()

    def set_rep_vec(self, 
                    rep_vec: numpy.ndarray
                    ) -> None:
        """Set the value of the replica vector (self._rep_vec) with a copy of 
        rep_vec (rep_vec.copy()).
        
        Also update the replica coordinate object (self._coor).
        See self.rep_vec_to_coor(rep_vec)
        
        Parameters
        ----------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            The vector used to set the value of replica vector (self._rep_vec).
        """
        self._rep_vec = rep_vec.copy()

        # Also update the system coordinate.
        self._coor = self.rep_vec_to_coor(rep_vec)

    def get_num_dofs(self) -> int:
        """Get the no. degrees-of-freedom in the replica vector (self._rep_vec).
        
        Returns
        -------
        n_dofs: int
            The no. degrees-of-freedoms in the replica vector (self._rep_vec).
        """
        return self._rep_vec.shape[0]

    @abc.abstractmethod
    def coor_to_rep_vec(self, 
                        coor: object
                        ) -> numpy.ndarray:
        """Convert the replica coordinate object (coor) to the representation of
        the replica vector. 
        
        Parameters
        ----------
        coor: object
            The replica coordinate object. 

            NOTE: Child class implementations of this abstract method must not 
                  modify the passed coor object, which shares the same memory 
                  allocation with self._coor. Modifications on the passed object
                  will unexpectedly change self._coor. 
                  See self.set_coor(coor). 

        Returns
        -------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            A vector converted from the replica coordinate object (self._coor).
        """

    @abc.abstractmethod
    def rep_vec_to_coor(self, 
                        rep_vec: numpy.ndarray
                        ) -> object:
        """Convert the representation of the replica vector (rep_vec) to a new 
        replica coordinate object.

        Parameters
        ----------
        rep_vec: numpy.ndarray of shape (n_dofs, )
            The representation of the replica vector.

        Returns
        -------
        coor: object
            A new coordinate object converted from the representation of the 
            replica vector (rep_vec).
        """

    @abc.abstractmethod
    def get_ener_grad(self) -> Tuple[float, numpy.ndarray]:
        """Get the energy and gradients of the replica vector (self._rep_vec). 
        
        Returns
        -------
        (ener, grad): (float, numpy.ndarray of shape (n_dofs, ) )
            ener: the energy of the replica vector;
            grad: the gradient of the replica vector.
        """

class RepCartBase(RepBase):
    """The base class of replicas in the Cartesian space."""

    def __init__(self, 
                 coor:             object, 
                 pot:              PotBase = None,
                 is_mass_weighted: bool = True, 
                 mscale_vec:       numpy.ndarray = None):
        """Create a Cartesian replica object.
        
        Parameters
        ----------
        coor: object
            The replica coordinate object. 
        
        pot: PotBase, default=None
            The replica potential object. 
            Note that not all replica sub-class requires a pot object, in the
            case that a potential function is handled directly at the level of 
            RepBase, self._pot always has None value.  

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
        RepBase.__init__(self, coor, pot)

        # Number of atoms. 
        self._n_atoms = self.get_num_atoms()

        # If the replica vector is mass_weighted. 
        wscale_vec = numpy.ones((self._n_atoms, ))
        
        # Apply mass-weighting
        self._is_mass_weighted = is_mass_weighted

        if self._is_mass_weighted == True:
            wscale_vec *= self.get_atom_mass_vec()

        # Apply mass-scaling. 
        if mscale_vec is None: # Don't scale atomic masses. 
            mscale_vec = numpy.ones((self._n_atoms, ))
            self._is_mass_scaled = False
        else:
            self._is_mass_scaled = True
        
        # Sanity checks on mscale_vec.
        if not isinstance(mscale_vec, numpy.ndarray):
            raise TypeError("Incompatible type of mscale_vec to numpy.ndarray.")

        else: 
            if mscale_vec.shape != (self._n_atoms, ):
                raise ValueError("Incompatible shape of mscale_vec to the "
                                 "number of atoms in openmm.system. ")
                                 
        self._mass_scale_vec = mscale_vec.copy()

        wscale_vec *= self._mass_scale_vec

        # Repeat each element three times to match the shape of the replica 
        # vectors (self._rep_vec)
        self._wscale_vec = numpy.repeat(wscale_vec, 3)

    def is_mass_weighted(self) -> bool:
        """Get if the weight scaling vector (self._wscale_vec) is mass-weighted. 
        """
        return self._is_mass_weighted

    def is_mass_scaled(self) -> bool:
        """Get if the weight scaling vector (self._wscale_vec) is mass-scaled. 
        """
        return self._is_mass_scaled

    def get_wscale_vec(self) -> numpy.ndarray:
        """Get the copy of the weighting scaling vector (self._wscale_vec.copy()
        ) for the scaling of the replica vector (self._rep_vec) degrees-of-free-
        doms.

        Returns
        -------
        wscale_vec: numpy.ndarray of shape (n_dofs, )
            The copy of the weighting scaling vector for the scaling of the
            replica vector degrees-of-freedoms.
        """
        return self._wscale_vec.copy()

    @abc.abstractmethod
    def get_num_atoms(self) -> int:
        """Get the number of atoms from the self._pot object or the simulating 
        system.
        """

    @abc.abstractmethod
    def get_atom_mass_vec(self)-> numpy.ndarray:
        """Get the copy of the atomic mass vector.
        
        Returns
        -------
        mass_vec: numpy.ndarray of shape (n_atoms, )
            The copy of the mass vector.
        """
