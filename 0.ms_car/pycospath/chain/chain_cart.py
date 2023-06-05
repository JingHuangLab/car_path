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

# Implementation Note I. 
# Z. Song, 01 Jun 2022.
# ChainCart objects serves as the interface to decouple the {replica vector <-> 
# chain vector} and {replica vector <-> system coordinates} correspondence. The 
# chain vector accounts for the mass-weighting and best-fitting operations when 
# computing RMS distances and potential grads. 

# Imports
import numpy
from typing import List, Tuple, Union

from pycospath.chain import ChainBase
from pycospath.comms import RepCartBase
from pycospath.utils import rigid_fit, rms

class ChainCart(ChainBase):
    """The chain of Cartesian replicas."""

    def __init__(self, 
                 replicas:     List[RepCartBase],
                 is_rigid_fit: bool = True):
        """Create a chain of Cartesian replicas.
        
        Parameters
        ----------
        replicas: List[RepCartBase]
            A list of RepCartBase objects that makes up the chain.

        is_rigid_fit: bool, default = True,
            If the replicas should be rigid-fitted onto the first replica to 
            compute the inter-replica RMS ditances.
        """
        ChainBase.__init__(self, replicas)

        # Fill the chain_vec with the rep_vec from each replica.
        chain_vec = numpy.zeros((self._n_reps, self._n_dofs))

        for i_rep in range(self._n_reps):
            if not isinstance(self._rep_list[i_rep], RepCartBase):
                raise TypeError("ChainCart must be initialized with "
                                "List[RepCartBase].")

            rep: RepCartBase = self._rep_list[i_rep]
            chain_vec[i_rep, :] = rep.get_rep_vec()

        # Optional: Rigid-fit each replica onto their preceding replica. 
        self._is_rigid_fit = is_rigid_fit

        if self._is_rigid_fit == True: 
            for i_rep in range(1, self._n_reps):
                chain_vec[i_rep, :], _, _ = rigid_fit(chain_vec[i_rep-1, :], 
                                                      chain_vec[i_rep,   :])
        
        self._chain_vec = chain_vec
        
        # Set the weight scaling vector for the replica vector dofs. 
        rep: RepCartBase = self._rep_list[0]
        self._wscale_vec = rep.get_wscale_vec()
      
    def is_rigid_fit(self) -> bool:
        """Get if the rigid-fit protocol is applied for the chain vector. 
        
        Returns
        -------
        best_fit: bool
            If the rigid-fit protocol is used for the chain.
        """
        return self._is_rigid_fit
        
    def get_wscale_vec(self) -> numpy.ndarray:
        """Get the copy of the weighting scaling vector (self._wscale_vec.copy()
        ) for the scaling of the rows of the chain vector (self._chain_vec) 
        degrees-of-freedoms.
        
        Returns
        -------
        wscale_vec: numpy.ndarray of shape (n_dofs, )
            The copy of the weighting scaling vector (self._wscale_vec.copy()) 
            for the scaling of the chain vector (self._chain_vec) degrees-of-
            freedoms.
        """
        return self._wscale_vec.copy()
        
    def get_chain_vec(self) -> numpy.ndarray:
        """Get a copy of the chain vector (self._chain_vec.copy()).

        Returns
        -------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            A copy of the chain vector (self._chain_vec.copy()).
        """
        return self._chain_vec.copy()

    def set_chain_vec(self, 
                      chain_vec: numpy.ndarray
                      ) -> None:
        """Set the value of the chain vector (self._chain_vec) with a copy of 
        chain_vec (chain_vec.copy()). 
        
        Also update the replica vector (rep_vec) for all replica objects on this 
        chain. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The vector used to set the value of chain vector (self._chain_vec).
        """
        self._chain_vec = chain_vec.copy()

        # Also update the rep_vec for each Rep2D. 
        c_vec_val = self._chain_vec.copy() # such that the mem alloc differs. 

        for i_rep in range(self._n_reps):
            rep_vec_row = c_vec_val[i_rep, :]

            # Optional: Rigid-fit each row of chain_vec back to rep_vec. 
            if self._is_rigid_fit == True:
                rep_vec_ref = self._rep_list[i_rep].get_rep_vec()
                rep_vec_row, _, _ = rigid_fit(rep_vec_ref, rep_vec_row)
            
            self._rep_list[i_rep].set_rep_vec(rep_vec_row)

    def _get_aligned_chain_vec(self):
        """Do rigid fitting"""
        aligned_cvec = numpy.zeros(self._chain_vec.shape)
        for i_rep in range(1, self._n_reps):
            aligned_cvec[i_rep, :], _, _ = rigid_fit(
                                              self._chain_vec[i_rep-1, :], 
                                              self._chain_vec[i_rep, :]  )
        return aligned_cvec
    
    def get_eners_grads(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the energies vector and the gradients vector of the chain vector 
        (self._chain_vec).

        NOTE: If self._is_rigid_fit == True, the gradient vector computed from 
              the replica objects are the gradients on the unfitted replica 
              vectors. The unfitted gradient vectors should be rotated with the
              same rigid-fit protocol used in the rep_vec -> chain_vec tranform
              to produce the fitted gradient vector for the chain vector.

        Returns
        -------
        (eners_vec, grads_vec): (numpy.ndarray of shape (n_reps, ), 
                                 numpy.ndarray of shape (n_reps, n_dofs))
            eners_vec: The energies  vector of the chain vector;
            grads_vec: The gradients vector of the chain vector. 
        """
        # Fill the energy and gradient from each replica in chain_vec.
        eners_vec = numpy.zeros(shape=(self._n_reps, ))
        grads_vec = numpy.zeros(shape=(self._n_reps, self._n_dofs))

        for r in range(self._n_reps):
            # The ener & unrotated grad.
            ener, grad = self._rep_list[r].get_ener_grad()

            # Optional: Rigid-fit requires that the grads to be rotated as well.
            if self._is_rigid_fit == True:
                rep_vec_raw = self._rep_list[r].get_rep_vec()
                rep_vec_fit = self._chain_vec[r]
                
                # Rotate rep_vec grads onto the chain_vec.
                _, ro, _ = rigid_fit(rep_vec_fit, rep_vec_raw)
                grad = (grad.reshape(-1, 3) @ ro).flatten()
            
            eners_vec[r]    = ener
            grads_vec[r, :] = grad

        return eners_vec, grads_vec

    def get_rms(self, 
                chain_vec:    numpy.ndarray,
                return_grads: bool = False
                ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        """Computes the mass-weighted root-mean-square (RMS) distances b/w the 
        adjacent rows of the chain vector (self._chain_Vec): 
            ||r_{i} - r_{i+1}||_{2} ;
        and the gradients of RMS distances:
            ||r_{i} - r_{i+1}||_{2} w.r.t. r_{i}, if return_grads is true.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector along which the RMS b/w adjacent replica vectors is
            to be computed. 
        
        return_grads: bool
            If the gradients of RMS distances should also be returned. 
            
        Returns
        -------
        rms: numpy.ndarray of shape (n_reps-1, )
            The RMS distances b/w adjacent rows of the chain vector. 
        
        (rms, rms_grad): (numpy.ndarray of shape (n_reps-1, ), 
                          numpy.ndarray of shape (n_reps-1, n_dofs))
            The RMS distances b/w adjacent rows of the chain vector and their 
            gradients w.r.t. r_{i}. Returns only if return_grads = True. 
        """
        wscale = self._wscale_vec.copy()
        redund = 3      # 3 dof weights per atom.

        return rms(chain_vec, wscale, redund, return_grads)