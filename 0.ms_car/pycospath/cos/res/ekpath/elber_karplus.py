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

from pycospath.cos   import ResCosBase
from pycospath.chain import ChainBase

class ElberKarplusPath(ResCosBase):
    """The Elber-Karplus Path method.
    
    A Method for Determining Reaction Paths in Large Molecules: Application to 
    Myoglobin.
    R. Elber, M. Karplus, Chem. Phys. Lett. 1987, 139, 375-380
    10.1016/0009-2614(87)80576-6
    
    Note that this method utilizes linear tangent to represent the path, it does
    not account for the curvature of the reaction path and, in theory, does not 
    find the steepest descent path (SDP). """

    def __init__(self, 
                 chain:     ChainBase,
                 verbose:   bool = True,
                 eb_fconst: float = 1000):
        """Create a Elber-Karplus Path calculation.
        
        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        
        eb_fconst: float, default=1000
            The restraint force constant in the unit of kcal/mol/Angstrom.
        """
        ResCosBase.__init__(self, chain, verbose)

        self._eb_fconst = eb_fconst

    def _get_res_grads_vec(self,
                           chain_vec: numpy.ndarray
                           ) -> numpy.ndarray:
        """Determine and return the restraint gradient vector that acts on the  
        chain vector.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the constraint condition.

        Returns
        -------
        restr_grads_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The gradient vector that gives the restraint forces. 
        """
        # TODO: continue at here.
        
