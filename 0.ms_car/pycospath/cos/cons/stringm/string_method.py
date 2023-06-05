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
import numpy, copy
from typing import Union

from pycospath.cos   import ConsCosBase
from pycospath.chain import ChainBase
from pycospath.utils import IntpolBase, CubicSplineIntpol

class StringMethod(ConsCosBase):
    """The Simplified String Method: 

    Simplified and improved string method for computing the minimum energy paths
    in barrier-crossing events.
    W. E, W. Ren, E. Vanden-Eijnden, J. Chem. Phys., 2007, 126, 164103.  
    10.1063/1.2720838

    The path is represented as a parametrized string and the replicas along the
    path is re-distributed (re-parametrized) for equal arc-length. Such string
    re-parametrization could be regarded as an implicit constraint condition and
    is the most numerically stable chain-of-states on constraint conditions. 

    This method in general has better numerical stability than the version with
    gradient projections. 
    """

    def __init__(self, 
                 chain:   ChainBase, 
                 verbose: bool = False, 
                 intpol:  Union[str, IntpolBase] = 'cspline'):
        """Create a Simplified String Method calculation. 

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        
        intpol: Union[str, IntpolBase], default='cspline'
            An instance of the type of interpolator to fit for the path. 
        """
        ConsCosBase.__init__(self, chain, verbose)

        # Interpolators.
        if isinstance(intpol, IntpolBase):
            self._intpol_instnce = intpol
        
        elif isinstance(intpol, str):
            
            if not intpol in ['cspline']:
                raise ValueError(f"Unknown interpolator spec: {intpol}")

            self._intpol_instnce = CubicSplineIntpol()

        else:
            raise TypeError("The interpolator must be an IntpolBase object.")
        
        # If the string has been re-parametrized for at least once.
        self._init_reparam = False
        # The list of interpolators to re-parametrize the path.
        self._intpol_list = []     

    def _get_path_tan(self,
                      chain_vec:numpy.ndarray
                      ) -> numpy.ndarray:
        """Get the path tangent vector at each replica along the chain. 

        Here: Parametrized tangent.
              Raises a RuntimeError if re-parametrization has not been performed
              for at least once. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector on which the path tangent vectors are computed.

        Returns
        -------
        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs).
            The path tangent vector at each replica along the chain.
        """
        # NOTE: this method is used to project potential gradients in the String
        #       Method with Gradient Projections: Simplified String Method does 
        #       not make the use of the path tangent vector.
        if self._init_reparam == False:
            raise RuntimeError("The string has not yet been re-parametrized.")

        # Location of replicas on the string. 
        a_vec = self._cons_alpha_vec(chain_vec)

        # Compute string tangent vector. 
        path_tan_vec = numpy.zeros(chain_vec.shape)

        for i_intpol in range(len(self._intpol_list)):
            intpol: IntpolBase = self._intpol_list[i_intpol]
            path_tan_vec[:, i_intpol] = intpol.get_grad(a_vec)
        
        return path_tan_vec

    def _get_cons_chain_vec(self, 
                            chain_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Determine and return the chain vector that satisfies the constraint
        condition. 

        Specifically, this method first fits the interpolators, re-parametrizes 
        the string, and re-distributes the replicas along the string for equal 
        arc-length. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the constraint condition.

        Returns
        -------
        constr_chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector that satisfies the constraint condition.
        """
        # Replica location vector along the path. 
        alpha_vec = self._cons_alpha_vec(chain_vec)

        # Re-parametrize the replica locations along the path. 
        chain_vec = self._cons_reparametrize(chain_vec, alpha_vec)

        return chain_vec

    def _cons_alpha_vec(self, 
                        chain_vec: numpy.ndarray
                        ) -> numpy.ndarray:
        """Get the replica location vector, that is, the alphas of each replica 
        along the parametrized path. 
        
        The alphas are normalized to [0, 1]. 

        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector.
        
        Returns
        -------
        alpha_vec: numpy.ndarray of shape (n_reps, )
            The alpha vector that denotes the normalized replica locations.
        """
        # Compute the total arclength as RMS distances along the chain.
        locs     = numpy.zeros((chain_vec.shape[0], ))
        locs[1:] = self.get_chain().get_rms(chain_vec)

        # Compute the normalized arclength (alpha_vec) along the string. 
        alpha_vec = numpy.cumsum(locs) / numpy.sum(locs)

        return alpha_vec

    def _cons_reparametrize(self, 
                            chain_vec: numpy.ndarray, 
                            alpha_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Reparametrize the replica degree-of-freedoms.

        Specifically, for each replica degree-of-freedom, this method:
        1. Fits the interpolators, and;
        2. Redistributes the string replicas for equal arc-length, and;
        3. Also updates the interpolator for this degree-of-freedom in 
           self._intpol_list. 

        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be reparametrized.

        alpha_vec: numpy.ndarray of shape (n_reps, )
            Normalized replica locations on the string. 

        Returns
        -------
        chain_vec_reparam: numpy.ndarray of shape (n_reps, )
            The reparametrized chain vector.
        """
        # Reset the init flag and interpolators. 
        self._init_reparam = True
        self._intpol_list = []

        for i_dof in range(chain_vec.shape[1]):
            # Deep copy an interpolator object and fit the interpolator.
            intpol = copy.deepcopy(self._intpol_instnce)
            intpol.fit(alpha_vec, chain_vec[:, i_dof])

            self._intpol_list.append(intpol)

            # Equi-arclength locations of each replica (new_alpha_vec)
            new_alpha_vec = numpy.linspace(0., 1., num=chain_vec.shape[0])

            # Compute the redistributed x.
            chain_vec[:, i_dof] = intpol.transform(new_alpha_vec)
        
        return chain_vec
    