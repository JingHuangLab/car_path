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
from io import TextIOWrapper

from pycospath.cos   import ConsCosBase
from pycospath.chain import ChainBase

class ReaxPathCons(ConsCosBase):
    """The Reaction Path with Holonomic Constraints method:

    Reaction Path Optimization with Holonomic Constraints and Kinetic Energy 
    Potentials. 
    J. Brokaw, K. Haas, J.-W. Chu, J. Chem. Theory Comput. 2008, 5 2050-2061
    10.1021/ct9001398

    The kinetic energy potential proposed in this work was not implemented. 
    """

    def __init__(self, 
                 chain:           ChainBase, 
                 verbose:         bool = True, 
                 cons_thresh:     float = 1e-8, 
                 cons_maxitr:     int = 200, 
                 cons_log_writer: TextIOWrapper = None):
        """Create a Reaction Path with Holonomic Constraints calculation.

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        
        cons_thresh: float, default=1e-8
            The threshold under which the holonomic constraints on equal RMS 
            distances b/w adjacent replicas are considered converged. 

        cons_maxitr: int, default=200
            The maximum no. iterations that the Lagrangian multiplier solver is
            allowed to loop to search for the constrained chain vector. 

        cons_log_writer: TextIOWrapper, default=None
            A TextIOWrapper object to log the Lagrangian update steps, it must
            implement a write() method for echoing information. None means no 
            logger. 
        """
        ConsCosBase.__init__(self, chain, verbose)

        # The threshold for the convergence of the holonomic constraints.
        self._cons_thresh = cons_thresh
        # The max iteration allowed for solving holonomic constraints.
        self._cons_maxitr = cons_maxitr
    
        # Logger for lagrangian multiplier calculations.
        if cons_log_writer == None:
            self._cons_is_log = False
            self._cons_logger = None
        else:
            self._cons_is_log = True
            self._cons_logger = cons_log_writer

    def _get_cons_chain_vec(self, 
                            chain_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Determine and return the chain vector that satisfies the constraint
        condition. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the constraint condition.

        Returns
        -------
        constr_chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector that satisfies the constraint condition.
        """
        # Initialize: 
        #         pv: the (n-1)-th step; 
        #         cr: the (n  )-th step; 
        #         nx: the (n+1)-th step.
        chain_vec_cr = numpy.copy(chain_vec)
        chain_vec_pv = numpy.copy(chain_vec)
        chain_vec_nx = numpy.zeros(chain_vec.shape)
        
        # Constraint criteria
        chain = self.get_chain()
        rms_cons = numpy.mean(chain.get_rms(chain_vec_cr))

        # Control variables for the Lagrangian solver.
        cons_cvg = 999.   # The convergence criteria.
        cons_itr = 0      # Logger for iteration steps.

        # Iteratively solve for the Lagrangian multipliers and update the chain
        # vector. 
        while cons_cvg >= self._cons_thresh and cons_itr < self._cons_maxitr:

            # The RMS and RMS gradients at the   (n)-th step.
            rms_cr, rms_grad_cr = chain.get_rms(chain_vec_cr, return_grads=True)
            # The RMS and RMS gradients at the (n-1)-th step.
            _,      rms_grad_pv = chain.get_rms(chain_vec_pv, return_grads=True)

            # Computes the tridiagonal coefs on the Lagrangian multipliers and 
            # solve for the Lagrangian multipliers.
            # l_coef is a coefficient matrix of shape (n_reps-1, n_reps-1)
            l_coef = self._cons_get_lcoef(rms_grad_cr, rms_grad_pv)

            # Cost of constraints: 
            cons_cost = rms_cons - rms_cr

            # Solve for the lambdas
            l_val  = numpy.linalg.solve(l_coef, cons_cost)

            # Evolve the chain vector towards the constraining condition.
            chain_vec_nx = self._cons_update(chain_vec_cr, rms_grad_cr, l_val)

            # Iterate chain_vecs. 
            chain_vec_pv = numpy.copy(chain_vec_cr)
            chain_vec_cr = numpy.copy(chain_vec_nx)
            
            # Determine if converged.
            rms_nx  = chain.get_rms(chain_vec_cr)
            cons_cvg = numpy.max(numpy.abs(rms_nx - rms_cons))
            cons_itr += 1

        if self.is_verbose():
            print(f"itr_steps: {cons_itr} {cons_cvg} {rms_cons}")
            
        if self._cons_is_log:
            self._cons_logger.write(
                f"itr_steps: {cons_itr} {cons_cvg} {rms_cons}\n")
            self._cons_logger.flush()

        return chain_vec_cr

    def _cons_get_lcoef(self, 
                        rms_grad_cr: numpy.ndarray, 
                        rms_grad_pv: numpy.ndarray
                        ) -> numpy.ndarray:
        """Compute the coefficient matrix for the Lagrangian lambdas.
        
        Parameters
        ----------
        rms_grad_cr: numpy.ndarray of shape (n_reps-1, n_dofs)
            The RMS gradients for the (n)-th step. 

        rms_grad_pv: numpy.ndarray of shape (n_reps-1, n_dofs)
            The RMS gradients for the (n-1)-th step.

        Returns
        -------
        lcoef_mat: numpy.ndarray of shape (n_reps-1, n_reps-1)
            The coefficient matrix of lambdas.
        """
        # no. lambdas = no. RMS distances (one lambda per RMS distance).
        n_l = rms_grad_cr.shape[0]  

        # Coefficients on the gradients of chain vector w.r.t. the lambdas.
        coef_l_minus_1 = -1. * rms_grad_pv[0:n_l-1]
        coef_l_        =  2. * rms_grad_pv[ :     ]
        coef_l_plus_1  = -1. * rms_grad_pv[1:     ]

        # Computes the tridiagonal matrix coefficients for the lambdas.
        lodiag = numpy.sum(rms_grad_cr[1:  ] * coef_l_minus_1, axis=1)
        ondiag = numpy.sum(rms_grad_cr[ :  ] * coef_l_,        axis=1)
        updiag = numpy.sum(rms_grad_cr[0:-1] * coef_l_plus_1 , axis=1)

        # Fills the coef in mat with the shape (n_reps-1, n_reps-1)
        lambda_coef = numpy.zeros((n_l, n_l))
        lambda_coef.ravel()[n_l::n_l+1] = lodiag # -1-off diagonal
        lambda_coef.ravel()[  0::n_l+1] = ondiag #  0-on  diagonal
        lambda_coef.ravel()[  1::n_l+1] = updiag # +1-off diagonal

        return lambda_coef

    def _cons_update(self, 
                     chain_vec_cr: numpy.ndarray, 
                     rms_grad_cr:  numpy.ndarray, 
                     lambda_val:   numpy.ndarray
                     ) -> numpy.ndarray:
        """The update equation of chain_vec_cr towards the RMS constraints.
        NOTE: no updates for the first and the last replica.

        Parameters
        ----------
        chain_vec_cr: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector for the (n)-th step

        rms_grad_cr: numpy.ndarray of shape (n_reps-1, n_dofs)
            The RMS gradients for the (n)-th step. 
            
        lambda_val: numpy.ndarray of shape (n_reps-1, )
            The vector of the values of Lagrangian lambdas.
        
        Returns
        -------
        chain_vec_nx: numpy.ndarray of shape (n_reps, n_dofs)
            The updated chain vector of the (n+1)-th step.
        """
        chain_vec_nx = numpy.copy(chain_vec_cr)

        # Update only the intermediate replicas vectors. 
        # NOTE: the index of lambdas correspond to the n-th RMS distance. 
        chain_vec_nx[1:-1] =   chain_vec_cr[1:-1]                              \
                             - lambda_val[0:-1, None] * rms_grad_cr[0:-1]      \
                             + lambda_val[1:  , None] * rms_grad_cr[1:  ]
        return chain_vec_nx
