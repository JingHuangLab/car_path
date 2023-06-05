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
from pycospath.utils import row_wise_cos

class ConstAdvRep(ConsCosBase):
    """The Constant Advancement Replicas method."""

    def __init__(self, 
                 chain:                 ChainBase, 
                 verbose:               bool = True, 
                 cons_thresh:           float = 1e-8, 
                 cons_maxitr:           int = 10, 
                 cons_curvreglr_thresh: int = 30,
                 cons_stepreglr_thresh: float = 1.25, 
                 cons_stepreglr_dnscal: float = .8, 
                 cons_log_writer:       TextIOWrapper = None):
        """Create a Constant Advancement Replicas calculation. 

        Parameters
        ----------
        chain: ChainBase
            The chain of replicas. 

        verbose: bool, default=True
            If information related to the calculation should be verbose. 
        
        cons_thresh: float, default=1e-8
            The threshold under which the holonomic constraints on equal RMS 
            distances b/w adjacent replicas are considered converged. 

        cons_maxitr: int, default=10
            The maximum no. iterations that the Lagrangian multiplier solver is
            allowed to loop to search for the constrained chain vector. 
        
        cons_curvreglr_thresh: int, default=30
            The threshold (in degrees) above which the curvature regularization
            becomes effective for the chain vector updates. Allowed values are 
            (0, 45)
            It explicitly controls the maximum angle allowed b/w the two vectors
            r_{i-1}-r_{i+1} and r{i-1}-r{i} of any three neighboring replicas on 
            the path. 
        
        cons_stepreglr_thresh: float, default=1.25 
            The threshold above which the regularization protocol for preventing 
            overstepping of Lagrangian multipliers becomes effective for the 
            chain vector updates. Allowed values are (1., 1.8].
            This regularization protocol explicitly determines if the Lagrangian 
            multiplier updates have overstepped from references to the previous 
            step and applies a Line-Search protocol to determine a set of
            Lagrangian multipliers with reasonable/regularized step size:
            If the mean-RMS of the chain-of-replicas after the coordinate update
            is higher by {cons_stepreglr_thresh} times mean-RMS before the
            coordinate update, the current Lagrangian multipliers are determined 
            to be overstepping, and is uniformly scaled down by a factor of
            {cons_stepreglr_dnscal}. 
            
        cons_stepreglr_dnscal: float, default=.8
            The down-scaling factor for the step size regularizing Line-Search. 
            The Lagrangian multipliers were scaled down by this factor until a
            proper step size is determined. Allowed values are (0., 1.).

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

        # The regularization threshold on the path curvature. 
        if cons_curvreglr_thresh <= 0. or cons_curvreglr_thresh > 45.:
            raise ValueError("The threshold for curvature regularization must "
                             "be >0 and <=45 degrees.")
        else:
            self._cons_curvreglr_thresh =                                      \
                numpy.cos(cons_curvreglr_thresh / 180. * numpy.pi)

        # The regularization threshold on the Lagrangian multipliers step size. 
        if cons_stepreglr_thresh <= 1. or cons_stepreglr_thresh > 1.8:
            raise ValueError("The threshold for Lagrangian multiplier step size "
                             "regularization must be > 1. and < 1.8.")
        else:
            self._cons_stepreglr_thresh = cons_stepreglr_thresh

        # The downscale factor for Lagrangian multiplier stepsize down search. 
        if cons_stepreglr_dnscal <= 0. or cons_stepreglr_dnscal >= 1.:
            raise ValueError("The downscaling factor for Lagrangian multiplier "
                             "step size must be > 0 and < 1.")
        else:
            self._cons_stepreglr_dnscal = cons_stepreglr_dnscal
        
        # Logger for lagrangian multiplier calculations.
        if cons_log_writer is None:
            self._cons_is_log = False
            self._cons_logger = None
        else:
            self._cons_is_log = True
            self._cons_logger = cons_log_writer
    
    def _get_path_tan(self,
                      chain_vec: numpy.ndarray
                      ) -> numpy.ndarray:
        """Get the path tangent vector at each replica along the chain. 

        Here: Approximated tangent. 
            r_{i-1} - r_{i+1} at replica i
            for all replicas along the path (except end-point replicas).
            The tangent of the first and the last replica is manually set to be 
            the linear tangent.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector on which the path tangent vectors are computed.

        Returns
        -------
        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs).
            The path tangent vector at each replica along the chain.
        """
        path_tan_vec = ConsCosBase._get_path_tan(self, chain_vec)
        path_tan_vec[1:-1] = chain_vec[0:-2] - chain_vec[2:  ]
        return path_tan_vec

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

        # Constraint criteria
        chain = self.get_chain()

        # Control variables for the Lagrangian solver.
        cons_cvg = 999.   # The convergence criteria.
        cons_itr = 0      # A counter for iteration steps.

        # Iteratively solve for the Lagrangian multipliers and update the chain
        # vector.
        while cons_cvg >= self._cons_thresh and cons_itr < self._cons_maxitr:
            
            # Regularized chain_vec for stable lcoef_mat inversion.
            chain_vec = self._cons_regulr(chain_vec)

            # The path tangent vector which accounts for the path curvature. 
            # Mandatory, must scale chain_vec by get_wscale_vec to compute path tangents. 
            path_tan_vec  = self._get_path_tan(chain_vec)
            # The RMS and RMS gradients at the current step. 
            rms, rms_grad = chain.get_rms(chain_vec, return_grads=True)
            # Objective distances on the rms. 
            rms_mean = numpy.mean(rms)

            # Computes the coefs on the Lagrangian multipliers and solve for the
            # Lagrangian multipliers.
            # l_coef is a coefficient matrix of shape (n_reps-1, n_reps-2)
            l_coef = self._cons_get_lcoef(rms_grad, path_tan_vec)

            # Apply reduced mode of QR decomposition to create:
            # q_l_coef:      orthogonal matrix of shape (n_reps-1, n_reps-2)
            # r_l_coef: upper-triangular matrix of shape (n_reps-2, n_reps-2)
            q_l_coef, r_l_coef = numpy.linalg.qr(l_coef, mode='reduced')
            
            # Solve for the lambdas.
            l_val = numpy.linalg.solve(r_l_coef, q_l_coef.T @ (rms_mean - rms))

            # Evolve the chain towards the constraint conditions and iterate the
            # chain_vec.
            chain_vec = self._cons_update(chain_vec, path_tan_vec, l_val)
            
            # Determine if converged.
            rms_nx   = chain.get_rms(chain_vec)
            cons_cvg = numpy.max(numpy.abs(rms_mean - rms_nx))
            cons_itr += 1
        
        if self.is_verbose():
            print(f"itr_steps: {cons_itr} {cons_cvg} {rms_mean}")
        
        if self._cons_is_log:
            self._cons_logger.write(
                f"itr_steps: {cons_itr} {cons_cvg} {rms_mean}\n")
            self._cons_logger.flush()
            
        return chain_vec
        
    def _cons_regulr(self,
                     chain_vec: numpy.ndarray
                     ) -> numpy.ndarray:
        """Regularize the chain vector to prevent kinks along the path. At the 
        i-th replica, this method computes the angle b/w r_{i-1} - r_{i+1} and 
        r_{i-1} - r_{i}. 
        If this angle is larger than the threshold, r_{i} is directly set to the
        mid-point of r_{i-1} and r_{i+1}. 
        Regularization is only done on the intermediate replicas.

        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to regularize.

        Returns
        -------
        regulr_chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The regularized chain vector. 
        """
        # Cosine: the metrics determining if regularization should be applied.
        diff_1 = chain_vec[0:-2] - chain_vec[1:-1] # The r_{i-1} - r_{i}   vecs.
        diff_2 = chain_vec[0:-2] - chain_vec[2:  ] # The r_{i-1} - r_{i+1} vecs.
        cosine = row_wise_cos(diff_1, diff_2)

        # Regularization condition.
        i_rep_regulr = numpy.where(cosine <= self._cons_curvreglr_thresh)[0]

        if i_rep_regulr.size != 0: # Regularization in effect. 
            chain_vec_mid = (chain_vec[0:-2] + chain_vec[2:  ]) / 2.
            chain_vec[i_rep_regulr+1] = chain_vec_mid[i_rep_regulr]

        return chain_vec

    def _cons_get_lcoef(self, 
                        rms_grad_cr:  numpy.ndarray, 
                        path_tan_vec: numpy.ndarray
                        ) -> numpy.ndarray:
        """Compute the coefficient matrix for the Lagrangian lambdas.
        NOTE: no lambda for the last replica.

        Parameters
        ----------
        rms_grad_cr: numpy.ndarray of shape (n_reps-1, n_dofs)
            The RMS gradients for the (n)-th step. 

        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The tangent vector of the path.

        Returns
        -------
        lcoef_mat: numpy.ndarray of shape (n_reps-1, n_reps-1)
            The coefficient matrix of lambdas.
        """
        # number of lambdas = n_reps-2 (with both end points fixed).
        n_l = rms_grad_cr.shape[0] - 1

        # Coefficient on the gradients of chain vector w.r.t. lambdas.
        B =  1. * path_tan_vec[1:n_l+1]         # gradients on r_{i}
        C = -1. * path_tan_vec[1:n_l+1]         # gradients on r_{i+1}
        
        # number of lambdas = n_reps-1
        # n_l = rms_grad_cr.shape[0]

        # # Coefficients on the gradients of chain vector w.r.t. the lambdas.
        coef_l_       =  1. * path_tan_vec[1:n_l+1]
        coef_l_plus_1 = -1. * path_tan_vec[1:n_l+1]
        
        # Computes the coefficient matrix for the lambdas.
        ondiag = numpy.sum(rms_grad_cr[0:-1] * coef_l_plus_1, axis=1)
        lwdiag = numpy.sum(rms_grad_cr[1:  ] * coef_l_,       axis=1)

        # Fills the coef in mat with the shape (n_reps-1, n_reps-2)
        lcoef_mat = numpy.zeros((n_l+1, n_l)) 
        lcoef_mat.ravel()[  0::n_l+1] = ondiag  # lambdas on (i+1)-th replicas.
        lcoef_mat.ravel()[n_l::n_l+1] = lwdiag  # lambdas on   (i)-th replicas.

        return lcoef_mat

    def _cons_update(self, 
                     chain_vec_cr:    numpy.ndarray,
                     path_tan_vec:    numpy.ndarray,
                     lambda_val:      numpy.ndarray,
                     lambda_regulr: bool = True
                     ) -> numpy.ndarray:
        """The update equation of chain_vec_cr towards the RMS constraints.
        NOTE: no updates for the first and the last replica.

        Parameters
        ----------
        chain_vec_cr: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector for the (n)-th step

        path_tan_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The path tangent vector at each replica along the chain..
            
        lambda_val: numpy.ndarray of shape (n_reps-2, )
            The vector of the values of Lagrangian lambdas for all intermediate
            replicas.
        
        Returns
        -------
        chain_vec_nx: numpy.ndarray of shape (n_reps-1, n_dofs)
            The updated chain vector of the (n+1)-th step.
        """
        # Regularize the step size of lambda updates to prevent extremely large 
        # coordinate update steps from taken, which leads to possible switch of 
        # replicas' geometrical order. That is, for one or some replicas, their
        # neighboring replicas are no longer the nearest neighbors, causing the
        # formation of "circular" path segments, which prevents sequential coor
        # updates to converge. In almost all cases, we see that extremely large 
        # values of lambdas would appear and the coordinates would become np.inf
        # and then some values are NaN, then a SVD-related exception would be 
        # raised from numpy. 

        # Currently, the regularization scheme is implemented as a line search
        # protocol that finds lambdas does not grow the mean rms length. 
        rms = self.get_chain().get_rms(chain_vec_cr)
        lambda_regulr = True
        
        while lambda_regulr == True:
            # 1. Compute the new mean_rms. 
            chain_vec_trial = numpy.copy(chain_vec_cr)
            chain_vec_trial[1:-1] += path_tan_vec[1:-1] * lambda_val[:, None]
            rms_trial = self.get_chain().get_rms(chain_vec_trial)

            # 2. Constraint not met and at least one step size is larger than the 
            # cost. 
            if  (rms * self._cons_stepreglr_thresh < rms_trial).any():
                lambda_regulr = True
                lambda_val *= self._cons_stepreglr_dnscal # scale lambdas down. 
            else:
                lambda_regulr = False

        # Update only the intermediate replicas vectors. 
        # NOTE: the index of lambdas correspond to the (i+1)-th replica
        chain_vec_nx = numpy.copy(chain_vec_cr)
        chain_vec_nx[1:-1] =   chain_vec_cr[1:-1]                              \
                             + path_tan_vec[1:-1] * lambda_val[:, None]

        return chain_vec_nx
    