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

from pycospath.cos import ConsCimgBase, ConstAdvRep

class CimgConstAdvRep(ConsCimgBase):
    """Implements the Climbing Image Constant Advancement Replicas method."""

    def __init__(self,
                 cos: ConstAdvRep):
        """Create a Climbing Image Constant Advancement Replicas calculation.

        Parameters
        ----------
        cos: ConstAdvRep
            The instance of the Constant Advancement Replicas method to perform 
            CI. 
        """
        if not isinstance(cos, ConstAdvRep):
            raise TypeError("Constant Advancement Replicas Climbing Image "
                            "decorator must be initialized on instances of the "
                            "Constant Advancement Replicas method. ")
        
        ConsCimgBase.__init__(self, cos)

    def _get_cons_chain_vec(self, 
                            chain_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Decorated ConsCosBase class member. 
        Determine the constrained chain vector and return the chain vector that 
        satisfies the constraint condition. 

        Here: Divide the chain vector into two sub chain vectors at the highest 
              energy replica (the CI replica). Both parts includes the end-point
              and the CI replica. Note that since the Constant Advancement 
              Replicas method does not change the last replica vector during the
              solution of the constraints, the second sub chain is reversed in 
              as to prevent the replica vector updates of the CI replica. 
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The vector representation of the chain of replicas.

        Returns
        -------
        constr_chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector satisfying the constraint condition.
        """
        cimg_rep_i = self.get_i_cimg_rep()         # Index for CI replica.

        # Break the chain_vec at the CI replica and make two sub chain_vecs: 
        # chain_vec_pre and chain_vec_post. Note that chain_vec_post is flipped 
        # such that the CI replica is the last (Constant Advancement Replicas 
        # does not enforce constraints on the last replica). 
        chain_vec_pre  = numpy.copy(chain_vec[0         :cimg_rep_i+1, :])
        chain_vec_post = numpy.copy(chain_vec[cimg_rep_i:            , :])
        chain_vec_post = numpy.flip(chain_vec_post, axis=0)

        # Get constrained chain_vec. 
        cos: ConstAdvRep = self.get_cos()
        constr_chain_vec_pre  = cos._get_cons_chain_vec(chain_vec_pre)
        constr_chain_vec_post = cos._get_cons_chain_vec(chain_vec_post)
        constr_chain_vec_post = numpy.flip(constr_chain_vec_post, axis=0)

        # Set chain_vec results. 
        constr_chain_vec = numpy.zeros(chain_vec.shape)
        constr_chain_vec[0         :cimg_rep_i+1, :] = constr_chain_vec_pre
        constr_chain_vec[cimg_rep_i:            , :] = constr_chain_vec_post

        return constr_chain_vec
        