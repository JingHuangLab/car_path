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

import numpy

from pycospath.cos import ConsCimgBase, StringMethod, StringMethodGradProj

class CimgStringMethod(ConsCimgBase):
    """Implements the Climbing Image Simplified String method:
    
    Simplified and improved string method for computing the minimum energy paths
    in barrier-crossing events.
    W. E, W. Ren, E. Vanden-Eijnden, J. Chem. Phys., 2007, 126, 164103.  
    10.1063/1.2720838

    Climbing Image is performed by inverting the parallel component of the grad
    vector for simplicity. 
    """

    def __init__(self,
                 cos: StringMethod):
        """Create a Climbing Image Simplified String Method calculation.

        Parameters
        ----------
        cos: StringMethod
            The instance of the Simplified String Method to perform CI. 
        """
        if not isinstance(cos, StringMethod):
            raise TypeError("String Method Climbing Image decorator must be "
                            "initialized on instances of the String Method.")
        
        if isinstance(cos, StringMethodGradProj):
            raise TypeError("Gradient projected String Method does not allow "
                            "Image Climbing.")

        ConsCimgBase.__init__(self, cos)

    def _get_cons_chain_vec(self, 
                            chain_vec: numpy.ndarray
                            ) -> numpy.ndarray:
        """Decorated ConsCosBase class member. 
        Determine the constrained chain vector and return the chain vector that 
        satisfies the constraint condition. 

        Here: Divide the chain vector into two sub chain vectors at the highest 
              energy replica (the CI replica). Both parts includes the end-point
              and the CI replica. Both sub chain vectors are then re-parametrized 
              independently without changing their end-points.
        
        Parameters
        ----------
        chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector to be imposed with the constraint condition and the
            climbing image protocol.

        Returns
        -------
        constr_chain_vec: numpy.ndarray of shape (n_reps, n_dofs)
            The chain vector satisfying the constraint condition.
        """
        # Index for the CI replica.
        cimg_rep_i = self.get_i_cimg_rep()

        # Break the chain_vec at the CI replica and make two sub chain_vecs: 
        # chain_vec_pre and chain_vec_post. 
        chain_vec_pre  = numpy.copy(chain_vec[0         :cimg_rep_i+1, :])
        chain_vec_post = numpy.copy(chain_vec[cimg_rep_i:            , :])
        
        # The constraints were imposed on the two sub chain_vecs, respectively.
        cos: StringMethod  = self.get_cos() 
        constr_chain_vec_pre  = cos._get_cons_chain_vec(chain_vec_pre)
        constr_chain_vec_post = cos._get_cons_chain_vec(chain_vec_post)

        # Set chain_vec results. 
        constr_chain_vec = numpy.copy(chain_vec)
        constr_chain_vec[0         :cimg_rep_i+1, :] = constr_chain_vec_pre
        constr_chain_vec[cimg_rep_i:            , :] = constr_chain_vec_post

        return constr_chain_vec
        