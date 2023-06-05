# Unit test on Cartesian implementation of Kabsch algorithm to best-fit two
# set of points via rigid rotational and translational transform. 
# Zilin Song, 16 Jun 2022

import pytest

import numpy
from random import randint

from pycospath.utils import rigid_fit

def test_inumpyut():
    """If non-1D arrays are used as inumpyut, expect a ValueError."""
    
    # Make any 2D array.
    a = numpy.random.random_sample((randint(1, 9), randint(1, 9)))

    with pytest.raises(ValueError, 
                       match="A_raw and B_raw must be flattened 1D arrays."):
        rigid_fit(a, a)

    """If arrays of different shape are used as inumpyut, expect a ValueError."""

    # Generate two 1D arrays of different shapes. 
    n_dof = numpy.random.choice(numpy.arange(1, 999), size=2, replace=False)
    a = numpy.random.random_sample(n_dof[0]*3)
    b = numpy.random.random_sample(n_dof[1]*3)

    with pytest.raises(ValueError, 
                       match="A_raw and B_raw should have identical shapes."):
        rigid_fit(a, b)
        
    """If 1D arrays were used as inumpyut, no errors should be raised."""

    # Inumpyut 1D array of 3*n_dof elements so that no exception should be raised.
    n_dof = randint(1, 20)*3
    a = (numpy.random.random_sample(n_dof) - .5)
    
    try:
        rigid_fit(a*randint(1, 10), a*randint(1, 10))
    except Exception as e:
        assert False, f"1D arrays of shape ({n_dof}, ) should not raise {e}."
    
def test_3d_best_fit():
    """If the rigid best fit outputs correct results. """
    # 1. Make 1D array of shape (3*n_dofs);
    # 2. Reshape it to 2D of shape (n_dofs, 3);
    # 3. Rotate and translate it with synthetic matrices;
    # 4. Inumpyut the transformed and original array to rigid_best_fit to see
    #    if the original array is reproduced;
    # 5. Test if the rotational and translational matrix is desired.

    ro_raw = numpy.random.random_sample((3, 3))  # Covariance of ref and raw.
    V, S, W = numpy.linalg.svd(ro_raw)
    V[:, -1] = numpy.sign((numpy.linalg.det(V @ W))) * V[:, -1]
    ro_true = V @ W

    # Make good translational matrix.
    tr_raw = numpy.random.random_sample((3, ))

    # Make A as 3D point cloud.
    A: numpy.ndarray = numpy.random.random_sample((numpy.random.randint(4, 1000), 3))
    # Rotate and translate to make B.
    B = A @ ro_true + tr_raw
    
    tr_true = numpy.mean(A, axis=0)
    B_test, ro_test, tr_test = rigid_fit(A.flatten(), B.flatten())

    assert numpy.allclose(A.flatten(), B_test, atol=1e-12)

    assert numpy.allclose(ro_test @ ro_true, numpy.identity(3), atol=1e-12)

    assert numpy.allclose(tr_test, tr_true, atol=1e-12)
