# Unit test on row-wise vector operations. 
# Zilin Song, 16 Jun 2022

import pytest

import numpy 
from random import randint

from pycospath.utils import row_wise_proj, row_wise_cos

def test_input():
    """If non 2D arrays are used as input, expect a ValueError."""

    a = numpy.random.random_sample((randint(1, 9), randint(1, 9), randint(1, 9)))
    b = numpy.random.random_sample((randint(1, 9), randint(1, 9), randint(1, 9)))

    with pytest.raises(ValueError,
                       match="A and B must be 2D array."):
        row_wise_proj(a, b)
        
    with pytest.raises(ValueError,
                       match="A and B must be 2D array."):
        row_wise_cos(a, b)

    # 1D 
    a = a.flatten()
    b = b.flatten()

    with pytest.raises(ValueError,
                       match="A and B must be 2D array."):
        row_wise_proj(a, b)
        
    with pytest.raises(ValueError,
                       match="A and B must be 2D array."):
        row_wise_cos(a, b)

    """If arrays of different shape are used as input, expect a ValueError."""

    # Generate two 1D arrays of different shapes. 
    n_dof = numpy.random.choice(numpy.arange(1, 999), size=4, replace=False)
    a = numpy.random.random_sample((n_dof[0], n_dof[1]))
    b = numpy.random.random_sample((n_dof[2], n_dof[3]))

    with pytest.raises(ValueError, 
                       match="A and B should have identical shapes."):
        row_wise_proj(a, b)

    with pytest.raises(ValueError, 
                       match="A and B should have identical shapes."):
        row_wise_cos(a, b)

    # Make any 2
    n_dof = numpy.random.choice(numpy.arange(1, 999), size=2, replace=False)
    a = numpy.random.random_sample((n_dof[0], n_dof[1]))
    b = numpy.random.random_sample((n_dof[0], n_dof[1]))
    
    try:
        row_wise_proj(  a*randint(1, 10), b*randint(1, 10))
        row_wise_cos(a*randint(1, 10), b*randint(1, 10))
    except Exception as e:
        assert False,                                                          \
            f"2D arrays of shape ({n_dof[0]}, {n_dof[1]}) should not raise {e}."
    
def test_row_wise_outputs():
    """Perform the following tests. 
    
    Test the projections of row vectors in array A onto those in array B.
    
    For each row,
    1. Get the projected vector;
    2. Check if the vector minus the projected vector could produce a vector 
       that is perpendicular to the reference vector.

    Test the cosine values of row vectors in arrays A and B. This test depends
    on if the row_wise_projection tests are correct.

    For each row,
    1. Get the projection of B onto A; 
    2. compute the ratio of projection / B, check the ratio with cosine. 
    """

    # ---------------
    # Test proj_row_wise(). 
    n_dof = randint(3, 100)
    n_vec = 10
    scaling_a = randint(1, 20)
    scaling_b = randint(1, 20)
    
    # Make the two vectors.
    A = (numpy.random.random_sample((n_vec, n_dof)) - .5) * scaling_a
    b = (numpy.random.random_sample((n_vec, n_dof)) - .5) * scaling_b

    # Get projected b.
    proj_A_b = row_wise_proj(A, b)

    # Get perpendicular component of b.
    perp_A_b = b - proj_A_b

    # Diagonal terms should be zeros.
    abs_diags = numpy.abs(numpy.diag(perp_A_b @ A.T))
    max_abs_diags = numpy.max(abs_diags)
    
    assert max_abs_diags <= 1e-12,                                             \
        f"Projection failed. Reference A = {A}; To-be-projected b = {b}; "     \
        f"Projected Proj_A_b = {proj_A_b}; (b - proj_A_b) = {perp_A_b}\n"      \
        f"Note that the 0 diagonal of (b - proj_A_b) @ A.T should be zero. "   \
        f"But diag = {abs_diags}."

    # ---------------
    # Test cosine_row_wise. 
    check_proj_row_wise = max_abs_diags <= 1e-12

    if check_proj_row_wise == False:
        assert False,                                                          \
            "Test on cosine_row_wise() is not performed due to the failure of "\
            "preceding assert on proj_row_wise() results."
    
    n_dof = randint(3, 100)
    n_vec = 10
    scaling_a = randint(1, 20)
    scaling_b = randint(1, 20)

    # Make the two vectors.
    A = (numpy.random.random_sample((n_vec, n_dof)) - .5) * scaling_a
    b = (numpy.random.random_sample((n_vec, n_dof)) - .5) * scaling_b

    proj_A_b = row_wise_proj(A, b)

    # Account for when cosine is larger than 180 degrees.
    signs = numpy.sign(proj_A_b[:, 0] * A[:, 0]) 

    norm_proj_A_b = numpy.linalg.norm(proj_A_b, axis=1) * signs
    norm_b        = numpy.linalg.norm(b,        axis=1)
    cosn_true = norm_proj_A_b / norm_b

    cosn_A_b = row_wise_cos(A, b)

    cosn_diff = cosn_A_b - cosn_true

    assert numpy.max(numpy.abs(cosn_diff)) <= 1e-12,                           \
        f"Row-wise cosine calculation failed. Reference cosn = {cosn_true}; "  \
        f"Computed cosn = {cosn_A_b}; Deviations = {cosn_diff}; Max dev. = "   \
        f"{cosn_diff} too large. "