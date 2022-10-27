"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 2

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 4

This script tests the functions created in
the exercise 4, given in the Coursework 2 of the module MATH70024.

"""

import pytest
import cla_utils
import cw2
import numpy as np
import copy

# Tests the modified banded matrix algorithm
@pytest.mark.parametrize('n, eps', [(20, 0.1), (40, 0.05), (87, 0.01)])
def test_banded_LU_modified(n,eps):
    
    A = cw2.overlappingBlocksPrototype(n,eps) # Custom overlapping blocks \
        # random prototype matrix A
    A0 = copy.deepcopy(A)
    
    cw2.banded_LU_modified(A) # In place modified LU decomposition

    # Extracting L and U
    m =4 * n + 1
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)

    
    A1 = np.dot(L, U) # A1 = LU
    err = A1 - A0

    assert(np.linalg.norm(err) < 1.0e-6)

    
