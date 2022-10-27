"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 3

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 2

This script tests the functions created in
the exercise 2, given in the Coursework 3 of the module MATH70024.

"""

import pytest
import cla_utils
import cw3
import numpy as np
import copy

# Tests the custom eigenvalue computation algorithm of A
@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('n', [3, 10, 15])
def test_getEigenvalueA(n):
    
    A = cw3.buildA(n)
    A0 = copy.deepcopy(A)
    eigval = cw3.getEigenvalueA(A)
    
    for val in eigval :
        assert(np.abs(np.linalg.det(A0 - val * np.identity(2*n))) < 1.0e-6)
        
    
    

# Tests the custom eigenvalue computation algorithm of B
@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('n', [3, 10, 15])
def test_getEigenvalueB(n):
    
    B = cw3.buildB(n)
    B0 = copy.deepcopy(B)
    eigval = cw3.getEigenvalueB(B)
    
    for val in eigval :
        assert(np.abs(np.linalg.det(B0 - val * np.eye(2*n))) < 1.0e-5)
        
        

        