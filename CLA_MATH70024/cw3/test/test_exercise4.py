"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 3

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 4

This script tests the functions created in
the exercise 4, given in the Coursework 3 of the module MATH70024.

"""
import pytest
import cla_utils
import cw3
import numpy as np
import copy
import scipy
from numpy import random


# Tests serialization 
@pytest.mark.parametrize('n', [5, 15, 50])
def test_serialization(n) :
    A = random.uniform(size=(n,n))
    v = cw3.serialize(A)
    B = cw3.deserialize(v)
    
    assert(np.linalg.norm(A-B) < 1e-16)
    
# Tests H_apply function 
@pytest.mark.parametrize('n', [5, 15, 50])
def test_H_apply(n) :
    A = random.uniform(size=(n,n))
    mu, lbda = random.uniform(10), random.uniform(10)
    v = cw3.serialize(A)
    hv, lapSum = cw3.H_apply(v, mu, lbda, lapSumCondition = True)
    
    assert(int(hv.shape[0]**0.5) == n) # Testing size
    assert(lapSum < 1e-6) # Sum over entries needs to be rougly equal to zero
    
# Tests GMRES modification 
@pytest.mark.parametrize('n', [4, 9, 25])
def test_GMRES_Modification(n) :
    func = cw3.H_apply
    b = random.uniform(size=(n,))
    x, _ = cla_utils.GMRES(A = 0, b = b, maxit = 1000, tol = 1.0e-3, x0=None, \
                         return_residual_norms=False,return_residuals=False, func = func)
    b0 = func(x)
    assert(np.linalg.norm(b0 - b) < 1.0e-3)
    
    
# Tests GMRES modification with preconditioning
@pytest.mark.parametrize('n', [9]) # Scipy source code of scipy.linalg.solve_banded \
    # seems to have an error, asserting that l + u + 1 == ab.shape[0] which is not the case\
       # of all banded matrix
def test_GMRES_Preconditioning(n) :
    A = random.uniform(size=(n,n))
    precondFunc = cw3.M_solve
    b = random.uniform(size=(n,))
    x, _ = cla_utils.GMRES(A = A, b = b, maxit = 200, tol = 1.0e-4, x0=None, \
                         return_residual_norms=False,return_residuals=False,\
                             func = None, precondFunc = precondFunc)

    assert(np.linalg.norm(np.dot(A,x)-b) < 1e-14)
    
    


    