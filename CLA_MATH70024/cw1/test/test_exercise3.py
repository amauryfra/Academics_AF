"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 3 

This script tests the functions created in
the exercise 3, given in the Coursework 1 of the module MATH70024.

"""

import pytest
import cla_utils
import cw1
from numpy import random
import numpy as np
import copy

# Tests the storage of v vectors
@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_Rv_householder(m,n):
    random.seed(1878*m)
    A = random.randn(m, n)
    A0 = copy.deepcopy(A)
    Rv = cw1.Rv_householder(A0)
    R = np.triu(Rv) # Getting rid of the v column vectors 
    assert(np.linalg.norm(np.dot(R.T, R) - np.dot(A.T, A)) < 1.0e-6)
    
    m = Rv.shape[0]
    
    # Retreiving Q
    Q = np.identity(m, dtype=float)
    for k in range(n) :
        v = Rv[k:,k] / np.linalg.norm(Rv[k:,k])
        Q[k:,:] = Q[k:,:] - 2 * np.outer(v,np.dot(np.conjugate(v),Q[k:,:]))
    Q = Q.transpose()

    # Checking orthonormality
    assert(np.linalg.norm(np.dot(Q.transpose(), Q) - np.identity(m,dtype=float)) < 1.0e-6)
    # Checking QR factorisation
    assert(np.linalg.norm(np.dot(Q, R) - A) < 1.0e-6)

    
# Tests the Q^*b computation
@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_Rv_multiplication(m,n):
    
    random.seed(1878*m)
    A = random.randn(m, n)
    A0 = copy.deepcopy(A)  
    Rv = cw1.Rv_householder(A0)
    
    b = random.randn(m)
    b0 = copy.deepcopy(b)
    
    Qb = cw1.Rv_multiplication(Rv,b) # Computing Q^*b
    
    Q,_ = np.linalg.qr(A,mode='complete') # Numpy QR factorisation
    Qb_true = np.dot(Q.transpose().conjugate(),b0) # Computing expected Q^*b
    
    assert(np.linalg.norm(Qb_true-Qb)<1.0e-6)
    
    
    
# Tests the least squares solver
@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_Rv_ls(m,n) :
    
    random.seed(1878*m)
    A = random.randn(m, n)
    A0 = copy.deepcopy(A)  
    Rv = cw1.Rv_householder(A0)
    
    b = random.randn(m)
    b0 = copy.deepcopy(b)
    
    x_true = np.linalg.lstsq(A, b,rcond=-1)

    x = cw1.Rv_ls(Rv,b0)

    assert(np.linalg.norm(x-x_true[0])<1.0e-6) # Checking if x is a least squares minimizer
    
    
    
    
    
    
    
    
    
    