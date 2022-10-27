"""
Imperial College London
Computational Linear Algebra (MATH70024)
Mastery component

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

This script tests the functions created in
the mastery.py file, provided for the Mastery component of the module MATH70024.

"""

import pytest
import cla_utils
import mastery
import numpy as np
from numpy import random
import scipy
import copy
import warnings
warnings.filterwarnings("ignore")


# Tests the HSS method
@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('n', [5, 10, 15])
def test_HSS(n) :
    
    # Generating a random positive definite matrix
    d = random.randn(n)
    d = np.abs(d)
    D = np.diag(d) # Only strictly positive values on the diagonal
    B = random.uniform(size=(n,n))
    Q,_ = cla_utils.householder_qr(B) # Retreiving a random unitary matrix
    A = np.dot(Q.conj().T,np.dot(D,Q))
    assert(mastery.verifyPositiveDefinite(A))
    
    b = random.uniform(size=(n,))
    
    x, nits = mastery.HSS(A, b, alpha = 10, x0 = None, tol = 1e-6, maxit = 10000) 
    
    assert(nits != -1)
    assert(np.linalg.norm(np.dot(A,x) - b) < 1e-6)

# Tests the IHSS method
@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('n', [5, 10, 15])
def test_IHSS(n) :
    
    # Generating a random positive definite matrix
    d = random.randn(n)
    d = np.abs(d)
    D = np.diag(d) # Only strictly positive values on the diagonal
    B = random.uniform(size=(n,n))
    Q,_ = cla_utils.householder_qr(B) # Retreiving a random unitary matrix
    A = np.dot(Q.conj().T,np.dot(D,Q))
    assert(mastery.verifyPositiveDefinite(A))
    
    b = random.uniform(size=(n,))
    
    x, nits = mastery.IHSS(A, b, alpha = 10, x0 = None, tol = 1e-6, maxit = 10000) 
    
    assert(nits != -1)
    assert(np.linalg.norm(np.dot(A,x) - b) < 1e-6)
    
    
    