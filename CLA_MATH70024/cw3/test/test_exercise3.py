"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 3

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 3

This script tests the functions created in
the exercise 3, given in the Coursework 3 of the module MATH70024.

"""
import pytest
import cla_utils
import cw3
import numpy as np
import copy
import scipy
from numpy import random

# Tests tridiagonal reduction of A
def test_tridiagonalA() :
    
   A = cw3.buildNewA()  
   T = copy.deepcopy(A)
    
   T = scipy.linalg.hessenberg(T)
   m = T.shape[0]
    
   i1 = np.tril_indices(m, k=-2)
   assert (np.linalg.norm(T[i1]) < 1e-6)
   assert (np.linalg.norm(T.transpose()[i1]) < 1e-6)
   
   
# Tests question (3c)
@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('n', [3, 5, 7]) # Works for small matrix
def test_customTridiagQR(n) :

    # Random symmetric matrix
    A = random.uniform(size=(n,n))
    A = (A + A.T)/2
    
    eigval, _ = cw3.customTridiagQR(A, maxit = 1000)
    
    for val in eigval :
         assert(np.abs(np.linalg.det(A - val * np.eye(n))) < 1.0e-4)
         
         

# Pure QR 'weekly exercises' testing adapted to the Wilkinson shift implementation
@pytest.mark.parametrize('m', [5, 10, 20])
def test_pure_QR(m):
    random.seed(1302*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2 = cla_utils.pure_QR(A0, maxit=10000, tol=1.0e-6, shiftCondition=True)
    #check it is still Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    #check for upper triangular
    assert(np.linalg.norm(A2[np.tril_indices(m, -1)])/m**2 < 1.0e-5)
    #check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)
         
