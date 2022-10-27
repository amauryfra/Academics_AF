"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 4

This script tests the functions created in
the exercise 4, given in the Coursework 1 of the module MATH70024.

"""

import pytest
import cla_utils
import cw1
from numpy import random
import numpy as np

# Tests the computation of the staionnary point
@pytest.mark.parametrize('m, n', [(20, 7), (40, 13), (87, 9)])
def test_x_computation(m,n) :
    
    random.seed(1878*m)
    A = random.randn(m, n)
    b = random.randn(m,1)
    l = random.uniform(low=-10000,high=10000)
    
    x = cw1.x_computation(A,b,l)
    
    dPhi_dx = 2 * np.dot(np.dot(A.T,A), x) - 2 * np.dot(A.T, b) + 2 * l * x

    assert(np.linalg.norm(dPhi_dx)<1.0e-5) # Checking if x is a stationnary point


# Tests that lambda given by lambda_search provides an x that is such that ||x||
@pytest.mark.parametrize('m, l', [(20, 0.1), (40, 10), (87, 100)])
def test_lambda_search(l,m) :
    
    random.seed(1878*m)
    A = random.randn(m, m)
    b = random.randn(m,1)
    
    lbda = cw1.lambda_search(l,A,b)

    x = cw1.x_computation(A,b,lbda)
    
    assert(abs(np.linalg.norm(x)-1)<1.0e-5 ) # Checking ||x|| = 1
