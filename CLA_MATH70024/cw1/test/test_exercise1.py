"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 1 - (b)

This script tests the function created in
the exercise 1, given in the Coursework 1 of the module MATH70024.

"""

import pytest
import cla_utils
import cw1
import numpy as np
import copy

# Tests the compression of C
def test_compress_C():
    
    C = np.loadtxt('./cw1/C.dat', delimiter=',') # Loading data
    C_copy = copy.deepcopy(C)
    
    _ , _ , C_compress = cw1.compress_C(C)
    
    assert(np.linalg.norm(C_copy-C_compress) < 1.0e-8) # Checking we can precisely retreive C