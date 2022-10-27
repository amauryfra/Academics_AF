import numpy as np
from numpy import random


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    eig_values , _ = np.linalg.eig(np.dot(A.T,A)) # Retreiving eigen values of A^TA \
        # since norm 1 maximizers of ||Ax||^2 are eigen vectors of A^TA, and for them \
            # we have ||Ax|| = root(lambda).
    
    o2norm = max(np.abs(eig_values))**0.5 # ||A||^2 = ||A^TA||

    return o2norm

def norm_inequality_verif() :
    """
    Verifies that ||Ax|| <= ||A|| ||x||, where ||A|| is given by operator_2_norm.
    The verification is for various and random A and x. 
    ||Ax|| and ||x|| is given by Numpy 2-norm np.linalg.norm. 
    """
    
    for i in range(25) : # 25 tests
    
        m = random.randint(2,500) # Random m between 2 and 500
        n = random.randint(2,500) # Random n between 2 and 500
        
        A = random.randn(m, n) # Random matrix 
        x = random.randn(n) # Random vector x
        
        
        Ax = np.dot(A,x)
        
        norm_Ax = np.linalg.norm(Ax) # Computation of the useful norms 
        norm_A = operator_2_norm(A)
        norm_x = np.linalg.norm(x)
        
        assert(norm_Ax <= norm_A * norm_x) # Verification
        
        
def inequality_theorem_verif() :
    """
    Verifies that ||AB|| <= ||A|| ||B||, where ||AB||, ||A|| and ||B|| are given 
    by operator_2_norm. The verification is for various and random A and B. 
    """
    for i in range(25) : # 25 tests
    
        m = random.randint(2,500) # Random m between 2 and 500
        n = random.randint(2,500) # Random n between 2 and 500
        l = random.randint(2,500) # Random n between 2 and 500
        
        A = random.randn(l, m) # Random matrix 
        B = random.randn(m, n) # Random matrix 
        
        
        norm_AB = operator_2_norm(np.dot(A,B)) # Computation of the useful norms 
        norm_A = operator_2_norm(A)
        norm_B = operator_2_norm(B)
        
        assert(norm_AB <= norm_A * norm_B) # Verification

def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :param A: an mxn-dimensional numpy array

    :return ncond: the condition number
    """

    eig_values , _ = np.linalg.eig(np.dot(A.T,A)) # Retreiving eigen values of A^T A 

    A_o2norm = max(np.abs(eig_values))**0.5 # Same process as in operator_2_norm
    
    Ainv_o2norm = (1 / min(np.abs(eig_values)))**0.5 # If lambda is an eigen value of B than \
        # 1/lambda is an eigen value of B^-1
    
    ncond = A_o2norm * Ainv_o2norm

    return ncond

# Exercise 3.20
p = 12345678
q = 1
x1 = p - (p**2 + q)**(1/2)
x2 = - q / (p + (p**2 + q)**(1/2))

p1 = x1**2 - 2 * p * x1 - q # Approximately 0.011
p2 = x2**2 - 2 * p * x2 - q # Outputs 0.0



