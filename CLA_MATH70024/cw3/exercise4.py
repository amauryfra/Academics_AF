"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 3

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 4 - (a), (b), (d), (e)

This script computes the solutions for questions (a), (b), (d) and (e) of
the exercise 4, given in the Coursework 3 of the module MATH70024.

"""

import numpy as np
from numpy import random
import cla_utils
import copy
import scipy
import matplotlib.pyplot as plt
import scipy.linalg as la

# Question (a) 

def serialize(A) :
    """
    Computes the serialized vector of matrix A, i.e. the 1-D vector corresponding 
    to the flatten version of the inputed matrix.

    :param A: an mxm-dimensional numpy array
    
    :return v: an m^2-dimensional numpy array (1-D)
    """
    return(A.T.flatten())

def deserialize(v) :
    """
    Computes the matrix A from its serialized version. The function transforms
    the 1-D flatten vector back to the corresponding matrix.

    :param v: an m^2-dimensional numpy array (1-D)
    
    :return A: an mxm-dimensional numpy array
    """
    m = int(np.sqrt(v.shape[0])) # Retreiving matrix size
    return(v.reshape(m,m).T)


# Question (b)

def H_apply(v, mu = 1, lbda = 1, lapSumCondition = False) :
    """
    Computes the flatten version of matrix H = mu I + lambda A. 

    :param v: an m^2-dimensional numpy array (1-D)
    :param mu: a float, given parameter
    :param lbda: a float, given parameter
    
    :return hu: an m^2-dimensional numpy array 
    """
    
    U = deserialize(v) # Retreiving corresponding matrix
    
    m = U.shape[0] # Retreiving m
    
    A = np.zeros((m,m))
    
    # Negative Laplacian
    A[:-1,:] = U[:-1,:] - U[1:,:]
    A[1:,:] += U[1:,:] - U[:-1,:]
    A[:,:-1] += U[:,:-1] - U[:,1:]
    A[:,1:] += U[:,1:] - U[:,:-1] 
    
    lapSum = np.sum(A)
    
    # Scaling and adding muI 
    A *= lbda
    A += mu * U
    
    hu = serialize(A)
    
    if lapSumCondition :
        return(hu, lapSum)
    
    return(hu)
    

def M_solve(x0, mu = 1, lbda = 1) :
    """
    Applies one iteration of the sweeping algorithm given in question (4d)
    
    :param x0: an m^2-dimensional numpy array (1-D)
    :param mu: a float, given parameter
    :param lbda: a float, given parameter
    
    :return sol: an m^2-dimensional numpy array, the solution
    """
    
    X = deserialize(x0) # Corresponding matrix
    m = X.shape[0]
    
    T = 2 * np.identity(m) 
    T += np.diag([-1 for s in range(m-1)], k = 1)
    T += np.diag([-1 for s in range(m-1)], k = -1)
    T *= lbda
    T += mu * np.identity(m)
    
    Xhat = np.zeros((m,m))
    Y = np.zeros((m,m))
    
    for k in range(m) : # First system
        
        Xhat[k,:] = la.solve_banded((1,1),T,X[k,:])
    
    for k in range(m) : # Second system
    
        Y[:,k] = la.solve_banded((1,1),T,X[:,k]-np.dot(T-mu*np.identity(m),Xhat[:,k]))
        
    sol = serialize(X)
    
    return(sol)
        
    
    
