"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 3 - (c), (d), (e)

This script computes the solutions for questions (c), (d) and (e) of
the exercise 3, given in the Coursework 1 of the module MATH70024.

"""

import numpy as np
import cla_utils
import scipy


# Question (c)

def Rv_householder(A, kmax=None):
    """
    Given a real mxn matrix A, finds the reduction to upper triangular matrix R
    using the Householder transformations. It stores the ongoing vectors v - useful
    to the computation - in the remaining space of R (initially filled with zeros).

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return Rv: an mxn-dimensional numpy array containing an upper \
    triangular matrix, along with the stored values of v in the lower triangular 
    remaining space.
    """
    
    # Code globally similar to the one done in the weekly exercises
    
    m, n = A.shape
    if kmax is None:
        kmax = n
        
    for k in range(kmax) : # Transformations of the columns of A to Rv
    
        x = A[k:m,k]
        
        e1 = np.eye(1,m-k,0,dtype = float)[0] # e1 of same size as sub-column A[k:m,k]
    
        
        if np.sign(x[0]) == 0 : # Taking care of this specific case separately 

            
            v = np.linalg.norm(x) * e1 + x # np.sign(x[0]) set to 1
            v = v / np.linalg.norm(v) # Normalized by 2-Norm
            
            A[k:m,k:n] = A[k:m,k:n] - 2 * np.outer(v,((v.transpose().dot(A[k:m,k:n])))) # As in pseudo-code
            
            v = (A[k,k]/v[0]) * v ##### Scaling to fit the value that overlaps the diagonal #####
            A[k:,k] = v ##### Stacking v in the lower part of the column #####
            
        else :
            
            v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
            v = v / np.linalg.norm(v)
            
            A[k:m,k:n] = A[k:m,k:n] - 2 * np.outer(v,((v.transpose().dot(A[k:m,k:n]))))
            
            v = (A[k,k]/v[0]) * v ##### Scaling to fit the value that overlaps the diagonal #####
            A[k:,k] = v ##### Stacking v in the lower part of the column #####
        
    return A ##### A has been transformed in Rv #####


# Question (d)

def Rv_multiplication(Rv, b):
    """
    Given an mxn matrix Rv, in which are stored the Householder transformation 
    vectors v, the function computes Q^*b, where Q is the QR decomposition matrix 
    related to R, the upper triangular matrix stored in Rv.

    :param Rv: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return a: an m-dimensional numpy array being the computation of Q^*b
    """

    m, n = Rv.shape # Getting the shape of Rv
    
    for k in range(n) : # Transformation of b through Householder process
    
        v = Rv[k:,k] # Retreiving v
        v = v / (np.linalg.norm(v)) # De-scaling it 
        
        b[k:m] = b[k:m] - (2 * np.dot(v,np.dot(v.transpose(),b[k:m])))  # Householder process
        # Implicit multiplication

    return b # b has been transformed in Q^*b


# Question (e)

def Rv_ls(Rv, b):
    """
    Given an mxn matrix Rv, in which are stored the Householder transformation 
    vectors v, the function computes a vector x minimizing the least squares problem
    ||Ax-b||, where A is such that A=QR, R being the upper triangular matrix 
    stored in Rv.

    :param Rv: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array minimizing said ||Ax-b||
    """

    m, n = Rv.shape # Getting the shape of Rv
    
    R = np.triu(Rv) # Getting rid of the v column vectors 
    
    b = Rv_multiplication(Rv, b) # Qhat^* b = b[:n] for such a modfied b

    x = scipy.linalg.solve_triangular(R[:n,:n],b[:n]) # Rhat = R[:n,:n]
    
    return x # Minimizing ||Ax-b||

