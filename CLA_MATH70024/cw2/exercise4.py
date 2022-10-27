"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 2

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 4 - (a), (c), (e)

This script computes the solutions for questions (a), (c) and (e) of
the exercise 4, given in the Coursework 2 of the module MATH70024.

"""

import numpy as np
from numpy import random
from cla_utils import LU_inplace

# Question (a)

def overlappingBlocksPrototype(n, epsilon) :
    """
    Given an integer n and a real number epsilon, computes a random
    overlapping blocks matrix prototype of the form 
    A = I + epsilon * ∑ (k=1 to n) Bk, where Bk = v_ijk if 4(k-1) < i,j <= 4k+1,
    Bk = 0 otherwise.

    :param n: an integer greater than 2
    :param epsilon: float number striclty positive

    :return A: an (4n+1)x(4n+1)-dimensional numpy array
    """
    
    A = np.identity(4*n +1) # Initializing A to Identity
    random.seed(1878*n) 
    
    for k in range(1,n+1) :
        
        Bk = np.zeros((4*n+1,4*n+1)) # Initializing Bk to the zero matrix
        
        # Building Bk by inserting uniformly taken random numbers between 0 and 1 \
            # in the relevant places
        Bk[4*(k-1) : 4*k+1 , 4*(k-1) : 4*k+1] = np.random.uniform(low=0,high=1,size=(5 , 5))
        # We noticed that all blocks are of size (5,5)

        A += epsilon * Bk
    
    return(A)


print("")
# Generating a prototype
A = overlappingBlocksPrototype(n=2,epsilon=0.1)
print("Prototype of overlapping blocks matrix A : ")
print(A)
 
# Performing LU decomposition 
LU_inplace(A)
# Extracting L and U (as seen in test code)
n = A.shape[0]
L = np.eye(n)
i1 = np.tril_indices(n, k=-1)
L[i1] = A[i1]
U = np.triu(A)
print("")
print("")
print("LU decomposition of A :")
print("")
print("L : ")
print(L)
print("")
print("U : ")
print(U)
print("")


# Question (c)

def banded_LU_modified(A) :
    """
    Given an (4n+1)x(4n+1) overlapping blocks prototype matrix A, performs 
    LU decomposition using the in-place scheme and the modified banded matrix algorithm.

    :param A: an (4n+1)x(4n+1)-dimensional numpy array

    """
    
    m = A.shape[0] # Getting m = 4n + 1
    
    for k in range(m-1) :
        
        block_nb = int(k/4) + 1 # The block sub-matrix we are currently processing
        # 1 <= block_nb <= n
        max_index = 4 * block_nb + 1
        
        L = A[k+1:max_index,k] / A[k,k] # Temporary storage
        
        A[k+1:max_index,k:max_index] = A[k+1:max_index,k:max_index] \
            - np.outer(L,A[k,k:max_index]) # LU decomposition algorithm
        A[k+1:max_index,k] = L

    return(A)


# Question (e)

def lower_triang(A) :
    """
    Given a matrix A, performs a lower triangularization using an elimination process,
    made in-place.

    :param A: an mxm-dimensional numpy array
    """
    
    m = A.shape[0]
    
    for k in range(1,m) :
        
        A[:k,k] = A[:k,k] / A[k,k]
        A[:k,k:] = A[:k,k:] - np.outer(A[:k,k],A[k,k:])
    


def tridiag_banded(A, b) :
    """
    Given an (4n+1)x(4n+1) overlapping blocks prototype matrix A, and (4n+1)-length  
    vector b, we perform a triangularization of each block, to transform the
    system Ax=b into the smaller system A'x'=b', where A' is a tridiagonal matrix.
    We then solve the smaller system, before reconstructing the eliminated entries
    using back substitution.

    :param A: an (4n+1)x(4n+1)-dimensional numpy array
    :param b: an (4n+1)-dimensional numpy array
    """

    m = A.shape[0] # Getting m = 4n + 1
    
    newA = np.zeros((m,m))
    newb = np.zeros(m)
    
    n = int((m-1) / 4)
    
    triDiagA = np.zeros((3*(n-1),3*(n-1)))
    triDiagAdapted_b = np.zeros(3*(n-1))
    
    for k in range(n-1) :
        
        block_nb = k + 1 # The block sub-matrix we are currently processing
        max_index = 4 * block_nb + 1
        prev_index = 4 * (block_nb-1)
        
        block = A[prev_index:max_index,prev_index:max_index]
        
        block = np.column_stack((block,b[prev_index:max_index])) 
        
        if block_nb%2 != 0 :
            
            LU_inplace(block)
            block = np.triu(block)
            newb[prev_index:max_index] = block[:,5:].reshape(5,)  
            newA[prev_index:max_index,prev_index:max_index] = block[:,:5]
        
        else :
            
            lower_triang(block)
            newb[prev_index:max_index] = block[:,5:].reshape(5,) 
            L = np.eye(5)
            i1 = np.tril_indices(5, k=-1)
            L[i1] = block[i1]
            newA[prev_index:max_index,prev_index:max_index] = L
            
        
        triDiagA[3*(block_nb-1) : 3*(block_nb), 3*(block_nb-1) : 3*(block_nb)] = \
            newA[max_index-2:max_index+1,max_index-2:max_index+1]
            
        triDiagAdapted_b[block_nb-1:block_nb+2] = newb[max_index-2:max_index+1]
        
    return(newA,newb,triDiagA,triDiagAdapted_b)
        
        
        
    
    



