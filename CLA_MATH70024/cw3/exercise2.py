"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 3

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 2 - (a), (c), (d)

This script computes the solutions for questions (a), (c) and (d) of
the exercise 2, given in the Coursework 3 of the module MATH70024.

"""

import numpy as np
import cla_utils

# Question(a)
def buildA(n) :
    """
    Given an integer n, computes tridiagonal matrix A that is such that
    Aij = -1 if j = i - 1, Aij = 1 if j = i + 1 and Aij = 0 everywhere else.

    :param n: an integer greater than 1

    :return A: an (2n)x(2n)-dimensional numpy array
    """
    
    A = np.zeros((2*n,2*n), dtype = complex) # Initializing A to Identity
    i,j = np.indices(A.shape)
    A[i==j+1] = -1
    A[i==j-1] = 1
    
    return(A)

if __name__ == "__main__" :
    # Trying pure QR algorithm with various sizes of A
    
    A = buildA(2)
    R = cla_utils.pure_QR(A, maxit = 1000, tol = 1e-7)
    print("")
    print(R)
    print("")
    
    A = buildA(3)
    R = cla_utils.pure_QR(A, maxit = 1000, tol = 1e-7)
    print("")
    print(R)
    print("")
    
    A = buildA(10)
    R = cla_utils.pure_QR(A, maxit = 1000, tol = 1e-7)
    print("")
    print(R)
    print("")
    
    # Computing eigenvalues of A
    A = buildA(3)
    print("")
    print(np.linalg.eig(A))
    print("")


# Question(c)
def getEigenvalueA(A) :
    """
    Given the considered tridiagonal matrix A, computes its eigenvalues using
    method described in question (b).
    
    :param A: an (2n)x(2n)-dimensional numpy array
    
    :return eigvals: an (2n)-dimensional numpy array containing the eigenvalues

    """

    
    R = cla_utils.pure_QR(A, maxit = 1000, tol = 1e-7) # Pure QR
    
    a,b = np.indices(A.shape)
    eigvals = R[a==b-1]
    nonZeroIndices = (np.abs(eigvals) > 1e-7)
    eigvals = eigvals[nonZeroIndices] # Keeping non zero entries
    eigvals *= 1j

    eigvals = np.append(eigvals,eigvals.conj())
    
    return(eigvals)
    
if __name__ == "__main__" :
    # Trying custom eigenvalue computation algorithm with various sizes of A
    
    A = buildA(2)
    eigvals = getEigenvalueA(A)
    print("")
    print(eigvals)
    print("")
    
    A = buildA(3)
    eigvals = getEigenvalueA(A)
    print("")
    print(eigvals)
    print("")
    
    A = buildA(10)
    eigvals = getEigenvalueA(A)
    print("")
    print(eigvals)
    print("")
        
    
# Question(d)
def buildB(n) :
    """
    Given an integer n, computes tridiagonal matrix B that is such that
    Aij = -1 if j = i - 1, Aij = 2 if j = i + 1 and Aij = 0 everywhere else.

    :param n: an integer greater than 1

    :return B: an (2n)x(2n)-dimensional numpy array
    """
    
    B = np.zeros((2*n,2*n), dtype = complex) # Initializing A to Identity
    i,j = np.indices(B.shape)
    B[i==j+1] = -1
    B[i==j-1] = 2
    
    return(B)

if __name__ == "__main__" :
    # Trying pure QR algorithm with B
    B = buildB(3)
    R = cla_utils.pure_QR(B, maxit = 10000, tol = 1e-14)
    print("")
    print(R)
    print("")


def getEigenvalueB(B) :
    """
    Given the considered tridiagonal matrix B, computes its eigenvalues using
    method described in question (d).
    
    :param B: an (2n)x(2n)-dimensional numpy array
    
    :return eigvals: an (2n)-dimensional numpy array containing the eigenvalues

    """
    
    R = cla_utils.pure_QR(B, maxit = 10000, tol = 1e-20) # Pure QR
    
    a,b = np.indices(B.shape)
    
    upperCoeff = R[a==b-1] # Retreiving the relevant coeffs A_{k,k+1} and A_{k,k+1}
    lowerCoeff = R[a==b+1]
    
    eigvals = np.zeros(len(upperCoeff), dtype = complex) # Initializing eigvals
    
    for k in range(len(upperCoeff)) :
        eigvals[k] = np.sqrt(upperCoeff[k] * lowerCoeff[k])
        
    eigvals = np.append(eigvals,eigvals * -1.0)
    nonZeroIndices = (np.abs(eigvals) > 1e-7)
    eigvals = eigvals[nonZeroIndices] # Keeping non zero entries
    
    return(eigvals)


if __name__ == "__main__" :
    # Trying custom eigenvalue computation algorithm with various sizes of A
    
    B = buildB(2)
    eigvals = getEigenvalueB(B)
    print("")
    print(eigvals)
    print("")
    
    B = buildB(3)
    eigvals = getEigenvalueB(B)
    eig, _ = np.linalg.eig(B)
    print("")
    print(eig)
    print(eigvals)
    print("")
    
    B = buildB(10)
    eigvals = getEigenvalueB(B)
    print("")
    print(eigvals)
    print("")




