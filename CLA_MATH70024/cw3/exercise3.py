"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 3

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 3 - (b), (c), (d), (e)

This script computes the solutions for questions (b), (c), (d) and (e) of
the exercise 3, given in the Coursework 3 of the module MATH70024.

"""

import numpy as np
from numpy import random
import cla_utils
import copy
import scipy
import matplotlib.pyplot as plt
import timeit 

# Question (b)
def buildNewA() :
    """
    Computes symmetric matrix A that is such that Aij = 1 / (i + j + 1)

    :return A: an 5x5-dimensional numpy array
    """
    # Building 5x5 matrix A
    A = np.zeros((5,5), dtype = complex)
    for i in range(5) :
        for j in range(5) :
            A[i,j] = 1 / (i + j + 1)
    
    return(A)

if __name__ == "__main__" :
    # Investigations
    A = buildNewA()
    T = copy.deepcopy(A)
    print("")
    print(A)
    print("")
    
    cla_utils.hessenberg(T) # Tridiagonal matrix reduction
    print("")
    print(T)
    print("")
    
    R, _ = cla_utils.pure_QR(T, maxit = 1000, tol = 0, cw3Condition = True)
    print("")
    print(R)
    print("")
    
    # R is roughly upper triangular 
    i,j = np.indices(R.shape)
    eigvalR = R[i==j]
    print("")
    print(eigvalR)
    print("")
    
    for val in eigvalR :
        print(np.abs(np.linalg.det(R - val * np.identity(R.shape[0]))))


# Question (c)

def customTridiagQR(A0, maxit) :
    """
    This function is the implementation of procedure described in question (3c). 


    :param A0: an mxm numpy array
    :param maxit: the maximum number of iterations

    :return eigval: an m-dimensional numpy array containing the eigenvalues of A
    :return res: a mxp-dimensionnal numpy array containing the residuals of each QR
    iterations
    """
    
    m = A0.shape[0] # Retreiving m
    
    A = 1.0 * A0 # Avoid in-place working
    
    eigval = np.zeros(m, dtype = complex) # Initialization
    res = np.array([[]])
    
    #cla_utils.hessenberg(A) # Tridiagonal matrix reduction
    A = scipy.linalg.hessenberg(A) # Tridiagonal matrix reduction
    
    for k in reversed(range(0,m)) :
        A, r = cla_utils.pure_QR(A, maxit = maxit, tol = 0, cw3Condition = True)
        eigval[k] = A[k,k]
        if k !=0 :
            res = np.append(res,r) # Adding array of successive stopping condition candidates 
        A = A[:k,:k]

    return(eigval, res)
        
if __name__ == "__main__" :
    # Investigations
    A0 = buildNewA() 
    eigval, res = customTridiagQR(A0, maxit = 1000)
    print("")
    print(eigval)
    eigvalTrue, _ = np.linalg.eig(A0)
    print(eigvalTrue)
    print("")
    print(res)
    print("")
    #plt.plot(res)
    #plt.yscale('log')
    #plt.show
    
    n = 15
    C = random.uniform(size=(n,n))
    C = (C + C.T)/2
    eigval, res = customTridiagQR(C, maxit = 1000)
    print("")
    print(eigval)
    eigvalTrue, _ = np.linalg.eig(C)
    print(eigvalTrue)
    #plt.plot(res)
    #plt.yscale('log')
    #plt.show

def timeable_unmodifiedQR():
    """
    Doing an example with the unmodified QR.
    """
    A = buildNewA()
    T = copy.deepcopy(A)
    T = scipy.linalg.hessenberg(T) # Tridiagonal matrix reduction
    _, _ = cla_utils.pure_QR(T, maxit = 1000, tol = 0, cw3Condition = True)
    
def timeable_modifiedQR():
    """
    Doing an example with the modified QR.
    """
    A = buildNewA()
    T = copy.deepcopy(A)
    _, _ = customTridiagQR(T, maxit = 1000)


def time_eigQR():
    """
    Get some timings for the QR eigenvalue computation.
    """

    print("Timing for unmodfied QR | Size 5x5")
    print(timeit.Timer(timeable_unmodifiedQR).timeit(number=1))
    
    print("")
    
    print("Timing for modified QR | Size 5x5")
    print(timeit.Timer(timeable_modifiedQR).timeit(number=1))
    
if __name__ == "__main__" :
    print("")
    time_eigQR()


# Question (d)

def customTridiagQR_modified(A0, maxit) :
    """
    This function is the implementation of procedure described in question (3c),
    with the use of the Wilkinson shift. 


    :param A0: an mxm numpy array
    :param maxit: the maximum number of iterations

    :return eigval: an m-dimensional numpy array containing the eigenvalues of A
    :return res: a mxp-dimensionnal numpy array containing the residuals of each QR
    iterations
    """
    
    m = A0.shape[0] # Retreiving m
    
    A = 1.0 * A0 # Avoid in-place working
    
    eigval = np.zeros(m, dtype = complex) # Initialization
    res = np.array([[]])
    
    #cla_utils.hessenberg(A) # Tridiagonal matrix reduction
    A = scipy.linalg.hessenberg(A) # Tridiagonal matrix reduction
    
    for k in reversed(range(0,m)) :
        A, r = cla_utils.pure_QR(A, maxit = maxit, tol = 0, cw3Condition = True, \
                                 shiftCondition = True)
        eigval[k] = A[k,k]
        if k !=0 :
            res = np.append(res,r) # Adding array of successive stopping condition candidates 
        A = A[:k,:k]

    return(eigval, res)

if __name__ == "__main__" :
    # Investigations
    A0 = buildNewA() 
    eigval_A_Wilk, res_A_Wilk = customTridiagQR_modified(A0, maxit = 1000)
    print("")
    print(eigval_A_Wilk)
    eigvalTrue, _ = np.linalg.eig(A0)
    print(eigvalTrue)
    print("")
    print(res_A_Wilk)
    print("")
    
    #plt.plot(res_A_Wilk)
    #plt.yscale('log')
    #plt.show
    
    n = 15
    C = random.uniform(size=(n,n))
    C = (C + C.T)/2
    
    eigval_Rand_Wilk, res_Rand_Wilk = customTridiagQR_modified(C, maxit = 1000)
    print("")
    print(eigval_Rand_Wilk)
    eigvalTrue, _ = np.linalg.eig(C)
    print(eigvalTrue)
    
    
    _, res_Rand = customTridiagQR(C, maxit = 1000)
    #plt.plot(res_Rand_Wilk)
    #plt.plot(res_Rand)
    #plt.yscale('log')
    #plt.show
    
    
# Question (e)

if __name__ == "__main__" :
    
    # Investigations
    D = np.diag([15-i for i in range(15)])
    O = np.ones((15,15), dtype = complex)
    A1 = D + O
    
    A0 = buildNewA() 
    
    _, res_A0 = customTridiagQR(A0, maxit = 1000)
    _, res_A1 = customTridiagQR(A1, maxit = 1000)
    #plt.plot(res_A0, label = '5x5 defined A')
    #plt.plot(res_A1, label = 'A = D + O')
    #plt.yscale('log')
    #plt.legend(loc="lower left")
    #plt.show
    
    _, res_A0_Wilk = customTridiagQR_modified(A0, maxit = 1000)
    _, res_A1_Wilk = customTridiagQR_modified(A1, maxit = 1000)
    #plt.plot(res_A0_Wilk, label = '5x5 defined A')
    #plt.plot(res_A1_Wilk, label = 'A = D + O')
    #plt.yscale('log')
    #plt.legend(loc="lower left")
    #plt.show
    
    
    
    
    
    








