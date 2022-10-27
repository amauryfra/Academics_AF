"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 2

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 5 - (b), (d), (e)

This script computes the solutions for questions (b), (d) and (e) of
the exercise 5, given in the Coursework 2 of the module MATH70024.

"""

import numpy as np
from numpy import random
import timeit
import matplotlib.pyplot as plt


# Question (b)

def getBandedMatrix(n) :
    """
    Given an integer n, generates a random banded (n-1)**2 x (n-1)**2 matrix, 
    with upper and lower bandwidth of (n-1).

    :param n: an integer greater than 3

    :return A: an (n-1)**2x(n-1)**2-dimensional numpy array
    """
    
    m = (n-1)**2
    A = np.zeros((m,m)) # Initializing A

    for k in range(n-1) : # Adding a sub-diagonal of random numbers at each step 
        
        A += np.diag(random.uniform(low=0,high=1,size=m-k),k)
        A += np.diag(random.uniform(low=0,high=1,size=m-k),-k)
    
    return(A)



def LU_banded(A, p, q):
    """
    Compute the LU factorisation of a banded matrix A, given the upper bandwidth 
    p and the lower bandwidth q, using the in-place scheme.

    :param A: an mxm-dimensional numpy array
    :param p: an integer
    :param q: an integer

    """
    
    m = A.shape[0] # Getting m
    
    for k in range(m-1) :
        
        n = np.min([k+q,m]) # As in pseudo-code (Section 4.4)
        s = np.min([k+p,m])
        
        L = A[k+1:s,k] / A[k,k] # Temporary storage
        A[k+1:s,k:n] = A[k+1:s,k:n] - np.outer(L,A[k,k:n]) # As in pseudo-code
        A[k+1:s,k] = L 


# Showing that the operations count grows not faster than n**4

def computation_time_LU_banded() :
    """
    Plotting the computation time of LU_banded divided by n**4 over n.
    """
    
    computation_times_to_n4 = [] # Initializing a list that will contain \
        # the computation times divided by n**4
    
    for n in range(4,51) : # n going from 4 to 50
        
        A = getBandedMatrix(n) # Generating a random banded matrix
        
        def timeable_LU_banded():
            """
            Doing an example with the LU_banded function. Size is n.
            """
            test = LU_banded(A,n-1,n-1)
                
        computation_times_to_n4 += \
            [timeit.Timer(timeable_LU_banded).timeit(number=1)/n**4]
            # Adding LU decomposition computation time divided by n**4 to the list 
            
    # Plotting
    plt.plot([i+4 for i in range(47)], computation_times_to_n4)
    plt.xlabel('n')
    plt.ylabel('Computation time of LU_banded divided by n**4')
    plt.show()
            
    

# Question (e)

def getS(s0, r0, x, y) : # Only building once, it is a given parameter
    """
    Given the parameters s0 and r0, and the grid vectors x and y, computes 
    the matrix S filled with the values S_i,j, as defined in equation (8).

    :param s0: float number
    :param r0: float number
    :param x: an m-dimensional numpy array
    :param y: an m-dimensional numpy array

    :return S: an mxm-dimensional numpy array
    """
    
    m = len(x)
    
    S = np.zeros((m,m)) # Initializing S
    
    for i in range(m) :
        for j in range(m) :
            S[i,j] = s0 * np.exp( -(x[i] - 1/4)**2 / r0**2 - (y[j] - 1/4)**2 / r0**2 )
            # Given in (8)
            
    return(S)

def getB(alpha, x, y) : # Only building once, it is a given parameter
    """
    Given the parameter alpha, and the grid vectors x and y, computes 
    the matrix B filled with the values b^{k}_{i,j}, as defined in equation (8).

    :param alpha: float number
    :param x: an m-dimensional numpy array
    :param y: an m-dimensional numpy array

    :return B: an mxm-dimensional numpy array storing a 2-tuple
    """
    
    m = len(x)
    
    # Initializing B as an array of tuples 
    zero = np.empty((), dtype=object)
    zero[()] = (0, 0)
    B = np.full((m, m), zero, dtype=object) 
    
    for i in range(m) :
        for j in range(m) :
            B[i,j] = (-alpha * np.sin(np.pi * x[i]) * np.cos(np.pi * y[j]), \
                      alpha * np.cos(np.pi * x[i]) * np.sin(np.pi * y[j]))     
            # Given in (8)
    return(B)
    


def RHS6(S, B, U, mu=1) :
    """
    Given the parameter mu, and the matrix S, B and U, computes 
    the right hand side of equations (6).

    :param S: an mxm-dimensional numpy array
    :param B: an mxm-dimensional numpy array
    :param U: an mxm-dimensional numpy array
    :param mu: float number

    :return VectRHS6: an m-dimensional numpy array
    """
    
    S_vect = S.T.flatten() # Flatten version of S using the check{S} ordering
    v = U.flatten() # Flatten version of U using the initial ordering
    
    m = S_vect.shape[0]
    
    VectRHS6 = np.zeros(m) # Initializing VectRHS6
    
    n = (m**0.5 + 1)
    
    deltax = 1 / n
    
    for k in range(1,m-1) : # O(n**2)
        
         i = int(k//n)-1 # k = (k//n) n + k%n
         j = int(k%n)-1 # Retreiving i and j from flattened index
        
         VectRHS6[k] = S_vect[k] - (1/(2*deltax)) * float(B[i,j][1]) * (v[k+1] - v[k-1]) \
            + (mu/(deltax**2)) * (v[k-1] + v[k+1]) # Given in (8)
    
    return(VectRHS6)


def RHS7(S, B, U, mu=1) :
    """
    Given the parameter mu, and the matrix S, B and U, computes 
    the right hand side of equations (7).

    :param S: an mxm-dimensional numpy array
    :param B: an mxm-dimensional numpy array
    :param U: an mxm-dimensional numpy array
    :param mu: float number

    :return VectRHS7: an m-dimensional numpy array
    """
    
    S_vect = S.flatten() # Flatten version of S using the initial ordering
    v = U.T.flatten() # Flatten version of U using the check{v} ordering
    
    m = S_vect.shape[0]
    
    VectRHS7 = np.zeros(m) # Initializing VectRHS6
    
    n = (m**0.5 + 1)
    
    deltax = 1 / n
    
    for k in range(1,m-1) :
        
         j = int(k//n)-1 # k = (k//n) n + k%n
         i = int(k%n)-1 # Retreiving i and j from flattened index
        
         VectRHS7[k] = S_vect[k] - (1/(2*deltax)) * float(B[i,j][0]) * (v[k+1] - v[k-1]) \
            + (mu/(deltax**2)) * (v[k-1] + v[k+1]) # Given in (8)
    
    return(VectRHS7)


def buildC1(B, c, mu=1) :
    """
    Given the parameter mu and c, and the matrix B, computes 
    the matrix multiplying check{v} in the left hand side of equations (6).

    :param B: an mxm-dimensional numpy array
    :param c: float number
    :param mu: float number


    :return C1: an mxm-dimensional numpy array
    """
    
    n = B.shape[0]
    m = n**2
    C1 = np.zeros((m,m)) # Initializing C1
    
    deltax = 1 / n # Delta x
    
    C1 += np.diag(((4*mu)/deltax**2 + c) * np.ones(m),0) # Diagonal coefficients
    
    B_list = [] # Retreiving the b_ij in this list
    
    for k in range(m) : # O(n**2)
        
        j = int(k//n) # k = (k//n) n + k%n
        i = int(k%n) # Retreiving i and j from flattened index

        B_list += [B[i,j][0]] # b^1
    
    B_list = np.array(B_list)
    C1 += np.diag((1/(2*deltax)) * B_list[:m-1] - mu/deltax**2 * np.ones(m-1),1)
    C1 += np.diag((-1/(2*deltax)) * B_list[1:] - mu/deltax**2 * np.ones(m-1),-1)
    
    return(C1)


def buildD1(B, c, mu=1) :
    """
    Given the parameter mu and c, and the matrix B, computes 
    the matrix multiplying v in the left hand side of equations (7).

    :param B: an mxm-dimensional numpy array
    :param c: float number
    :param mu: float number


    :return D1: an mxm-dimensional numpy array
    """
    
    n = B.shape[0]
    m = n**2
    D1 = np.zeros((m,m)) # Initializing D1
    
    deltax = 1 / n # Delta x
    
    D1 += np.diag((4*mu/deltax**2 + c) * np.ones(m),0) # Diagonal coefficients
    
    B_list = [] # Retreiving the b_ij in this list
    
    for k in range(m) : # O(n**2)
        
        i = int(k//n) # k = (k//n) n + k%n
        j = int(k%n) # Retreiving i and j from flattened index
        
        B_list += [B[i,j][1]] # b^2
    
    B_list = np.array(B_list)
    D1 += np.diag((1/(2*deltax)) * B_list[:m-1] - mu/deltax**2 * np.ones(m-1),1)
    D1 += np.diag((-1/(2*deltax)) * B_list[1:] - mu/deltax**2 * np.ones(m-1),-1)
    
    return(D1)
                

                  
def solve_L_banded(L, b, q):
    """
    Solves the system Lx=b for x with L banded lower triangular, with 
    bandwidth q, by forward substitution.

    :param L: an mxm-dimensional numpy array, assumed banded lower triangular 
    :param b: an m-dimensional numpy array
    :param q: an integer
    
    :return x: an m-dimensional numpy array

    """

    m = b.shape[0] # Retreiving m 
    
    x = np.zeros(m) # Initializing x
    
    x[0] = b[0] / L[0,0] # First value of substitution
    
    for i in range(1,m) : # Iterative substitution (forward)
        
        x[i] = ( b[i] - np.dot(L[i,max(0,i-q):i],x[max(0,i-q):i]) ) / L[i,i]
        # Only relevant computations not involving zeros are made
        
    return(x)


def solve_U_banded(U, b, p):
    """
    Solves the system Ux=b for x with U banded upper triangular, with 
    bandwidth p, by backward substitution.

    :param U: an mxm-dimensional numpy array, assumed banded upper triangular 
    :param b: an m-dimensional numpy array
    :param p: an integer
    
    :return x: an m-dimensional numpy array
    
    """
                     
    m = b.shape[0] # Retreiving m 
    
    x = np.zeros(m) # Initializing x
    
    x[m-1] = b[m-1] / U[m-1,m-1] # First value of back substitution
    
    for i in reversed(range(m-1)) : # Iterative substitution (backwards)
        
        x[i] = ( b[i] - np.dot(U[i,i+1:min(m,i+p+1)],x[i+1:min(m,i+p+1)]) ) / U[i,i]
        
    return(x)


def testSolution(B, U, VectRHS7 ,c, mu=1) :
    """
    Quantifies how good the solution given by U verifies equation (4).
    
    :param B : an mxm-dimensional numpy array
    :param U : an mxm-dimensional numpy array
    :param VectRHS7: an m-dimensional numpy array
    :param c: float number
    :param mu: float number
    
    :return espilon: a float number
    
    """
    
    m = VectRHS7.shape[0]
    
    n = (m**0.5 + 1)
    
    deltax = 1 / n
    
    v = U.flatten() # Flatten version of U using the check{v} ordering
    
    VectLHS7 = np.zeros(m) # Left hand side
    
    for k in range(1,m-1) : # O(n**2)
        
         i = int(k//n)-1 # k = (k//n) n + k%n
         j = int(k%n)-1 # Retreiving i and j from flattened index
         
         VectLHS7[k] = (float(B[i,j][1])/(2 * deltax)) * (v[k+1]-v[k-1]) - \
             (mu/deltax**2) * (v[k+1] + v[k-1] - 4*v[k]) + c*v[k]
         
    
    epsilon = np.linalg.norm(VectLHS7-VectRHS7) # O(n**2)
    
    return(epsilon)


def LU_extraction(A) :
    """
    Extracts L and U of an in-place LU decomposition.
        
    :param A : an mxm-dimensional numpy array

        
    :return L: an mxm-dimensional numpy array
    :return U: an mxm-dimensional numpy array
        
    """
    
    m = A.shape[0]
    
    # Extracting L and U (as seen in test code)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
        
    return(L, U)


# We now have all the building blocks


def exercise5_main(n, tolerance, s0, r0, c, alpha, mu=1) :
    """
    Solves the stationary advection-reaction-diffusion equation, by an iterative
    method, for S and b given in equation (8). The parameter n is such that the
    grid spacing is Delta_x = 1/n. The parameter tolerance gives the termination 
    condition of the iteration loop. The other parameters of the equation are 
    also inputed.

    :param n: an integer
    :param tolerance: a float number
    :param s0: a float number
    :param r0: a float number
    :param c: a float number
    :param alpha: a float number
    :param mu: a float number
    
    :return U: an nxn-dimensional numpy array
    
    """
    
    # Creating the grid
    x = np.arange(1/n, 1, 1/n)
    y = np.arange(1/n, 1, 1/n)
    
    # Getting S and B matrix
    S = getS(s0, r0, x, y) # O(n**2)
    B = getB(alpha, x, y) # O(n**2)
    
    # Initial guess for the iterative method
    U = np.zeros((n-1,n-1))
    
    # Initial epsilon, quantifying how the solution satisfies the equation (4)
    VectRHS7 = RHS7(S, B, U, mu) 
    epsilon = testSolution(B, U, VectRHS7 ,c, mu)
    
    while epsilon > tolerance :
        
        # From k to k + 1/2
        VectRHS6 = RHS6(S, B, U, mu) 
        C1 = buildC1(B, c, mu)
        LU_banded(C1, 1, 1) # Bandwidth is 1, in-place scheme
        L, Up = LU_extraction(C1)
        Y = solve_L_banded(L, VectRHS6, 1)
        X = solve_U_banded(Up, Y, 1)
        
        # X back to U
        U = X.reshape(n-1,n-1).transpose()
        
        # From k + 1/2 to k + 1
        VectRHS7 = RHS7(S, B, U, mu) 
        D1 = buildD1(B, c, mu)
        LU_banded(D1, 1, 1)
        L, Up = LU_extraction(D1)
        Y = solve_L_banded(L, VectRHS7, 1)
        X = solve_U_banded(Up, Y, 1)
        
        # X back to U
        U = X.reshape(n-1,n-1)
        
        # Test the accuracy of the solution
        epsilon = testSolution(B, U, VectRHS7 ,c , mu)
    
    return(U)



