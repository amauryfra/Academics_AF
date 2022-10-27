import numpy as np
import numpy.random as random


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m = A.shape[0] # Retreiving m
    
    Q = np.zeros((m,k+1), dtype=complex) # Initializing Q
    H = np.zeros((k+1,k), dtype=complex) # Initializing H

    Q[:,0] = b / np.linalg.norm(b) # q1

    for n in range(k) :
        
        v = np.dot(A, Q[:,n]) # As in pseudo-code
        
        H[:n+1,n] = np.dot(Q[:,:n+1].conj().T,v)
        v = v - np.dot(Q[:,:n+1],H[:n+1,n])
        
        H[n+1,n] = np.linalg.norm(v)
        
        Q[:,n+1] = v / H[n+1,n] # H[n+1,n]  = ||v|| here

    return Q, H

import cla_utils

def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False, func = None, precondFunc = None):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical
    :param func: optionally provide a function instead of matrix A 
    :param func: optionally provide a function that acts as a solver for
     Mz = y, where M is a preconditioning matrix
    
    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    if x0 is None:
        x0 = b
    
    m = b.shape[0]
    
    if precondFunc != None :
        b = precondFunc(b)
    
    Q = np.array(b.reshape(m,1) / np.linalg.norm(b)) # Initializing Q
    H = np.zeros((maxit+1,maxit+1)) # Initializing H
    
    nits = -1 # Initializing number of iterations
    rnorms = []
    r = []
    
    bNorm = np.linalg.norm(b) # Computed only once

    for n in range(maxit) :

        
        nits += 1 # Doing one more iteration
        
        # Arnoldi
        if func == None and  precondFunc == None : # func is CW3 condition
            v = np.dot(A, Q[:,n]) # As in pseudo-code
        elif func == None and precondFunc != None :
            v = precondFunc(np.dot(A, Q[:,n])) # Preconditioning CW3 condition
        elif func != None and precondFunc != None :
            v = precondFunc(func(Q[:,n]))
        else :
            v = func(Q[:,n])
            
        H[:n+1,n] = np.dot(Q[:,:n+1].conj().T,v)
        v = v - np.dot(Q[:,:n+1],H[:n+1,n])
        H[n+1,n] = np.linalg.norm(v)
        
        
        # Least squares
        e1 = np.eye(1,n+1,0)[0] # e1 base vector
        y = cla_utils.householder_ls(H[:n+1,:n+1], bNorm * e1) 
        
        # Computing xn
        x = np.dot(Q,y)
        
        Q = np.column_stack((Q,v/H[n+1,n]))
        
        # Checking tolerance
        if func == None :
            r += [np.dot(A,x)-b]
        else :
            r += [func(x)-b]
        rnorms += [np.linalg.norm(r[n])]
        if rnorms[n] < tol :
            break
        
    if nits == maxit - 1 :
        nits = -1
            
    if return_residuals == True and return_residual_norms == True :
        return(x, nits, r, rnorms)
    
    elif return_residuals == True and return_residual_norms == False :
        return(x, nits, r)
    
    elif return_residuals == False and return_residual_norms == True :
        return(x, nits, rnorms)
    
    else :
        return(x, nits)

    return()


def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
