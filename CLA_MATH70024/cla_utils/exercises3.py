import numpy as np


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape
    if kmax is None:
        kmax = n 
    
        
    for k in range(kmax) : # Transformations of the columns of A to R
    
        x = A[k:,k]
        
        e1 = np.eye(1,m-k,0,dtype = complex)[0] # e1 of same size as sub-column A[k:m,k]
        
        try :
        
            if np.sign(x[0]) == 0 : # Taking care of this specific case separately 
    
                
                v = np.linalg.norm(x) * e1 + x # np.sign(x[0]) set to 1
                v = v / np.linalg.norm(v) # Normalized by 2-Norm
                
                A[k:m,k:n] = A[k:m,k:n] - 2 * np.outer(v,((v.conj().T.dot(A[k:m,k:n])))) # As in pseudo-code
                
            else :
                
                v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
                v = v / np.linalg.norm(v)
                
                A[k:m,k:n] = A[k:m,k:n] - 2 * np.outer(v,np.dot(v.conj().T, A[k:m,k:n]))
                
        except :
            
            pass
            

    return A # A has been transformed in R (such that A = QR)

import scipy.linalg 

def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    m, _ = A.shape # Getting m
    #_, k = b.shape # Getting k
    
    Ahat = np.column_stack((A,b)) # Extended array with vectors b_i stacked at the end
    
    R = householder(A) # Getting R such that A = QR 
    
    R_ = householder(Ahat, kmax=m) # Getting bhat (R_[:,m:]) such that Rx = bhat \
        # (implicit multiplication)

    x = scipy.linalg.solve_triangular(R,R_[:,m:]) # Getting x using scipy triangular solver

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    m, n = A.shape # Getting m, n

    I = np.identity(m, dtype = complex) # mxm identity matrix
    
    Ahat = np.column_stack((A,I)) # Extended array with I stacked at the end
    
    R_ = householder(Ahat, kmax=n) # Getting Q* (last m columns of R_) by implicit multiplication
    
    R = R_[:,:n] # Getting R such that A = QR 
    
    Q = R_[:,n:].conj().T
    
    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    m, n = A.shape # Getting m, n
    
    Ahat = np.column_stack((A,b)) # Extended array with vector b stacked at the end
    
    R = householder(Ahat) # Performing Householder triangulation
    
    x = scipy.linalg.solve_triangular(R[:n,:n],R[:n,n]) # Rhat = R[:n,:n], Qhat^* b = R[:n,n]
    
    return x

