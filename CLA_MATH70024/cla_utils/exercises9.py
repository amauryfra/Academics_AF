import numpy as np
import numpy.random as random

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """

        #elif (store_iterations == True):
            
    v = [x0] # Initializing list of iterates with initial guess
    
    for k in range(maxit) :
        
        w = np.dot(A,v[k])
        w = w / np.linalg.norm(w) # As in pseudo-code
        
        lambda0 = np.dot(w.conj().T,np.dot(A,w))
        
        r = np.dot(A,w) - lambda0 * w
        v += [w] # Storing iterates
        
        if np.linalg.norm(r) < tol : 
            break # Stop if tolerance is reached 
        
    if store_iterations :
        x = np.array(v)
    else :
        x = v[len(v)-1]
        
    return(x, lambda0)


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    
    m = A.shape[0] # Retreiving m
    
    v = [x0] # Initializing list of iterates with initial guess
    lbdas = [] # Initializing list of eigenvalues iterates  
    
    M = A - mu * np.identity(m)
    
    for k in range(maxit) :
        
        w = np.linalg.solve(M, v[k])
        w = w / np.linalg.norm(w) # As in pseudo-code
        v += [w] # Storing iterates
        l0 = np.dot(w.conj().T,np.dot(A,w))
        lbdas += [l0] # Storing eigenvalues
        
        r = np.dot(A,w) - l0 * w
        
        if np.linalg.norm(r) < tol : 
            break # Stop if tolerance is reached 
    
    if store_iterations :
        x = np.array(v)
        l = np.array(lbdas)
    else :
        x = v[len(v)-1]
        l = lbdas[len(lbdas)-1]
        
    return(x, l)


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    m = A.shape[0] # Retreiving m
    
    v = [x0] # Initializing list of iterates with initial guess
    lbdas = [np.dot(v[0].conj().T,np.dot(A,v[0]))] # Initializing list of eigenvalues iterates  
    
    for k in range(maxit) :
        
        w = np.linalg.solve(A - lbdas[k] * np.identity(m),v[k]) # As in pseudo-code
        w = w / np.linalg.norm(w) 
        v += [w] # Storing iterates
        
        l0 = np.dot(w.conj().T,np.dot(A,w))
        lbdas += [l0] # Storing eigenvalues
        
        r = np.dot(A,w) - l0 * w
        
        if np.linalg.norm(r) < tol : 
            break # Stop if tolerance is reached 

    if store_iterations :
        x = np.array(v)
        l = np.array(lbdas)
    else :
        x = v[len(v)-1]
        l = lbdas[len(lbdas)-1]
        
    return(x, l)

import copy 
import cla_utils

def pure_QR(A, maxit, tol, cw3Condition = False, shiftCondition = False) :
    """
    For matrix A, apply the QR algorithm and return the result. 
    
    It assures convergence by verifying how close to an upper triangular the
    iterated matrix is. It checks how small are the lower triangular values of
    said matrix.


    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations   
    :param tol: termination tolerance

    :return Ak: the result
    """
    m = A.shape[0] # Retreiving m
    
    Ak = 1.0 * A # Pre-implemented function works in-place
    
    res = np.array([np.abs(Ak[m-1,m-2])]) # Stopping condition candidates 
    
    I = np.identity(m, dtype = complex)
    
    
    for k in range(maxit) :
        
        if shiftCondition : # Wilkinson shift CW3
            
            a = Ak[m-1,m-1]
            b = Ak[m-1,m-2]
            delta = (Ak[m-2,m-2] - a) / 2
            sgn = np.sign(delta) if np.sign(delta) != 0 else 1
            mu = a - (sgn * b**2) / (np.abs(delta) + np.sqrt(delta**2 + b**2))
            
            Ak = Ak - mu * I
        
        
        # As householder_qr has been written for real matrix
        Q,R = np.linalg.qr(Ak, mode = 'complete')
        
        if shiftCondition :
            Ak = np.dot(R,Q) + mu * I

        else : 
            Ak = np.dot(R,Q)

                
        coeff = Ak[m-1,m-2]
        res = np.append(res,np.abs(coeff))
        
        if not cw3Condition : # Modified QR CW3 condition
            if np.linalg.norm(Ak[np.tril_indices(m, -1)])/(m**2) < tol :
                break
        else : 
            if np.abs(Ak[m-1,m-2])< 1e-12 :
                break
            
    if not cw3Condition :
        return(Ak)
    else :
        return(Ak, res)



