"""
Imperial College London
Computational Linear Algebra (MATH70024)
Mastery component

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

This script runs the implementations used in the Mastery component 
of the module MATH70024.

"""


import scipy.io
import numpy as np
from numpy import random
import cla_utils
import timeit
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

if __name__ == "__main__" :
    data = scipy.io.loadmat('/.../airfoil_matrix.mat')
    A0 = data['A'].toarray() # Retreiving relevant coefficient matrix A
    plt.spy(A0, markersize = 1)
    plt.show()
    

def verifyPositiveDefinite(A) :
    """
    Verifies if the inputed matrix A is positive definite. Returns True if the 
    matrix is indeed positive definite, False otherwise. 
    """
    return(np.all(np.linalg.eigvals(A) > 0))

if __name__ == "__main__" :
    print("")
    print("Is considered matrix A0 positive definite ? Answer : " + str(verifyPositiveDefinite(A0)))
    print("")

def HSS(A, b, alpha, x0, tol, maxit) :
    """
    For a matrix A, solve Ax=b using the HSS method.

    :param A: an nxn numpy array
    :param b: n-dimensional numpy array
    :param alpha: floating point number strictly positive, method's coefficient
    :param x0: the initial guess (if not present, use b)
    :param tol: floating point number, the tolerance for termination
    :param maxit: integer, the maximum number of iterations

    :return x: an n-dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    """
    
    if x0 is None :
        x0 = b
    
    n = A.shape[0] # Size of coefficient matrix
    I = np.identity(n)
    Ia = alpha * I
    
    Astar = A.conj().T
    H = (1/2) * (A + Astar) # HS splitting
    S = (1/2) * (A - Astar)
    
    M1 = Ia + H
    M2 = Ia + S
    
    N1 = Ia - S
    N2 = Ia - H
    
    M1inv = np.linalg.inv(M1) # Computing n x n inverses once
    M2inv = np.linalg.inv(M2) # Not doable in practice, used from theoretical perspective
    
    b1 = np.dot(M1inv,b) # Computing those vectors only once
    b2 = np.dot(M2inv,b)
    
    C1 = np.dot(M1inv, N1) # Computing those matrix only once
    C2 = np.dot(M2inv, N2)
    
    nits = -1 # Initializing number of iterations
    
    x = x0
    
    for k in range(maxit) :
        
        nits += 1
        
        x = np.dot(C1, x) + b1 # Step k -> k + 1/2
        x = np.dot(C2, x) + b2 # Step k + 1/2 -> k + 1
        
        res = np.linalg.norm(np.dot(A,x) - b) # Residual
        
        if res < tol : # If tolerance is met than stop here
            break
    
    if nits == maxit - 1 :
        nits = -1
        
    return(x, nits)
    

if __name__ == "__main__" :
    n = A0.shape[0]
    b = random.uniform(size=(n,))
    x0 , nits0 = HSS(A0, b, alpha = 10, x0 = None, tol = 0.0001, maxit = 10000) 
    assert(nits0 != -1)


def theorem32(A, alpha, tol) :
    """
    Verifies if the sufficient convergence condition given in Theorem 3.2 is 
    verified for the inputed matrix A, parameter alpha and tolerance level tol.
    
    :param A: an nxn numpy array
    :param alpha: floating point number strictly positive, method's coefficient
    :param tol: floating point number, the tolerance for termination
    
    :return condition: a boolean value
    """
    
    n = A.shape[0] # Size of coefficient matrix
    I = np.identity(n)
    Ia = alpha * I
    
    Astar = A.conj().T
    H = (1/2) * (A + Astar) # HS splitting
    S = (1/2) * (A - Astar)
    
    theta = np.linalg.norm(np.dot(A,np.linalg.inv(Ia+S)))
    rho = np.linalg.norm(np.dot((Ia + S),np.linalg.inv(Ia + H)))
    
    s = []
    for lbda in np.linalg.eigvals(H) :
        s += [(alpha - lbda) / (alpha + lbda)]
    sigma = np.max(s)
    
    condition = (sigma + theta * rho * tol) * (1 + theta * tol) < 1
    
    return(condition)

if __name__ == "__main__" :
    print("")
    print("The IHSS convergence sufficient condition is here : " + str(theorem32(A0,3,0.0001)))
    print("")

def IHSS(A, b, alpha, x0, tol, maxit) :
    """
    For a matrix A, solve Ax=b using the IHSS method.

    :param A: an nxn numpy array
    :param b: n-dimensional numpy array
    :param alpha: floating point number strictly positive, method's coefficient
    :param x0: the initial guess (if not present, use b)
    :param tol: floating point number, the tolerance for termination
    :param maxit: integer, the maximum number of iterations

    :return x: an n-dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    """

        
    if x0 is None :
        x0 = b
    
    n = A.shape[0] # Size of coefficient matrix
    I = np.identity(n)
    Ia = alpha * I
    
    Astar = A.conj().T
    H = (1/2) * (A + Astar) # HS splitting
    S = (1/2) * (A - Astar)
    
    M1 = Ia + H
    M2 = Ia + S
    
    N1 = Ia - S
    N2 = Ia - H
    
    nits = -1 # Initializing number of iterations
    x = x0
    
    for k in range(maxit) :
        
        nits += 1
        
        # Step k -> k + 1/2
        r = b - np.dot(A,x)
        res = np.linalg.norm(r)
        x, _ = cla_utils.GMRES(M1, np.dot(N1,x) + b, maxit = 1000, tol = tol * res)
        
        # Step k + 1/2 -> k + 1
        r = b - np.dot(A,x)
        res = np.linalg.norm(r)
        x, _ = cla_utils.GMRES(M2, np.dot(N2,x) + b, maxit = 1000, tol = tol * res)
        
        r = b - np.dot(A,x)
        res = np.linalg.norm(r) # Residual
        if res < tol : # If tolerance is met than stop here
            break

    if nits == maxit - 1 :
        nits = -1
        
    return(x, nits)

if __name__ == "__main__" :
    
    x1 , nits1 = IHSS(A0, b, alpha = 3, x0 = None, tol = 0.0001, maxit = 10000) 
    assert(nits1 != -1)
    
    # Generating a random positive definite matrix with m = 10000
    m = 10
    d = random.randn(m)
    d = np.abs(d)
    D = np.diag(d) # Only strictly positive values on the diagonal
    B = random.uniform(size=(m,m))
    Q,_ = cla_utils.householder_qr(B) # Retreiving a random unitary matrix
    C = np.dot(Q.conj().T,np.dot(D,Q))
    
    c = random.uniform(size = (m,))

    print("")
    print("Is considered matrix C positive definite ? Answer : " + str(verifyPositiveDefinite(C)))
    print("")
    
    _ , nitsC = IHSS(C, c, alpha = 10, x0 = None, tol = 0.0001, maxit = 10000) 
    assert(nitsC != -1)


def timeable_HSS_4253():
    """
    Doing an example with the HSS solver function. Parameters are alpha = 3, 
    x0 = b, tol = 0.001 and maxit = 10000. Size is n = 4253.
    """
    _ , _ = HSS(A0, b, alpha = 3, x0 = None, tol = 0.001, maxit = 10000) 


def timeable_IHSS_4253():
    """
    Doing an example with the IHSS solver function. Parameters are alpha = 3, 
    x0 = b, tol = 0.001 and maxit = 1000. Size is n = 4253.
    """
    _ , _ = IHSS(A0, b, alpha = 3, x0 = None, tol = 0.001, maxit = 1000) 


def time_solver():
    """
    Get some timings for the solver algorithms.
    """

    print("Timing for HSS | Size 4253")
    print(timeit.Timer(timeable_HSS_4253).timeit(number=1))
    
    print("")
    
    print("Timing for IHSS | Size 4253")
    print(timeit.Timer(timeable_IHSS_4253).timeit(number=1))
    
    print("")
    
if __name__ == "__main__" :
    time_solver()





