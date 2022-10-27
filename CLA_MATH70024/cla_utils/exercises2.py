import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    
    m,n = Q.shape # Retreiving the number of orthonormal vectors and their size
    
    u = np.empty(n, dtype=np.complex128) # Initialising vector u which will contain (u_1,...,u_n)
    
    v_star = np.conj(v) # Getting the conjugate of v
    
    v_proj = np.zeros(m, dtype=np.complex128) # Initialising vector u_1q_1 + u_2q_2 + ... + u_nq_n
    
    
    for i in range(n) : # Computing the u_i and building v_proj
    
        u[i] = v_star.dot(Q[:,i]) # Projection coefficient
        
        v_proj += u[i] * Q[:,i] # By definition of v_proj
    
    
    r = v - v_proj # By definition of r

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    Q_star = np.conj(Q) # Getting the adjoint of Q
    Q_star = np.transpose(Q_star)
    
    x = Q_star.dot(b) # As Q^-1 = Q*

    return x



import timeit
import numpy.random as random

random.seed(1651)

A0 = random.randn(100,100)
Q0, R0 = np.linalg.qr(A0)
b0 = random.randn(100)
A1 = random.randn(200,200)
Q1, R1 = np.linalg.qr(A1)
b1 = random.randn(200)
A2 = random.randn(400, 400)
Q2, R2 = np.linalg.qr(A2)
b2 = random.randn(400)


def timeable_numpy_solving_100():
    """
    Doing an example with the builtin numpy solver function. Size is 100;
    """
    
    test = np.linalg.solve(Q0, b0)
    
def timeable_numpy_solving_200():
    """
    Doing an example with the builtin numpy solver function. Size is 200.
    """
    
    test = np.linalg.solve(Q1, b1)

def timeable_numpy_solving_400():
    """
    Doing an example with the builtin numpy solver function. Size is 400.
    """
    
    test = np.linalg.solve(Q2, b2)
    
def timeable_custom_solving_100():
    """
    Doing an example with the solveQ solver function. Size is 100.
    """
    
    test = solveQ(Q0, b0)
    
def timeable_custom_solving_200():
    """
    Doing an example with the solveQ solver function. Size is 200.
    """
    
    test = solveQ(Q1, b1)
    
def timeable_custom_solving_400():
    """
    Doing an example with the solveQ solver function. Size is 400.
    """
    
    test = solveQ(Q2, b2)

def time_solver():
    """
    Get some timings for solver algorithms.
    """

    print("Timing for solveQ | Size 100")
    print(timeit.Timer(timeable_custom_solving_100).timeit(number=1))
    print("Timing for built-in solver function | Size 100")
    print(timeit.Timer(timeable_numpy_solving_100).timeit(number=1))
    
    print("")
    
    print("Timing for solveQ | Size 200")
    print(timeit.Timer(timeable_custom_solving_200).timeit(number=1))
    print("Timing for built-in solver function | Size 200")
    print(timeit.Timer(timeable_numpy_solving_200).timeit(number=1))
    
    print("")
    
    print("Timing for solveQ | Size 400")
    print(timeit.Timer(timeable_custom_solving_400).timeit(number=1))
    print("Timing for built-in solver function | Size 400")
    print(timeit.Timer(timeable_numpy_solving_400).timeit(number=1))



# As size increases, the custom solver becomes more efficient than the built-in one.



def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    Q_star = np.conj(Q) # Getting the adjoint of Q
    Q_star = np.transpose(Q_star)

    P = Q.dot(Q_star) # Theorem 1.28

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """
    
    m,n = V.shape # m is the total size of the vector space

    Q0,R0 = np.linalg.qr(V, mode="complete") # Built-in complete QR factorisation
    
    V_free = np.empty((m,0), dtype=np.complex128) # Building a linearly independent \
        # set of vectors u_i,...,u_j 
        
    for i in range(n) : # Going through the set u_1,...,u_n
        try :
            if R0[i,i] != 0 :
                V_free = np.column_stack((V_free,V[:,i])) # Dropping vectors u_i for \
                    # which ri,i = 0
        except :
            continue
    
    m,n1 = V_free.shape # n1 is the size of the vector space spanned by u_1,...,u_n
    
    assert n1 <= m # If n1 = m the subspace orthogonal to U is (0), which may \
        # generate errors on the testing script

    Q1,R1 = np.linalg.qr(V_free, mode="complete") # Built-in complete QR factorisation
        
    Q = Q1[:,n1:m] # The remaining columns form the orthonormal basis of \
        # the subspace orthogonal to U
    
    if Q.size == 0 : # If n1 == m
        
        Q = np.zeros(m, dtype=np.complex128) # The subspace orthogonal to U is (0)
    
    return Q



def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    
    m,n = A.shape # Retreiving m and n
    
    R = np.zeros((n,n), dtype=np.complex128) # Initialising R

    # First step of the GS algorithm
    R[0,0] = np.linalg.norm(A[:,0])
    A[:,0] = A[:,0] / R[0,0] # Modifying A directly 
    
    # Second step of the GS algorithm
    # Needs to be outside of for loop because of problematical Numpy slices returns \
        # when a zero is involved at the beginning of the slice
    R[0,1] = A.transpose().conjugate()[0,:].dot(A[:,1])
    A[:,1] = A[:,1] - R[0,1] * A[:,0]
    R[1,1] = np.linalg.norm(A[:,1]) # 2-Norm
    A[:,1] = A[:,1] / R[1,1]
    
    # Step j of the GS algorithm
    for j in range(2,n) :
    
        R[:j-1,j] = A.transpose().conjugate()[:j-1,:].dot(A[:,j])[0] # This dot \
            # product function returns an one-element array
        A[:,j] = A[:,j]  - A[:,:j-1].dot(R[:j-1,j]) # As in pseudo-code \
            # but using matrix multiplication and Numpy slices
        
        R[j,j] = np.linalg.norm(A[:,j]) # As in pseudo-code
        A[:,j] = A[:,j] / R[j,j]
        
        
    return R



def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    
    m,n = A.shape # Retreiving m and n
    
    R = np.zeros((n,n), dtype=np.complex128) # Initialising R
    
    for i in range(n) : # Only one loop can be used, using Numpy slices and \
        # matrix multiplication
        
        R[i,i] = np.linalg.norm(A[:,i]) # 2-Norm
        A[:,i] = A[:,i] / R[i,i] # Modifying A directly 
        
        R[i,i+1:] = A[:,i+1:].transpose().dot(A[:,i].conjugate()) # As in pseudo-code \
            # but using matrix multiplication and Numpy slices
        A[:,i+1:] = A[:,i+1:] - np.outer(A[:,i],R[i,i+1:])
        
    return R

import copy
# Testing on square matrix
m_ = 200 # Tryout m=2, m=3, m=50, m=100, ...
A3 = random.randn(m_, m_) + 1j*random.randn(m_, m_)

def test_orthog(A3_ = A3) :
    
    A4 = copy.deepcopy(A3_)
    
    m = A3_.shape[0] 
    
    GS_classical(A3)
    GS_modified(A4)
    
    # If qi and qj are orthogonal, (Q^*Q)i,j for i!=j should be perfectly equal to 0
    # The slight difference represents the errors propagated by the algorithm
    
    print("Q^*Q-I for classical GS :")
    print(A3_.transpose().conjugate().dot(A3_)-np.identity(m))
    print("")
    print("||Q^*Q-I|| for classical GS :")
    print(np.linalg.norm(A3_.transpose().conjugate().dot(A3_)-np.identity(m)))
    
    print("")
    print("")
    
    print("Q^*Q-I for modfied GS :")
    print(A4.transpose().conjugate().dot(A4)-np.identity(m))
    print("")
    print("||Q^*Q-I|| for modfied GS :")
    print(np.linalg.norm(A4.transpose().conjugate().dot(A4)-np.identity(m)))
    
    # Seems that ||Q^*Q-I|| for classical GS  >> ||Q^*Q-I|| for modfied GS as m increases



# Taken from test_exercises2.py and modifying D
m_ = 10
A5 = random.randn(m_, m_) + 1j*random.randn(m_, m_)
U0, _ = np.linalg.qr(A5)
A5 = random.randn(m_, m_) + 1j*random.randn(m_, m_)
V0, _ = np.linalg.qr(A5)

p_ = random.rand() # A probability in [0,1]
Z = np.random.choice([0, 1], size=(m_,), p=[p_, 1-p_]) # An array filled with 0 \
    # and 1s chosen randomly with probability p ,1-p 

Z = 1.0*Z # Filled with floats
for i in range(len(Z)) :
    if Z[i] == 0 : # Elements of Z initialised to 0 will take large random real numbers
        Z[i] = random.rand() * random.randint(1000, 10000)
    else :
        Z[i] = 1.0 + 0.1 * random.rand() # Elements of Z initialised to 1 will take \
            # small random real numbers

D0 = np.diag(Z) # Diagonal with small and large numbers

A5 = np.dot(U0, np.dot(D0, V0))

#print(A5)
#print("")
#test_orthog(A5) 

# Run test_orthog(A5) to notice that ||Q^*Q-I|| for classical GS  >>> ||Q^*Q-I|| for modfied GS \
    # under such a D
    

def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    m,n = A.shape # Retreiving m and n
    
    R = np.identity(n, dtype=np.complex128) # Initialising R to identity
    
    R[k,k] = 1 / np.linalg.norm(A[:,k]) # First modified coefficient 
    
    if k < n :
    
        for i in range(k+1,n) :
            
            R[k,i] = - (A[:,k].transpose().conjugate().dot(A[:,i])) / np.linalg.norm(A[:,i]) 
            # Getting the rest of the row's coefficients - R[s,s+1] / R[s,s]
    
    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
