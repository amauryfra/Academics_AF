import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k-1 dimensional numpy array.
    :return Lk: an mxm dimensional numpy array.

    """

    k = m - lvec.shape[0] - 1 # Retreiving k
    
    Lk = np.identity(m) # Initializing Lk
    
    Lk[k+1:,k] = -lvec # Inserting lvec
    
    return(Lk)

# Testing

from numpy import random

def test_Lk_props() :
    """
    Tests the multiplication and inverse propreties of the L_k
    """
    for m in range(2,50) : # Going through several tests
    
        ka = random.randint(m-1)
        kb = ka + 1 # For consecutive Lk
        
        # Inverse properties
        lka = np.zeros(m)
        lka[ka+1:] = random.randn(m-ka-1)
        Lka = get_Lk(m, lka[ka+1:]) # Building Lk
        Lka_inv = np.identity(m) + np.outer(lka,np.eye(1,m,ka)) # I + lkek*
        Lka_true_inv = np.linalg.inv(Lka) # Numpy inverse
        assert(np.linalg.norm(Lka_inv-Lka_true_inv) < 1.0e-6)
        
        # Product of inverses properties
        lkb = np.zeros(m)
        lkb[kb+1:] = random.randn(m-kb-1)
        Lkb = get_Lk(m, lkb[kb+1:]) # Building Lk
        Lkb_true_inv  = np.linalg.inv(Lkb) # Numpy inverse
        Inv_product_true = np.dot(Lka_true_inv,Lkb_true_inv) # Numpy product
        Inv_product = np.identity(m) + np.outer(lka,np.eye(1,m,ka)) \
            + np.outer(lkb,np.eye(1,m,kb))
        # I + lkek* + l(k+1)e(k+1)*
        assert(np.linalg.norm(Inv_product-Inv_product_true) < 1.0e-6)



def LU_inplace(A):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    
    m = A.shape[0] # Getting m
    
    for k in range(m-1) :
        
        L = A[k+1:,k] / A[k,k] # Temporary storage
        A[k+1:,k:] = A[k+1:,k:] - np.outer(L,A[k,k:]) # As in pseudo-code
        A[k+1:,k] = L
    
    return(A)



def solve_L(L, b):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """

    m, k = b.shape # Retreiving m and k
    
    x = np.zeros((m,k)) # Initializing x
    
    x[0,:] = b[0,:] / L[0,0] # First value of substitution
    
    for i in range(1,m) : # Iterative substitution (forward)
        
        # np.dot(L[:,:i],x[:i,:])[i,:] As in pseudo-code with sum as a dot product
        
        x[i,:] = ( b[i,:] - np.dot(L[:,:i],x[:i,:])[i,:] ) / L[i,i]
        
    return(x)


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
                     
    m, k = b.shape # Retreiving m and k
    
    x = np.zeros((m,k)) # Initializing x
    
    x[m-1,:] = b[m-1,:] / U[m-1,m-1] # First value of back substitution
    
    for i in reversed(range(m-1)) : # Iterative substitution (backwards)
        
        # np.dot(U[:,i+1:],x[i+1:,:]) As in pseudo-code with sum as a dot product
        
        x[i,:] = ( b[i,:] - np.dot(U[:,i+1:],x[i+1:,:])[i,:] ) / U[i,i]
        
    return(x)


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """
                     
    # We retreive the k-th column of A^-1 using the fact that AA^(-1)_k = e_k, \
        # where e_k is the k-th orthonormal basis vector
    # We furthermore have A = LU => LUA^(-1) = I which can be seen as a system \ 
        # of equations involving the e_k
    # By noting Y = UA^(-1) we first solve LY = I and then solve Y = UA^(-1)
    
    m = A.shape[0] # Retreiving m
    
    I = np.identity(m)
    
    LU_inplace(A) # In-place LU decomposition
    
    # Extracting L and U (as seen in test code)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    
    Y = solve_L(L, I) # Solving LY = I
    
    Ainv = solve_U(U, Y) # Solving UA^(-1) = Y
    
    return(Ainv)
        
        
        
        
