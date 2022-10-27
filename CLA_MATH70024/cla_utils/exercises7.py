import numpy as np

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    
    p[i], p[j] = p[j], p[i] # Swapping p[i] and p[j]
        

def LUP_inplace(A, count_swaps = False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array
    :param count_swaps: boolean value, enables outputting the number of 
        swaps used in permutations

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """
                     
    m, _ = A.shape # Getting m
    
    p = [s for s in range(m)] # Representing the identity at this stage
    
    swaps_counts = 0 # Initializing the count of swaps needed
    
    for k in range(m-1) :
        
        i = np.argmax(np.abs(A[k:,k])) + k # Index of the max value of |u_ik| for i >= k 
        
        if i != k :
            swaps_counts += 1 # A swap is used
        
        A[[i, k]] = A[[k, i]] # Swapping rows i and k 
        
        perm(p, k, i) # Computing p_k
        
        # As in LU_inplace(A) (exercises6.py)
        L = A[k+1:,k] / A[k,k] # Temporary storage
        A[k+1:,k:] = A[k+1:,k:] - np.outer(L,A[k,k:]) # As in pseudo-code
        A[k+1:,k] = L
    
    if count_swaps : # If the user asked the number of swaps, managing the right \
        # execution of test function
        return(p,swaps_counts)
    
    return(p)


import cla_utils


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
    
    m = A.shape[0] # Retreiving m
    
    p = LUP_inplace(A) # In-place LUP decomposition
    
    # Extracting L and U (as seen in test code)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    
    Pb = b[p] # Permutation
    Pb = Pb.reshape(m,1) # Managing shape problems since 
        # some previous functions use an mxk-dimensional numpy array with several b_i
    
    y = cla_utils.solve_L(L, Pb) # Solving L(Ux) = Pb, with y = Ux

    x = cla_utils.solve_U(U, y) # Solving Ux = y
    
    x = x.reshape(m) # For test function to work

    return(x)           

    
    
def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
                     
    m = A.shape[0] # Retreiving m
    
    # In-place LUP decomposition with swaps count
    p,  numberSwaps = LUP_inplace(A, count_swaps = True) 
    
    # Extracting L and U (as seen in test code)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    
    detA = U.diagonal().prod() # Product of diagonal coefficients 
    
    # Determinant of P is (-1)**numberSwaps
    
    detA *= (-1)**numberSwaps # detA = det(P) * det(L) * det(U) with det(L) = 1
    
    return(detA)


    
