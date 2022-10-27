import numpy as np

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """
    A1 = A # Not performing algorithm in place
    m = A1.shape[0]
        
    x = A1[:,0]
    e1 = np.eye(1,m,0,dtype = complex)[0] # e1
        
    if np.sign(x[0]) == 0 : # Taking care of this specific case separately 

        v = np.linalg.norm(x) * e1 + x # np.sign(x[0]) set to 1
        v = v / np.linalg.norm(v) # Normalized by 2-Norm
            
        A1 = A1 - 2 * np.outer(v,np.dot(v.transpose(),A1)) # Householder process
        A1 = A1 - 2 * np.outer(np.dot(A1,v),v.transpose()) # Right multiplying
            
    else :
            
        v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
        v = v / np.linalg.norm(v)
            
        A1 = A1 - 2 * np.outer(v,np.dot(v.transpose(),A1)) 
        A1 = A1 - 2 * np.outer(np.dot(A1,v),v.transpose()) # Right multiplying
        
        
    return(A1)



def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m = A.shape[0]
    
    for k in range(m-2) :
        
        x = A[k+1:,k]
        e1 = np.eye(1,m-k-1,0,dtype = complex)[0] # e1 of same size as sub-column A[k:m,k]
        
        if np.sign(x[0]) == 0 : # Taking care of this specific case separately 
    
            v = np.linalg.norm(x) * e1 + x # np.sign(x[0]) set to 1
            v = v / np.linalg.norm(v) # Normalized by 2-Norm
                
           
            A[k+1:,k:] = A[k+1:,k:] - 2 * np.outer(v,np.dot(v.transpose(),A[k+1:,k:])) # As in pseudo-code
            A[k+1:,k+1:] = A[k+1:,k+1:] - 2 * np.outer(np.dot(A[k+1:,k+1:],v),v.transpose()) # Right multiplying
            
        else :
                
            v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
            v = v / np.linalg.norm(v)
                
            A[k+1:,k:] = A[k+1:,k:] - 2 * np.outer(v,np.dot(v.transpose(),A[k+1:,k:])) # As in pseudo-code
            A[k+1:,k+1:] = A[k+1:,k+1:] - 2 * np.outer(np.dot(A[k+1:,k+1:],v),v.transpose()) # Right multiplying
            

def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """

    m = A.shape[0]
    Q = np.identity(m)
    
    for k in range(m-2) :
        
        x = A[k+1:,k]
        e1 = np.eye(1,m-k-1,0)[0] # e1 of same size as sub-column A[k:m,k]
        
        if np.sign(x[0]) == 0 : # Taking care of this specific case separately 
    
            v = np.linalg.norm(x) * e1 + x # np.sign(x[0]) set to 1
            v = v / np.linalg.norm(v) # Normalized by 2-Norm
                
           
            A[k+1:,k:] = A[k+1:,k:] - 2 * np.outer(v,np.dot(v.transpose(),A[k+1:,k:])) # As in pseudo-code
            A[:,k+1:] = A[:,k+1:] - 2 * np.outer(np.dot(A[:,k+1:],v),v.transpose()) # Right multiplying
        
            Qk = np.identity(m)
            Qk[k+1:,k+1:] -= 2 * np.outer(v,v.transpose())
            Q = np.dot(Qk,Q)
            
        else :
                
            v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
            v = v / np.linalg.norm(v)
                
            A[k+1:,k:] = A[k+1:,k:] - 2 * np.outer(v,np.dot(v.transpose(),A[k+1:,k:])) # As in pseudo-code
            A[:,k+1:] = A[:,k+1:] - 2 * np.outer(np.dot(A[:,k+1:],v),v.transpose()) # Right multiplying
        
            Qk = np.identity(m)
            Qk[k+1:,k+1:] -= 2 * np.outer(v,v.transpose())
            Q = np.dot(Qk,Q)
        
    return(Q.transpose())
        

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """
    Q = hessenbergQ(A) # Transforming A to upper Hessenberg, retreiving Q
    V = hessenberg_ev(A) # Retreiving eigenvectors of H 
    V = np.dot(Q, V) # Transforming eigenvectors to those of A
    return(V)

import matplotlib.pyplot as plt

def rayleighQuotientAccuracy(m) :
    """
    Given an integer m, computes a random Hermitian matrix, finds an eigenvector v,
    computes a random perturbation vector r, and further compares the Rayleigh quotient 
    of v + eps * r with the related eigenvalue λ. 
    
    :param m: an integer 
    """
    
    B = np.random.random((m,m)) + np.random.random((m,m)) * 1j
    A = (B + B.conj().T)/2 # Computing Hermitian matrix
    
    w, V = np.linalg.eig(A)
    lbda = w[0] # Eigenvalue
    v = V[:, 0] # Eigenvector
    
    r0 = np.random.random(m) + np.random.random(m) * 1j # Initial perturbation vector
    
    epsilons = np.logspace(-8, -2, 100)
    errors = []
    
    for eps in epsilons :
    
        r = eps * r0 + v # Perturbation vector
        rayleigh = np.dot(r.conj().T,np.dot(A,r)) / np.dot(r.conj().T,r) # Rayleigh quotient 
        errors += [np.abs(rayleigh-lbda)]
    
    errors = np.array(errors)
    plt.loglog(epsilons,errors,'.')
    a = errors[20]/(epsilons[20] * epsilons[20])
    y = a * pow(epsilons,2)
    plt.loglog(epsilons, y)
    plt.show
    



    
    
    
    
    
    
    