import numpy as np
from numpy import random


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(random.randn(m, m))
    
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        
        Q1 = randomQ(m)
        R1 = randomR(m)
        
        A = np.dot(Q1,R1) # Forming A = Q1R1
        
        Q2 , R2 = np.linalg.qr(A) # Built-in numpy QR factorisation
        print("")
        print("Test number " + str(k+1))
        disp1 = "||Q2-Q1||=" + str(np.linalg.norm(Q2 - Q1))
        print(disp1) # Printing the norms
        disp2 = "||R2-R1||" + str(np.linalg.norm(R2 - R1))
        print(disp2)
        disp3 = "||A-Q2R2||=" +str(np.linalg.norm(A - np.dot(Q2,R2)))
        print(disp3)
        print("")



def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix 
    and b is an m dimensional vector.

    :param R: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """
    
    m, _ = R.shape # Getting the shape of R
    
    x = np.zeros(m) # Initializing x
    
    x[m-1] = b[m-1] / R[m-1,m-1] # First value of back substitution
    
    for i in reversed(range(m-1)) : # Iterative substitution (backwards)
        
        # np.dot(R[i,i+1:],x[i+1:]) is the pseudo-code sum as a dot product
        
        x[i] = ( b[i] - np.dot(R[i,i+1:],x[i+1:]) ) / R[i,i]
    
    return(x)
        
import sys   
sys.float_info # Representation of numbers in current OS

def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    
    epsilon = 2.220446049250313e-16 # Machine epsilon in my OS
    
    for k in range(20):
        
        A = random.randn(m, m)
        R = np.triu(A) # Random R
        
        y1 = random.randn(m) # Random vector
        x = solve_R(R, y1) # Solving Rx = y1 by back subsitution
        
        y2 = np.dot(R,x) # ||dR|| = ||y2-y1||
        
        norm = np.linalg.norm(y2-y1) / (np.linalg.norm(x) * np.linalg.norm(R))
        
        print("")
        print("Test number " + str(k+1))
        disp1 = "||dR||/||R||=" + str(norm)
        print(disp1)
        disp2 = "||dR||/||R||ε=" + str(norm/epsilon)
        print(disp2)
        print("")
        

import cla_utils.exercises3
import copy

def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    
    epsilon = 2.220446049250313e-16 # Machine epsilon in my OS
    
    for k in range(20):
        
        A = random.randn(m, m)
        A_copy = copy.deepcopy(A) # A is modified in place
        
        b1 = random.randn(m)
        
        x = cla_utils.exercises3.householder_solve(A_copy, b1)

        b2 = np.dot(A,x)
        b2 = b2.transpose()[0] # b2 as 1-D line array rather than column vector

        
        norm = np.linalg.norm(b2-b1) / (np.linalg.norm(x) * np.linalg.norm(A))
        
        print("")
        print("Test number " + str(k+1))
        disp1 = "||ΔA||/||A||=" + str(norm)
        print(disp1)
        disp2 = "||ΔA||/||A||ε=" + str(norm/epsilon)
        print(disp2)
        print("")

if __name__ == "__main__" :
    back_stab_householder_solve(10)



