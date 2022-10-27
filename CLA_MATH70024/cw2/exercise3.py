"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 2

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 3 - (c), (d)

This script computes the solutions for questions (c) and (d) of
the exercise 3, given in the Coursework 2 of the module MATH70024.

"""

import numpy as np

# Question (c) 

def custom_eigenvalues(A) :
    """
    Given a complex 2x2 matrix A, finds its eigenvalues using the classical 
    quadratic formula applied to the characteristic polynomial of A.

    :param A: an 2x2-dimensional numpy array

    :return eigenValues: an 2-dimensional numpy array containing the eigenvalues
        of A
    """
    
    disc = np.sqrt( (A[0,0] + A[1,1])**2 - \
                4 * (A[0,0] * A[1,1] - A[0,1] * A[1,0]) ) # Discriminant

    eigenValues = np.array([ ( A[0,0] + A[1,1] + disc ) / 2 , \
                            ( A[0,0] + A[1,1] - disc ) / 2]) # As given by polynomial roots
    
    return(eigenValues)

print("")
# Testing on A1 and A2 
A1 = np.array([[1,0],[0,1]], dtype=complex)
print("A1 eigenvalues :")
print(custom_eigenvalues(A1))
A2 = np.array([[1 + 1.0e-14,0],[0,1]],dtype=complex)
print("A2 eigenvalues :")
print(custom_eigenvalues(A2))

print("")

# Quantifying errors
err1 = np.linalg.norm( custom_eigenvalues(A1) - np.array([1,1]) )
print("Error made by our custom computation of A1 eigenvalues :")
print(err1)
err2 = np.linalg.norm( custom_eigenvalues(A2) - np.array([1 + 1.0e-14,1]) )
print("Error made by our custom computation of A2 eigenvalues :")
print(err2)



# Question (d) 

import sys   
machine_eps  = sys.float_info[8] # Representation of numbers in current OS
print("")
print("Value of √ε in current system :")
print(machine_eps**0.5)
print("")






