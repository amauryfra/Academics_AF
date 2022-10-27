"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 4 - (c), (d), (e)

This script computes the solutions for questions (c), (d) and (e) of
the exercise 4, given in the Coursework 1 of the module MATH70024.

"""

import numpy as np
import cla_utils
from numpy import random
import timeit

# Question (c)

def x_computation(A,b,l) :
    """
    Given a real mxn matrix A, and a relevant lambda Lagrange multiplier,
    finds the vector x that minimizes ||Ax-b||^2, provided that ||x||=1.

    :param b: an m-dimensional numpy array
    :param l: a float number as lambda Lagrange multiplier
    
    :return x: an n-dimensional numpy array giving the vector minimizing
    ||Ax-b||^2
    """
    
    m,n = A.shape # Getting the shape of A
    Q,R = cla_utils.householder_qr(A) # QR factorisation
    
    I = np.identity(m) # Computing useful matrix
    R_t = R.transpose()
    A_t = A.transpose()
    Atb = np.dot(A_t,b) 
    
    
    Y = cla_utils.householder_solve((I + (1/l) * np.dot(R,R_t)), np.dot(R,Atb))
    # Computing the Y expression using Householder solver
    
    x = 1/l * Atb - (1/(l**2)) * np.dot(R_t,Y)
    # Computing x
    
    return(x)



# Question (d)

def lambda_search(l,A,b) :
    """
    Given a starting lambda Lagrange multiplier, the matrix A and the vector b, 
    this function computes a lambda such that x minimizes ||Ax-b||^2 
    with having ||x|| = 1. 

    :param l: float number representing the starting lambda Lagrange multiplier
    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return l: a float number, a lambda Lagrange multiplier such that 
        x_computation(A,b,l) returns an x with ||x|| = 1
    """
    
    x = x_computation(A,b,l) # Computing x
     
    norm = np.linalg.norm(x) # ||x||
    diff = abs(norm-1.0) # Difference between ||x|| and 1
    
    if diff < 1.0e-5 : # If the difference is smaller than 10^-5
        
        return(l) # We have the right lambda
    
    else :
        
        if norm > 1 :
            
            l = (1+diff)*l # Slightly increasing lambda 
            return(lambda_search(l,A,b)) # Recursive programming
        
        else :
            
            l = l/(1+diff) # Slightly decreasing lambda 
            return(lambda_search(l,A,b)) # Restarting the process
        
        
        
# Question (e)

l = 1
m = 10
random.seed(1878*m)
A = random.randn(m, m)
b = random.randn(m,1)

print("")

def timeable_lambda_search_10():
    """
    Doing an example with the lambda search function. Size is 10;
    """
    test = lambda_search(l,A,b)


print("Timing for lambda_search | A of size (10,10)")
print(timeit.Timer(timeable_lambda_search_10).timeit(number=1))

print("")

m = 100
A = random.randn(m, m)
b = random.randn(m,1)

def timeable_lambda_search_100():
    """
    Doing an example with the lambda search function. Size is 100;
    """
    test = lambda_search(l,A,b)
    
    
print("Timing for lambda_search | A of size (100,100)")
print(timeit.Timer(timeable_lambda_search_100).timeit(number=1))
    
print("")

m = 250
A = random.randn(m, m)
b = random.randn(m,1)

def timeable_lambda_search_250():
    """
    Doing an example with the lambda search function. Size is 250;
    """
    test = lambda_search(l,A,b)
    
print("Timing for lambda_search | A of size (250,250)")
print(timeit.Timer(timeable_lambda_search_250).timeit(number=1))

print("")



