"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 2  - (a), (b)

This script computes the solutions for questions (a) and (b) of
the exercise 2, given in the Coursework 1 of the module MATH70024.

"""

import numpy as np
import cla_utils
import copy 
import matplotlib.pyplot as plt
import scipy


# Question (a)

x_axis = np.arange(0., 1.0001, 1./51) # Getting x_i

# Building V (Vandermonde matrix)
V = np.zeros((52,13),dtype=float)
for i in range(52) :
    for j in range(13) :
        V[i,j] = x_axis[i]**j


# Building b containing the f_i
b = np.zeros((52,1),dtype=float)
b[0] = 1
b[50] = 1

x_householder = cla_utils.householder_ls(V, b) # Solving the least squares problem \
    # by Householder factorisation


def polynomial(xValues,polyCoeffs) :
    """
    Computes the polynomial, defined by its coefficients in polyCoeffs, over the values 
    in xValues.

    :param xValues: an m-dimensional numpy array containing the values on which we will 
        compute the polynomial.
    :param polyCoeffs: an n-dimensional numpy array containing the coefficients of the polynomial,
        the first value being a_0 = P(0).

    :return polyValues: an m-dimensional numpy array containing the values \
         taken by the polynomial over xValues.
    """
    
    m = xValues.shape[0]
    n = polyCoeffs.shape[0]
    
    polyValues = np.zeros(m,dtype=float) # Initializing the vector containing the \
        # computations of the polynomial over xValues
    
    for i in range(m) :
        p = 0 # Erasing at each step
        for j in range(n) :
            p += polyCoeffs[j] * (xValues[i] ** j) # Computing the polynomial
        polyValues[i] = p
        
    return(polyValues)


polyValues = polynomial(x_axis,x_householder) # Computing the polynomial over the same values as f_i
plt.figure(figsize=(12, 8))
plt.plot(x_axis,b, label = 'function f') # Plotting both f and polynomial
plt.plot(x_axis,polyValues, label = 'polynomial')
plt.legend(loc="upper left")
plt.title("Polynomial of degree 12 fitting the function f, using the least squares method, performed by a Householder QR factorisation.")
plt.show()

#  Solving the least squares problem by GS factorisation
V_GS = copy.deepcopy(V)
R_GS = cla_utils.GS_classical(V_GS)
x_GS = scipy.linalg.solve_triangular(np.real(R_GS),np.real(V_GS).transpose().dot(b)) 
# Taking real parts to avoid computations errors 

polyValues = polynomial(x_axis,x_GS) # Computing the polynomial over the same values as f_i
plt.figure(figsize=(12, 8))
plt.plot(x_axis,b, label = 'function f') # Plotting both f and polynomial
plt.plot(x_axis,polyValues, label = 'polynomial')
plt.legend(loc="upper left")
plt.title("Polynomial of degree 12 fitting the function f, using the least squares method, performed by a Gram-Schmidt QR factorisation.")
plt.show()

#  Solving the least squares problem by GS modified factorisation
V_GSm = copy.deepcopy(V)
R_GSm = cla_utils.GS_modified(V_GSm)
x_GSm = scipy.linalg.solve_triangular(np.real(R_GSm),np.real(V_GSm).transpose().dot(b))

polyValues = polynomial(x_axis,x_GSm) # Computing the polynomial over the same values as f_i
plt.figure(figsize=(12, 8))
plt.plot(x_axis,b, label = 'function f') # Plotting both f and polynomial
plt.plot(x_axis,polyValues, label = 'polynomial')
plt.legend(loc="upper left")
plt.title("Polynomial of degree 12 fitting the function f, using the least squares method, performed by a Modified Gram-Schmidt QR factorisation.")
plt.show()



# Question (b)

print("")

MAX_VALUE_GSm_minus_HOUSE = np.max(np.abs(np.matrix.flatten(x_GSm)-x_householder))
print('Maximum difference between coefficients given by GS modified and Householder : ' \
      + str(MAX_VALUE_GSm_minus_HOUSE ))

print("")

MAX_VALUE_GS_minus_HOUSE = np.max(np.abs(np.matrix.flatten(x_GS)-x_householder))
print('Maximum difference between coefficients given by GS and Householder : ' \
      + str(MAX_VALUE_GS_minus_HOUSE ))

print("")   
    
ORTHO_TEST_GS = np.linalg.norm(V_GS.transpose().dot(V_GS)-np.identity(V_GS.shape[1]))
print('||QQ^* - I|| for Q given by the GS method : ' \
      + str(ORTHO_TEST_GS))
    
print("")
    
ORTHO_TEST_GSm = np.linalg.norm(V_GSm.transpose().dot(V_GSm)-np.identity(V_GSm.shape[1]))
print('||QQ^* - I|| for Q given by the GS modified method : ' \
      + str(ORTHO_TEST_GSm))  
    
print("")


