"""
Imperial College London
Computational Linear Algebra (MATH70024)
Coursework 1

Amaury Francou | CID : 01258326 | amaury.francou16@imperial.ac.uk
MSc Applied Mathematics 

Exercise 1  - (a), (b)

This script computes the solutions for questions (a) and (b) of
the exercise 1, given in the Coursework 1 of the module MATH70024.

"""

import numpy as np
import cla_utils
import copy 
from sys import getsizeof

# Question (a)

C = np.loadtxt('./cw1/C.dat', delimiter=',') # Loading data
C_copy = copy.deepcopy(C) # Copy to another memory location as some cla_utils \
    # functions change the data 'in place'

Q,R = cla_utils.householder_qr(C) # Q, R factorisation 

MAX_VALUE_UNDER_LINE_3 = np.max(np.abs(R[3:,:]))

print("")

print('Maximum absolute value within elements of R (strictly) under line 3 : ' \
      + str(MAX_VALUE_UNDER_LINE_3 ))

print("")


# Question (b)

def compress_C(C) :
    """
    Returns a compressed version of matrix C by performing QR factorisation
    and dropping irrelevantly small values columns.

    :param C: an 1000x100-dimensional numpy array containing the samples' values 

    :return Q_compress: an 1000x3-dimensional numpy array containing the relevant \
         values of Q.
    :return R_compress: 3x100-dimensional numpy array conntaining the relevant \
         values of R.
    :return C_compress: 1000x100-dimensional numpy array being the compressed \
         version of C.
    """
    
    Q,R = cla_utils.householder_qr(C) # QR factorisation
    
    Q_compress = Q[:,:3] # Keeping only useful information
    R_compress = R[:3,:]
    
    C_compress = Q_compress.dot(R_compress) # Compressed version of C
    
    return(Q_compress,R_compress,C_compress)



Q_compress,R_compress,_ = compress_C(C)

SIZE_C = getsizeof(C_copy) # Size in bytes of the memory storage of C
SIZE_Q_compress = getsizeof(Q_compress) # Size in bytes of the memory storage of Q_compress
SIZE_R_compress = getsizeof(R_compress) # Size in bytes of the memory storage of R_compress
SIZE_C_compress = SIZE_Q_compress + SIZE_R_compress # Size in bytes of the memory \
    # storage of C compressed, seen as the pair (Q_compress,R_compress)
    
print("Size in bytes of C : " + str(SIZE_C))
print("Size in bytes of the compressed version of C by the QR factorisation method : " \
      + str(SIZE_C_compress))

print("")
