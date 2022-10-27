import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """
    
    m,n = A.shape # Getting the shape of matrix A
    assert n == x.shape[0] # Verifying the multiplication is possible
    
    
    b = np.zeros(m, dtype=np.complex128) # Initialising vector b
    
    for i in range(m) : # Going through lines
        for j in range(n) : # Going through columns
            b[i] += A[i,j] * x[j] # Result (1.1) of the lecture notes

    return(b)

    
    
    


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in 
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """
    
    m,n = A.shape # Getting the shape of matrix A
    assert n == x.shape[0] # Verifying the multiplication is possible
    
    b = np.zeros(m, dtype=np.complex128) # Initialization of vector b
    
    for i in range(n) : # Going through column vectors of A
        b += A[:,i] * x[i] # Linear combination of the columns of A
        
    return(b)
    



def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u2: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """

    B = np.column_stack((u1,u2)) # Creating an m x 2 matrix with u1 and u2 as columns
    C = np.column_stack((v1,v2)) # Creating an n x 2 matrix with v1 and v2 as columns
    C = C.transpose()
    C = C.conjugate() # Transforming C in its adjoint matrix
        
    # The product BC now gives u1v1^*+u2v2^*

    A = B.dot(C)
    
    r = np.linalg.matrix_rank(A) # Getting the rank of matrix A
    assert r == 2 # Asserting it is indeed equal to 2
    
    # This multiplication creates m column-vectors of C^n that are some linear combination \
        # of u1 and u2. Thus, if u1 and u2 or not collinear then they span a vector space that has \
            # a dimension = 2. This will also be the case of matrix A, as we will be able to sort out \
                # only two independent vectors of it. Moreover, u1 and u2 being \
                    # chosen randomly, we can assert that they are not going to be collinear here.

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """
    
    # By multiplying A = (I + uv^*) with B = (I - (1 / (1 + (v^*)u)) uv^*), we \
        # obtain AB = I and BA = I which gives us the result. Thus, we have \
            # alpha = -(1 / (1 + (v^*)u)).
    
    m = u.shape[0] # Getting the length of the vectors
    Ainv = np.identity(m, dtype=np.complex128) # Initialization of matrix Ainv to Ainv = I
    
    v = v.transpose()
    v = v.conjugate() # Transforming v in its adjoint vector
    
    alpha = - 1 / (1 + v.dot(u)) # Computing alpha
    
    Ainv += alpha * np.outer(u,v) # As found previously 
    
    return Ainv


u0 = random.randn(400)
v0 = random.randn(400)
A1 = random.randn(400, 400)

def timeable_numpy_inversion():
    """
    Doing an example with the builtin numpy inverse function.
    """
    
    test = np.linalg.inv(A1)
    
def timeable_custom_inversion():
    """
    Doing an example with the rank1pert_inv inverse function.
    """
    
    test = rank1pert_inv(u0, v0)

def time_inverse():
    """
    Get some timings for inversion algorithms.
    """

    print("Timing for rank1pert_inv")
    print(timeit.Timer(timeable_custom_inversion).timeit(number=1))
    print("Timing for built-in inverse function")
    print(timeit.Timer(timeable_numpy_inversion).timeit(number=1))

    # Our custom inversion algorithm seems faster than the built-in one. However, \
        # it can be used only for matrix that are of the form A = I +uv^*


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i>=j and Ahat[i,j] = C[i,j] for i<j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """
    
    # A = B + iC, A^* = (B + iC)^* = B^t - iC^t = A = B + iC (since A is Hermitian) \
        # => (B^t-B) + i(C^t+C) = 0 which gives us the result.
    
    m = xr.shape[0] # Getting the shape (length) of vector x
    
    zr = np.zeros(m, dtype=np.float64) # Initialising vector zr
    zi = np.zeros(m, dtype=np.float64) # Initialising vector zi
    
    
    for i in range(m) : # Looping over the entries of x
    
        B_column = np.concatenate([Ahat[i,:i], Ahat[i:,i]]) # Retrieving the columns of B through Ahat
        C_column = np.concatenate([Ahat[:i,i], -Ahat[i,i:]]) # Retrieving the columns of C through Ahat
        C_column[i] = 0 # We took the hole line of Ahat including the diagonal. As C is anti-symmetric \
            # c_i,i = 0 for all i
        
        zr += xr[i] * B_column - xi[i] * C_column # Computing zr and zi
        
        zi += xi[i] * B_column + xr[i] * C_column # Ax = (Bxr - Cxi) + i(Bxi + Cxr)
    
    
    return zr, zi
