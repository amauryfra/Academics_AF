"""Scientific Computation Project 3, part 1
CID : 01258326
"""
import numpy as np
import matplotlib.pyplot as plt

def Tdisplay(lon,lat,T,levels=50):
    """
    Make contour plot with of single Temperature field given temperature matrix, T,
    and longitude and latitude arrays, lon and lat. The number of contour lines
    is set by levels.
    """
    plt.figure()
    plt.contour(lon,lat,T,levels)
    plt.colorbar()
    plt.axis('equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.grid()
    return None


def applyPCA(fname='data1.npz',display=False):
    """
    Part 1, question 1
    Input:
        fname: datafile
    Output:
        Atf: transformed data, two-dimensional numpy array with 365 columns
        eigenvectors : matrix composed of column eigenvectors of the covariance matrix
        eigenvalues : vector of corresponding eigenvalues
    """
    
    import numpy as np
    from scipy.linalg import eigh # For eigenvector and eigenvalue computation
    
    #---load data---#
    data = np.load(fname)
    T,lat,lon = data['T'],data['lat'],data['lon'] #temperature, latitude, longitude
    if display: Tdisplay(lon,lat,T[0,:,:])

    # Flattening and adapting the data 
    iMax, jMax, kMax = T.shape
    A = np.zeros((iMax, jMax * kMax)) # Initializing our adapted data matrix
    for i in range(iMax) :
        A[i,:] = T[i].flatten() # Stacking latitude and longitude on the same row
        
    # Standardizing
    meanRow = np.mean(A, axis = 0) # Mean of each row 
    standardDev = np.std(A, axis = 0) # Standard deviation of each row
    A = (A - meanRow) / standardDev

    # Performing PCA
    Cov = 1.0 / (iMax - 1) * np.dot(A.T, A) # Covariance matrix | Shape is 10224 x 10224
    # Eigenvector and eigenvalue computation for symmetric matrix
    # Computing all eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(Cov) 
    # Eigenvalues are sorted from low to high
    # Sorting from high to low 
    eigenvalues = np.flip(eigenvalues) # Shape is 10224
    eigenvectors = np.flip(eigenvectors, axis = 1) # Shape is 10224 x 10224
    # Transformation
    Atf = np.dot(A,eigenvectors)
    
    return Atf.T, eigenvectors, eigenvalues # Asked for 365 columns


def analyzePCA(fname='data1.npz'):
    """
    Part1, question 1: A
    Input:
        fname: datafile
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Applying PCA
    Atf, eigenvectors, eigenvalues = applyPCA(fname = fname)
    Atf = Atf.T # Working with days in rows
    
    # Importance of each components
    components = np.arange(1,Atf.shape[1] + 1)
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('PCA eigenvalues for temperature data')
    plt.xlabel('Eigenvalue number j, in log scale')
    plt.ylabel('$\lambda_j$', rotation = 0)
    #plt.yscale('log')
    plt.xscale('log')
    plt.scatter(x = components, y = eigenvalues, s = 50)
    
    ## Explained variance
    sumVarExplained = 0 # Sum of all variance explained
    cumulVarExplained = np.zeros(eigenvalues.shape[0]) # Cumulative explained variance
    totalVar = eigenvalues.sum() # Total variance
    variancesExplained = np.zeros(eigenvalues.shape[0])
    for ind, lbda in enumerate(eigenvalues) :
        varExplained = lbda / totalVar
        variancesExplained[ind] = varExplained
        sumVarExplained += varExplained
        cumulVarExplained[ind] = sumVarExplained
    # Plots
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'PCA for temperature data | Variance explained and cumulative' \
        ' variance explained for increasing number of components'
    plt.title(title)
    plt.xlabel('Component number j, in log scale')
    plt.xscale('log')
    plt.ylabel('$\lambda_j/V$ and $\sum_{k=1}^j \lambda_k/V$', rotation = 0, labelpad = 25)
    plt.scatter(components, variancesExplained, label = 'Variance explained')
    plt.scatter(components, cumulVarExplained, label = 'Cumulative variance explained')
    plt.plot(0.8 * np.ones(Atf.shape[1]), label = '80%')
    plt.legend(loc = 'best')
    
    
    ## Analysis
    # Separate top 2 components
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Variations over time of the top 2 components corresponding values')
    plt.xlabel('Days')
    plt.ylabel('Variation of temperature compared to the annual average')
    plt.plot(Atf[:,0], label = 'Top 1 component')
    plt.plot(Atf[:,1], label = 'Top 2 component')
    plt.legend(loc = 'best')
    # Projecting on the 2 top components subspace
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Temperature data visualized on the top 2 components spanned subspace')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.scatter(Atf[:,0], Atf[:,1], s = 200)
    for i in range(365) :   
            plt.annotate(str(i + 1), (Atf[i,0], Atf[i,1]), \
                    xytext = (Atf[i,0] - 1.25, Atf[i,1] - 0.25), fontsize = 4)
    # Projecting on the 3 top components subspace
    fig = plt.figure(figsize=(16, 12), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Temperature data visualized on the top 3 components spanned subspace')
    ax.set_xlabel('Top 1 component')
    ax.set_ylabel('Top 2 component')
    ax.set_zlabel('Top 3 component')
    ax.scatter3D(xs = Atf[:,0], ys = Atf[:,1], zs = Atf[:,2])
    # Separate top 12 components
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Variations over time of the top 12 components corresponding values')
    plt.xlabel('Days')
    plt.ylabel('Variation of temperature compared to the annual average')
    plt.plot(Atf[:,:12])
    # Reduced data
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Reduced data : variations over time of all components corresponding values')
    plt.xlabel('Days')
    plt.ylabel('Reduced data : variation of temperature compared to the annual average')
    Areduced = np.outer(eigenvectors.T[:,0],Atf.T[0,:])
    plt.plot(Areduced.T)

    return None




def rec1(R,p,l=0.0,niter=50):
    """
    Part 1, question 2(a): Estimate missing data stored in input
    array, R. Efficient and complete version of repair1.
    Input:
        R: 2-D data array (should be loaded from data2.npz)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    def cost(A,B,iK,jK,l):
        Ra = A.dot(B)
        dR = np.sum((Ra[iK,jK]-R[iK,jK])**2)
        C = dR + l*(np.sum(A**2)+np.sum(B**2))
        return C

    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    mu = np.mean(R0[iK,jK])+0.1*np.max(R0[iK,jK])
    A = np.ones((a,p))*np.sqrt(mu/p)
    B = np.ones((p,b))*np.sqrt(mu/p)


    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        #print("iteration z=",z)
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = np.sum(B[n,mlist[m]]**2)
                    Asum = 0
                    for j in mlist[m]:
                        Rsum = A[m,:].dot(B[:,j]) - A[m,n]*B[n,j]
                        Asum += (R[m,j] - Rsum)*B[n,j]
                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                    #Add code here to update B[m,n]
                    if m in nlist[n]:
                        Afac = np.sum(A[nlist[n],m]**2)
                        Bsum = 0
                        for i in nlist[n]:
                            Rsum = A[i,:].dot(B[:,n]) - A[i,m]*B[m,n]
                            Bsum += (R[i,n]-Rsum)*A[i,m]
                        B[m,n] = Bsum/(Afac+l) #New B[m,n]

        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        #if z%1==0: print("dA,dB=",dA[z],dB[z])
        #if z%10==0:
            #C = cost(A,B,iK,jK,l)
            #print("cost,C=",C)
    return A,B


def corruptData(T0, propMissing = 0.005) :
    """
    This function randomly corrupts a given proportion of entries in T0.

    Parameters
    ----------
    T0 : an (AxB)-dimensional numpy array - the ground truth matrix with no corrupted data 
    propMissing : a float number - the proportion of data to corrupt

    Returns
    -------
    T : an (AxB)-dimensional numpy array - the matrix with corrupted data
    corruptPairsIndex : a k-dimensionnal numpy array - contains the index of the corrupted 
        data

    """
    
    import copy
    import numpy as np
    T = copy.deepcopy(T0) # Working with T which will be modfied 
    
    # Processing the index
    aLen, bLen = T.shape # T of shape A x B 
    iIndex = np.arange(aLen)
    jIndex = np.arange(bLen)
    allPairsIndex = np.array([[i, j] for i in iIndex for j in jIndex])
    ab = allPairsIndex.shape[0]
    
    # Corrupting the data
    # Randomly choosing the index of the data to corrupt
    corruptIndex = np.random.choice(a = np.arange(ab), 
                                    size = int(propMissing * ab), 
                                    replace = False)
    corruptPairsIndex = allPairsIndex[corruptIndex]
    T[corruptPairsIndex[:,0],corruptPairsIndex[:,1]] = -1000
    
    return T, corruptPairsIndex
    


def estimationError(R, p, l, corruptPairsIndex, T0) :
    """
    This function computes a metric measuring the error made while estimating 
    missing/corrupted data by the use of rec1 process. It returns a nonnegative 
    float number, 0 corresponding to the perfect estimation rec1 can make.

    Parameters
    ----------
    R : an (AxB)-dimensional numpy array - the matrix with corrupted data set to -1000
    p : an integer - the dimension parameter
    l : a float number - l2-regularization parameter
    corruptPairsIndex : a k-dimensionnal numpy array - contains the index of the corrupted 
        data
    T0 : an (AxB)-dimensional numpy array - the ground truth matrix with no corrupted data 

    Returns
    -------
    errorsMean : a float number - the error metric

    """
    import numpy as np
    
    A, B = rec1(R = R, p = p, l = l) # Computing A and B
    T1 = np.dot(A,B) # Computing estimated matrix
    
    # Ground truth values
    groundTruth = T0[corruptPairsIndex[:,0],corruptPairsIndex[:,1]] 
    # Estimated values
    estimations = T1[corruptPairsIndex[:,0],corruptPairsIndex[:,1]] 

    # Relative errors 
    errors = np.abs((groundTruth - estimations) / groundTruth)
    
    return errors.mean()


def analyzeRec1(fname = 'data2.npz'):
    """
    Part1, question 2(a): 
    Input:
        fname: datafile
    """
    import numpy as np
    from scipy import interpolate
    
    # Loading the data
    data = np.load(fname)
    T0 = data['T'] # Ground truth original T
    
    # Rank of original matrix
    rankT0 = np.linalg.matrix_rank(T0)
    print('Rank of full data matrix T : ', rankT0)
    print('')
    
    # Test of different proportion of missing data
    T, corruptPairsIndex = corruptData(T0 = T0, propMissing = 0.05)
    error05 = estimationError(R = T, p = 6, l = 0, \
                                corruptPairsIndex = corruptPairsIndex, T0 = T0)
    print('Error for 5% of missing data : ', error05)
    print('')
    T, corruptPairsIndex = corruptData(T0 = T0, propMissing = 0.75)
    error75 = estimationError(R = T, p = 6, l = 0, \
                                corruptPairsIndex = corruptPairsIndex, T0 = T0)
    print('Error for 75% of missing data : ', error75)
    print('')
    
    
    # Setting corrupted the data to 0.5% for the rest of the analysis
    T, corruptPairsIndex = corruptData(T0 = T0, propMissing = 0.005)
    
    ## Varying lambda - p fixed to 6 ##
    lambdas = np.linspace(start = 0, stop = 85, num = 20)
    errorsLbda = np.zeros(lambdas.shape[0])
    for ind, lbda in enumerate(lambdas) :
        print(ind)
        # Computing error
        errorsLbda[ind] = estimationError(R = T, p = 6, l = lbda, \
                                corruptPairsIndex = corruptPairsIndex, T0 = T0)
    # Plot
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Errors made on missing data estimations for different $\lambda$ parameters' 
          ' | p = 6')
    plt.xlabel('$\lambda$')
    plt.ylabel('Error')
    plt.plot(lambdas, errorsLbda)
    
    
    ## Varying p - lambda fixed to 0 ##
    pList = np.arange(start = 1, stop = 102, step = 2)
    errorsP = np.zeros(pList.shape[0])
    for ind, p in enumerate(pList) :
        print(ind)
        errorsP[ind] = estimationError(R = T, p = p, l = 0., \
                                corruptPairsIndex = corruptPairsIndex, T0 = T0)
        print(errorsP[ind])
    # Plot
    averageP = interpolate.UnivariateSpline(pList,errorsP) # Average trend function
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title('Errors made on missing data estimations for different p parameters' 
          ' | $\lambda$ = 0')
    plt.xlabel('p')
    plt.ylabel('Error')
    plt.plot(pList[:47], errorsP[:47], label = 'Computations') # Cutting erratic last values
    plt.plot(pList[:47], averageP(pList[:47]), label = 'Average trend')
    plt.legend(loc = 'best')

    return errorsLbda


def rec2(R,p,l=0.0,eta=0.0,niter=50):
    """
    Part 1, question 2(b): Estimate data stored in input
    array, R with modified method.
    Input:
        R: 2-D data array
        p: dimension parameter
        l: l2-regularization parameter for A and B
        eta: l2-regularization parameter for beta
        niter: maximum number of iterations during optimization
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
        beta: a-element numpy array set during optimization
    """
    def cost(A,B,iK,jK,l,beta,eta):
        Ra = A.dot(B)
        a,b = Ra.shape
        betaM = beta*np.ones((b,1))
        betaM = betaM.T
        dR = np.sum((Ra[iK,jK]+betaM[iK,jK]-R[iK,jK])**2)
        C = dR + l*(np.sum(A**2)+np.sum(B**2))+eta*np.sum(beta**2)
        return C

    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data
    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B,beta
    mu = np.mean(R0[iK,jK])+0.1*np.max(R0[iK,jK])
    A = np.ones((a,p))*np.sqrt(mu/p)
    B = np.ones((p,b))*np.sqrt(mu/p)
    beta = np.zeros(a)
    Nj = np.ones(a)*b
    for i in range(a):
        x = R0[i,:]
        x = x[x!=-1000]
        beta[i] = x.mean()
        Nj[i] = len(x)

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        print("iteration z=",z)
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        #Code below should be modified to account for beta
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = np.sum(B[n,mlist[m]]**2)
                    Asum = 0
                    for j in mlist[m]:
                        Rsum = A[m,:].dot(B[:,j]) - A[m,n]*B[n,j]
                        Asum += (R[m,j]- beta[m] - Rsum)*B[n,j] ### Change here ###
                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                    #Update B[m,n]
                    if m in nlist[n]:
                        Afac = np.sum(A[nlist[n],m]**2)
                        Bsum = 0
                        for i in nlist[n]:
                            Rsum = A[i,:].dot(B[:,n]) - A[i,m]*B[m,n]
                            Bsum += (R[i,n] - beta[i] - Rsum)*A[i,m] ### Change here ###
                        B[m,n] = Bsum/(Afac+l) #New B[m,n]

        #Add code here to set beta to minimize the cost given the updated A and B matrices
        jKUnique = np.unique(jK) ### Change here ###
        ### Change here ###
        beta = np.sum(R0[:,jKUnique] - np.dot(A,B[:,jKUnique]), axis = 1)\
            / (jKUnique.shape[0] + eta) 

        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%1==0: print("dA,dB=",dA[z],dB[z])
        if z%10==0:
            C = cost(A,B,iK,jK,l,beta,eta)
            print("C=",C)
    return A,B,beta



if __name__=='__main__':
    #load data file for part 1, question 2
    fname = 'data2.npz'
    data = np.load(fname)
    T,lat,lon = data['T'],data['lat'],data['lon']
