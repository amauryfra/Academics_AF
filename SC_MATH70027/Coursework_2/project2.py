"""Scientific Computation Project 2
Amaury Francou
CID : 01258326
"""

def gSearch(G,x):
    """
    Code for Part 1, question 1
    Input:
        G: A weighted NetworkX graph
        x: an integer
    Output:
        Fdict: A dictionary
    """
    import networkx as nx
    import heapq
    import time # Importing time to assess wall times
    import numpy as np 
    
    wallTimesBH = [] # Initializing wall time list for binary heap min removal 
        # operation
    t1Full = time.time() # Timing full process
    
    
    Fdict = {}
    Mdict = {}
    Mlist = []

    heapq.heappush(Mlist,[0,x])
    Mdict[x]=Mlist[0]

    while len(Mlist)>0:
        
        t1BH = time.time()
        dpop,npop = heapq.heappop(Mlist)
        if npop != -1000:
            del Mdict[npop]
            t2BH = time.time()
            wallTimesBH += [t2BH-t1BH] # Adding current wall time observed for finding
            # the highest priority node and removing it from the queue
            
        
            Fdict[npop] = dpop
            for a,b,c in G.edges(npop,data='weight'):
                if b in Fdict:
                    pass
                elif b in Mdict:
                    dcomp = dpop + c
                    if dcomp<Mdict[b][0]:
                        L = Mdict.pop(b)
                        L[1] = -1000
                        Lnew = [dcomp,b]
                        heapq.heappush(Mlist,Lnew)
                        Mdict[Lnew[1]]=Lnew
                else:
                    dcomp = dpop + c
                    L = [dcomp,b]
                    heapq.heappush(Mlist,L)
                    Mdict[L[1]]=L
                    
    t2Full = time.time()
    wallTimesBH.pop() # The last value is always 0  
    # Returning the mean of the assessed wall times and full observed wall time
    return Fdict, np.mean(wallTimesBH), t2Full - t1Full 


def test_queue(Nmax = 10000, numberTests = 1):
    """
    This functions plots the recorded wall times of the binary heap minimum removal 
    operation and the full Dijkstra algorithm with binary heap implementation, made
    on random graphs, for an increasing number of nodes N.

    Parameters
    ----------
    Nmax : an integer greater than 55 - the maximum number of nodes 
    numberTests : an integer greater than 1 - number of random graphs
        generated for each N 

    """
    # Imports
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    wallTimesBH = np.zeros(Nmax-55) # Initializing wall times array
    wallTimesFull = np.zeros(Nmax-55)

    
    logn = np.zeros(Nmax-55) # Initializing reference trend arrays
    nlogn = np.zeros(Nmax-55)
    
    for ind, i in enumerate(range(55,Nmax)) : # Iterating over the number of nodes
    
        # Filling reference trend arrays
        logn[ind] = np.log(i) 
        nlogn[ind] = i * np.log(i)
        
        L = 3 * i # Nodes having in average 3 neighbors 
        
        for _ in range(numberTests) : # Running several tests to capture the typical behavior
            
            # Generating a random weighted graph
            G = nx.gnm_random_graph(i,L)
            for (u,v,w) in G.edges(data=True):
                w['weight'] = np.random.randint(10) + 1 # Weights between 1 and 10
                
            # Choosing a random starting node
            xInd = np.random.randint(len(G.edges))
            x = list(G.edges)[xInd][0]
            
            # Retreiving wall time observed for finding
            # the highest priority node and removing it from the queue
            _ ,  wallTimeBH, wallTimeFull = gSearch(G,x)
            
            # Adding wall time to array
            wallTimesBH[ind] += wallTimeBH
            wallTimesFull[ind] += wallTimeFull
        
        # Taking average of tests
        wallTimesBH[ind] = wallTimesBH[ind] / numberTests
        wallTimesFull[ind] = wallTimesFull[ind] / numberTests
    
    # Plots
    plt.figure()
    plt.title('Wall time observed for minimum removal operation in binary heap',\
              fontsize = 20)
    plt.xlabel('Number of nodes $N$', fontsize = 20)
    plt.ylabel('Wall time', fontsize = 20)
    plt.plot(wallTimesBH, label = 'Wall time')
    # A scaling factor is used and requires fine tuning
    plt.plot((0.98 * wallTimesBH[0] / logn[0]) * logn, label = 'log Trend') # General trend 
    plt.legend(loc = 'best', fontsize = 20)
    
    plt.figure()
    plt.title('Wall time observed for full Dijkstra algorithm with binary heap', \
              fontsize = 20)
    plt.xlabel('Number of nodes $N$', fontsize = 20)
    plt.ylabel('Wall time', fontsize = 20)
    plt.plot(wallTimesFull, label = 'Wall time')
    # A scaling factor is used and requires fine tuning
    plt.plot((wallTimesFull[0] / (1.85 * nlogn[0])) * nlogn, \
             label = 'nlogn Trend') # General trend
    plt.legend(loc = 'best', fontsize = 20)

    return None


    
def getPath(parent, n1, n2) :
    """
    This function computes a path in a graph given a 'parent' list, in which
    the i-th element is the node preceding node i in the path. 

    Parameters
    ----------
    parent : a list of integers - the i-th element is the node preceding i in the path
    n1 : an integer - the starting node in the path
    n2 : an integer - the end node in the path
    
    Returns
    -------
    path : a list of integers - the path between n1 and n2 in the graph that 
        follows the parent list 

    """
    
    n = n2
    path = [n2]
        
    while n != n1 : # Stopping condition - we arrived at n1 from n2
            
        n = parent[n]
        path += [n]
        
    path.reverse() # The list is flipped 
    
    return path
    
        
def find_pmax(Alist,n1,n2):
    """
    Part 1, Question 2
    Find maximum packet size, pmax, which can travel
    from node n1 to n2 in network (defined by adjacency list, Alist)

    Input:
    Alist: Adjacency list for graph. Alist[i] is a sub-list. Each element of the sublist
    of the form [x,y] where x is the node number for a neighbor of i and y is
    the edge weight for the link connecting nodes i and x.

    n1: graph node at beginning of path
    n2: graph node at end of path

    Output:
    pmax: maximum packet size which can travel between nodes n1 and n2
    R: A list of integers corresponding to a feasible route from n1 to n2 with
    p=pmax

    """
    import heapq
    
    Fdict = {} # Finalized nodes in this dict
    Mdict = {} # Queue dict
    Mlist = [] # Queue list organized in binary heap by heapq
    
    N = len(Alist) # Number of nodes
    parent = [0 for k in range(N)] # List containing the preceding node in the 
        # widest route to the final node corresponding to the index in the list 
    
    # Initializing the binary heap 
    heapq.heappush(Mlist,[1e-7,n1]) # We avoid division by 0
    Mdict[n1] = Mlist[0]
    
    # Dijkstra adapted 
    while len(Mlist)>0 :
    
        wpop, npop = heapq.heappop(Mlist) # Remove node with widest path from n1 to itself 
        # Retreiving said widest path 
        wpop = 1 / wpop # We stored the inverse because heapq pops the min (in log(N) time)
        
        
        if npop != -1000 :
            
            # Removing from queue and finalizing 
            del Mdict[npop] 
            Fdict[npop] = wpop
            
            # We stop once we arrived at target n2
            if npop == n2 :
                break
            
            # Looking for neighbors of node npop
            for neighbor in Alist[npop] :
                n = neighbor[0] # Node of neighbor 
                w = neighbor[1] # Weight between npop and n 
                
                # If already finalized -> do nothing 
                if n in Fdict :
                    pass
                
                # If already in queue -> is the new possible route better ? 
                elif n in Mdict :
                    
                    # Seeking for the biggest minimum weight in the routes to n
                    # Is it the route already seen previously ? -> 1 / Mdict[n][0]
                    # Or is it the new route where npop is the ultimate 
                    # node before arriving at n ? -> min(w,wpop)
                    prev = 1 / Mdict[n][0] # Previous provisional widest path 
                    w_new = max(prev, min(w, wpop))
                    
                    # If this new provisional widest path possible is bigger 
                    # than the previous one -> we update the value in the heap 
                    if w_new > prev : 
                        L = Mdict.pop(n)
                        L[1] = -1000
                        Lnew = [1 / w_new, n]
                        heapq.heappush(Mlist,Lnew)
                        Mdict[Lnew[1]]=Lnew
                    
                        # We also update the 'parent' ultimate node before arriving
                        # at n while using the widest path route 
                        parent[n] = npop
                        
                # If neighbor has never been seen before -> add it to queue
                else :
                    
                    # Compute the value of provisional widest path and add 
                    # node to the queue 
                    w_NN = min(wpop,w) 
                    L = [1 / w_NN, n]
                    heapq.heappush(Mlist,L)
                    Mdict[L[1]]=L
                    
                    # Update the 'parent' ultimate node before arriving at n 
                    parent[n] = npop
            
    
    pmax = int(Fdict[n2]) # The maximum minimum weight in routes from n1 to n2
    R = getPath(parent, n1, n2) # A feasible route from n1 to n2 with pmax
    return pmax, R
            

def solveSDE():
    """
    Part 2, question 1
    Simulate linear SDE M times with Dt=dt
    """
    import numpy as np
    import matplotlib.pyplot as plt

    #set model parameters
    T = 1
    l = 2 #lambda
    mu = 1
    X0 = 1

    #set numerical parameters
    M = 1000
    nt = 2**9
    dt = T/nt
    Nt = nt
    Dt = T/Nt

    fac1 = 1/(1-l*Dt)
    fac2 = -0.5*Dt*mu**2
    t = np.linspace(0,T,Nt+1)

    #initialize arrays
    dW= np.sqrt(dt)*np.random.normal(size=(nt,M))
    X = np.zeros((nt+1,M))
    X[0,:] = X0

    #Iterate over Nt time steps
    for j in range(Nt):
        mw = mu*dW[j,:]
        X[j+1,:] =(1+mw*(1+0.5*mw)+fac2)*fac1*X[j,:]

    return X

def solveSDE_modified(nt) :
    """
    Modified version of solveSDE for testing purpose 
    Part 2, question 1
    Simulate linear SDE M times with Dt=dt
    """
    import numpy as np
    import matplotlib.pyplot as plt

    #set model parameters
    T = 1
    l = 2 #lambda
    mu = 1
    X0 = 1

    #set numerical parameters
    M = 100000
    #nt = 2**9 ### Change made here | parametrizing ###
    
    dt = T/nt 
    Nt = nt 
    Dt = T/Nt 

    fac1 = 1/(1-l*Dt)
    fac2 = -0.5*Dt*mu**2

    #initialize arrays
    dW= np.sqrt(dt)*np.random.normal(size=(nt,M))
    
    X = np.zeros((nt+1,M))
    X[0,:] = X0

    #Iterate over Nt time steps
    for j in range(Nt):
        mw = mu*dW[j,:]
        X[j+1,:] =(1+mw*(1+0.5*mw)+fac2)*fac1*X[j,:]
    
    #### Extracting exact X(T) using computed W(T) ###
    WT = np.sum(dW, axis = 0) # W(T) = W(T) - W(0) = sum(W(j+1 dt) -W(j dt))
    XTexactList =  X0 * np.exp((l - 0.5 * (mu**2)) * T * np.ones(M) + mu * WT)

    return X, XTexactList #### Change in output ###

def testSDE() :
    """
    Part 2, question 1: Analyze results generated by method implemented in solveSDE
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    
    #### Accuracy #### and #### Efficiency ####
    
    # Iterating over nt => and dt
    ntList = np.arange(start = 2**3, stop = 2**8, step = 2**3)
    ntListLen = len(ntList)
    
    weakError = np.zeros(ntListLen) # Initializing error arrays
    strongError = np.zeros(ntListLen)
    
    wallTimes = np.zeros(ntListLen) # Initializing wall time arrays
    
    # Parameters
    X0 = 1
    l = 2
    T = 1
    
    # Mean of X(T)
    meanXT = X0 * np.exp(l * T)
    

    for ind, nt in enumerate(ntList) :
    
        
        for _ in range(5) : # Averaging over several tests 
        
            X, XTexactList = solveSDE_modified(nt)
        
            weakError[ind] += np.abs(np.mean(X[nt, :]) - meanXT)
            strongError[ind] += np.mean(np.abs(X[nt, :] - XTexactList))
            
        weakError[ind] =  weakError[ind] / 5
        strongError[ind] = strongError[ind] / 5
    
        t1 = time.time()
        X, _ = solveSDE_modified(nt)
        t2 = time.time()
        wallTimes[ind] = t2 - t1
        
    
    
    # PLots
    plt.figure()
    plt.title('Weak and strong errors against nt ',  fontsize = 20)
    plt.loglog(ntList, weakError, label = r'$|\bar{X}(T) - \exp(\lambda T)|$')
    plt.loglog(ntList,np.power(0.06 * ntList,-1.1), '-.', label = r'$\Delta t^{1.1}$')
    plt.loglog(ntList,np.power(0.06 * ntList,-1.05), ':', label = r'$\Delta t^{1.05}$')
    plt.loglog(ntList, strongError, label = r'$\overline{|X(T) - X_{exact}(T)|}$')
    plt.xlabel('nt',  fontsize = 20)
    plt.ylabel('errors',  fontsize = 20)
    plt.legend(loc = 'best',  fontsize = 20)
    
    plt.figure()
    plt.title('Walltime observed for solveSDE function against nt', fontsize = 20)
    plt.plot(ntList, wallTimes, label = 'Walltime observed')
    plt.plot(ntList,0.125 * wallTimes[0] * ntList, label = 'Linear trend')
    plt.xlabel('nt')
    plt.ylabel('Walltime')
    plt.legend(loc='best', fontsize = 20)
    
    
    
    
    #####################################################################################
    
    
    
    
    #### Stability ####
    
    # nt = 3
    Xsmall, _  = solveSDE_modified(nt = 3)
    Xsmall = np.mean(Xsmall,axis = 1)
    
    # nt = 1000
    Xbig, _  = solveSDE_modified(nt = 512)
    Xbig = np.mean(Xbig,axis = 1)
    
    meanXt = np.exp(2 * np.linspace(0,1,513)) # Mean at time t
    
    # Plots
    plt.figure()
    plt.title('Solutions X of SDE for several parameter nt', fontsize = 20)
    plt.plot(np.linspace(0,1,4), Xsmall, label = 'Mean of solutions X for nt = 3')
    plt.plot(np.linspace(0,1,513), Xbig, label = 'Mean of solutions X for nt = 512')
    #plt.plot(np.linspace(0,1,513), meanXT * np.ones(np.linspace(0,1,513).shape[0]), \
             #label = r'Expected mean at time T : $\langle X(T) \rangle$')
    plt.plot(np.linspace(0,1,513), meanXt, '-.', label = \
             r'$X_0 \exp(\lambda t)$')
    plt.legend(loc = 'best', fontsize = 20)
    plt.xlabel('t')
    plt.ylabel(r'$\overline{X}$')
    
    # Weak error at t for nt = 3 and nt = 512
    weakErrorSmall = np.zeros(np.linspace(0,1,4).shape[0]) 
    weakErrorBig = np.zeros(np.linspace(0,1,513).shape[0]) 
    for ind, t in enumerate(np.linspace(0,1,4)) :
        weakErrorSmall[ind] = np.abs(Xsmall[ind] - meanXt[ind])
    for ind, t in enumerate(np.linspace(0,1,513)) :
        weakErrorBig[ind] = np.abs(Xbig[ind] - meanXt[ind])

    plt.figure()
    plt.title(r'Weak error of convergence against $t$', \
              fontsize = 20)
    plt.plot(np.linspace(0,1,4),weakErrorSmall, label = 'nt = 3')
    plt.plot(np.linspace(0,1,513),weakErrorBig, label = 'nt = 512')
    plt.xlabel('t')
    plt.ylabel(r'$|\bar{X}(t) - \exp(\lambda t)|$')
    plt.legend(loc = 'best', fontsize = 20)
    

    return None #modify as needed




def model1(tf=20,Nt=1000,gamma=1.0):
    """
    Part 2, question 2(a)
    Simulate n-species ecosystem model

    Input:

    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    gamma: model parameter

    Output:
    tarray: size Nt+1 array
    xarray: Nt+1 x n array containing x for the n species at
            each time step including the initial condition.
    """
    import numpy as np

    tarray = np.linspace(0,tf,Nt+1)
    
    C = np.load('C.npy') #this code and the data files should be in the same folder
    alpha = np.load('alpha.npy')
    n = alpha.shape[0]
    xarray = np.zeros((Nt+1,n))
    
    # Step size 
    h = tf / Nt
    
    # Initial condition
    xarray[0,:] = 1e-6
    
    for i in range(Nt) :
        # Euler method 
        xarray[i+1,:] = xarray[i,:] +  \
            h * (xarray[i,:] * (alpha + gamma * np.dot(C,xarray[i,:])))

    return tarray,xarray


def analyze1():
    """
    Part 2, question 2(b)
    Analyze results generated by model1
    """
    #import needed modules here
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    ### Wall time ###
    NtList = np.arange(start = 1000, stop = 3000, step = 1000)
    NtListLen = len(NtList)
    walltimes = np.zeros(NtListLen)
    
    for ind, Nt in enumerate(NtList) :
        
        t1 = time.time()
        _ , _ = model1(tf=20,Nt=Nt,gamma=1.0)
        t2 = time.time()
        walltimes[ind] = t2 - t1
        
    # Plot
    plt.figure()
    plt.title('Walltime observed for model1 against the number of steps Nt')
    plt.plot(NtList,walltimes)
    plt.xlabel('Nt')
    plt.ylabel('Walltime')
    
    
    ### Species evolution ###
    tarray,xarray = model1()
    
    # Plot
    plt.figure()
    plt.title('Evolution of species $i=10$ over time')
    plt.plot(tarray,xarray[:,9])
    plt.xlabel('$t$')
    plt.ylabel('$x_{10}$')
    
    plt.figure()
    plt.title('Evolution of species $i=16$ over time')
    plt.plot(tarray,xarray[:,15])
    plt.xlabel('$t$')
    plt.ylabel('$x_{16}$')
    
    plt.figure()
    plt.title('Evolution of species $i=1$ over time')
    plt.plot(tarray,xarray[:,0])
    plt.xlabel('$t$')
    plt.ylabel('$x_{10}$')
    
    plt.figure()
    plt.title('Evolution of species over time')
    plt.plot(tarray,xarray)
    plt.xlabel('$t$')
    plt.ylabel('$x_{i}$')
    
    
    ### Gamma influence ###
    tarray,xarray = model1(gamma=100000)
    plt.figure()
    plt.title('Evolution of species over time | $\gamma$ = 100000')
    plt.plot(tarray,xarray)
    plt.xlabel('$t$')
    plt.ylabel('$x_{i}$')
        

    return None #modify as needed

