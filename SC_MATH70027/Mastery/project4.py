"""Scientific Computation Project 4
CID : 01258326
"""

import numpy as np
import matplotlib.pyplot as plt  
import networkx as nx
import scipy.spatial
import copy


def WTdist(M,Dh, metric = 'euclidean'):
    """
    Question 1:
    Compute WT distance matrix, X, given probability matrix, M, and Dh = D^{-1/2}
    """
    DM = np.dot(Dh,M.T) # Each column vector of DM is Dh M.T[i]

    # This useful scipy method computes distance between each
    # pair of row vectors of DM.T -> column vectors of DM
    X = scipy.spatial.distance.cdist(DM.T,DM.T, metric = metric)
        
    return X

def analyzeWTdist(t = 6) :
    
    ### Taken from main() ###
    ### Generating the working graph for tests ###
    A = np.load('data4.npy')
    G = nx.from_numpy_array(A)
    N = G.number_of_nodes()
    k = np.array(list(nx.degree(G)))[:,1]
    Dinv = np.diag(1/k)
    Dh = np.diag(np.sqrt(1/k))
    P = Dinv.dot(A) 
    M = np.linalg.matrix_power(P,t)
    #####################################
    
    # Computing WT distance
    X = WTdist(M,Dh) 
    nodeZeroWTdist = X[0,:]
    # Computing 'conventionnal distance'
    lengthsDict = nx.shortest_path_length(G, source = 0)
    nodeZeroConvDist = np.array([lengthsDict[i] for i in range(N)])
    
    # Plot
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'Comparison of WT distance and conventional distance, '\
    'from node zero to all nodes '\
    'on provided graph' 
    plt.title(title)
    plt.xlabel('Node $j$')
    #plt.yscale('log')
    plt.ylabel('Distances $d_{ij}$', rotation = 0, labelpad = 25)
    plt.plot(nodeZeroConvDist, label = 'Conventional distance - with scaling factor')
    plt.plot(10 * nodeZeroWTdist, label = 'WT distance')
    plt.legend(loc = 'best')
    
    return None
    

def WTdist2(M,Dh, Clist):
    """
    Question 2:
    Compute squared distance matrix, Y, given probability matrix, M, Dh = D^{-1/2},
    and community list, Clist
    """
    
    # Matrix such that R_aj is the probability fo to go from community a to node j 
    R = np.array([M[Clist[i],:].mean(axis = 0) for i in range(len(Clist))])
    
    # The squared distance between two communities a and b
    Y = WTdist(R,Dh, metric = 'sqeuclidean') # Metric is square euclidean
    
    return Y

def makeCdict(G,Clist):
    """
    For each distinct pair of communities a,b determine if there is at least one link
    between the communities. If there is, then b in Cdict[a] = a in Cdict[b] = True
    """
    m = len(Clist)
    Cdict = {}
    for a in range(m-1):
        for b in range(a+1,m):
            if len(list(nx.edge_boundary(G,Clist[a],Clist[b])))>0:
                if a in Cdict:
                    Cdict[a].add(b)
                else:
                    Cdict[a] = {b}

                if b in Cdict:
                    Cdict[b].add(a)
                else:
                    Cdict[b] = {a}
    return Cdict


def makeCmat(G,Clist) :
    """
    This function creates a pseudo-adjacency matrix C which is such that if the 
    communities a and b are linked then C_ab = 1. We have C_ab = 0 otherwise. 

    Parameters
    ----------
    G : a networkx graph object - the considered graph
    Clist : a list of lists - the communities

    Returns
    -------
    Cmat : an mxm-dimensional numpy array - the pseudo-adjacency matrix

    """
    
    m = len(Clist) # Number of communities
    Cmat = np.zeros((m,m)) # Initializing matrix giving linked pairs
    
    # Considering all distinct pairs 
    for a in range(m - 1) :
    #for a in range(m) :
        for b in range(a + 1, m) :
        #for b in range(m) :
            if len(list(nx.edge_boundary(G,Clist[a],Clist[b]))) > 0 :
                Cmat[a,b] = 1
                
    return Cmat + Cmat.T # The matrix is symmetric

def merge(a, b, Clist) :
    """
    This function merges community a and b and modifies Clist in-place accordingly. 

    Parameters
    ----------
    a : an integer - the first community's index
    b : an integer - the second community's index
    Clist : a list of lists - the communities

    Returns
    -------
    None
    
    """
    Clist[a] += Clist[b]
    Clist.pop(b)

def main(t=6,Nc=2):
    """
    WT community detection method
    t: number of random walk steps
    Nc: number of communities at which to terminate the method
    """
    
    # Read in graph
    A = np.load('data4.npy')
    G = nx.from_numpy_array(A)
    N = G.number_of_nodes()

    # Construct array of node degrees and degree matrices
    k = np.array(list(nx.degree(G)))[:,1]
    Dinv = np.diag(1/k)
    Dh = np.diag(np.sqrt(1/k))

    P = Dinv.dot(A) # Transition matrix
    M = np.linalg.matrix_power(P,t) # P^t

    # Initialize community list
    Clist = []
    for i in range(N):
        Clist.append([i])
    
    m = len(Clist)  # Number of communities initialized
    assert Nc < m
    
    Y = WTdist2(M,Dh,Clist) # Matrix storing the r_ab^2
    S = (1 / (2 * N)) * Y # Matrix storing the costs initialized
    Cmat = makeCmat(G,Clist) # Pseudo adjacency matrix
    L = np.ones(m) # Keeping track of cardinals of communities 
    
    while m > Nc :

        # Indices of linked communities
        ones_index = np.argwhere(Cmat > 0)
        # Linked communities with the lowest cost
        amin, bmin = ones_index[S[ones_index[:,0],ones_index[:,1]].argmin()]
        

        # Updating S
        S[amin,:] = (L[amin] + L) * S[amin,:] + (L[bmin] + L) * S[bmin,:] - \
            S[amin,bmin] * L
        S[amin,:] /= (L + L[amin] + L[bmin])
        S[:,amin] = 1.0 * S[amin,:] 
        S = np.delete(S, bmin, axis = 0)
        S = np.delete(S, bmin, axis = 1)
        
        
        # Update Cmat
        Cmat[amin,:] += Cmat[bmin,:]
        Cmat[amin,amin] = 0 # No ones on the diagonal
        Cmat[Cmat > 1] = 1
        Cmat[:,amin] = copy.deepcopy(Cmat[amin,:])
        Cmat = np.delete(Cmat, bmin, axis = 0)
        Cmat = np.delete(Cmat, bmin, axis = 1)
        
        # Updating L
        L[amin] += L[bmin]
        L = np.delete(L, bmin)
        
        # Merging
        merge(amin, bmin, Clist)

        m = len(Clist) # Updating m
        
    return Clist


def analyzeFD():
    """
    Question 4:
    """
    #%timeit out = main(t = 6, Nc = 2)
    
    # Importing data
    A = np.load('data4.npy')
    G = nx.from_numpy_array(A)
    N = G.number_of_nodes()
    
    #### t = 6 and Nc = 2 ###
    t = 6
    Nc = 2
    out = main(t, Nc = Nc)
    WTlabels = np.zeros(N) # Labels computed through WT clustering 
    trueLabelsArr = np.load('label4.npy') # True labels
    trueLabels = {}
    # Adapting data
    for i in range(N) :
        if trueLabelsArr[i] > 0 :
            lbl = '+' # Setting true labels
        else :
            lbl = '-'
        trueLabels[i] = lbl
        for j in range(Nc) :
            if i in out[j] :
                WTlabels[i] = j # Setting WT labels
    # Plots
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'Clustering of the functional network of the human brain '\
        'using the Walktrap algorithm | $t$ = 6 and $N_c$ = 2'
    plt.title(title)
    nx.draw(G, node_color = WTlabels, labels = trueLabels, \
            with_labels = True, font_color = 'white')
    ### ###
    
    #### t = 8 and Nc = 2 ###
    t = 8
    Nc = 2
    out = main(t, Nc = Nc)
    WTlabels = np.zeros(N) # Labels computed through WT clustering 
    trueLabelsArr = np.load('label4.npy') # True labels
    trueLabels = {}
    # Adapting data
    for i in range(N) :
        if trueLabelsArr[i] > 0 :
            lbl = '+' # Setting true labels
        else :
            lbl = '-'
        trueLabels[i] = lbl
        for j in range(Nc) :
            if i in out[j] :
                WTlabels[i] = j # Setting WT labels
    # Plots
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'Clustering of the functional network of the human brain '\
        'using the Walktrap algorithm | $t$ = 8 and $N_c$ = 2'
    plt.title(title)
    nx.draw(G, node_color = WTlabels, labels = trueLabels, \
            with_labels = True, font_color = 'white')
    ### ###
    
    #### t = 10000 and Nc = 2 ###
    t = 10000
    Nc = 2
    out = main(t, Nc = Nc)
    WTlabels = np.zeros(N) # Labels computed through WT clustering 
    trueLabelsArr = np.load('label4.npy') # True labels
    trueLabels = {}
    # Adapting data
    for i in range(N) :
        if trueLabelsArr[i] > 0 :
            lbl = '+' # Setting true labels
        else :
            lbl = '-'
        trueLabels[i] = lbl
        for j in range(Nc) :
            if i in out[j] :
                WTlabels[i] = j # Setting WT labels
    # Plots
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'Clustering of the functional network of the human brain '\
        'using the Walktrap algorithm | $t$ = 10000 and $N_c$ = 2'
    plt.title(title)
    nx.draw(G, node_color = WTlabels, labels = trueLabels, \
            with_labels = True, font_color = 'white')
    ### ###
    
    ### t = 6 and Nc = 6 ###
    t = 6
    Nc = 6
    out = main(t, Nc = Nc)
    WTlabels = np.zeros(N) # Labels computed through WT clustering 
    # Adapting data
    for i in range(N) :
        for j in range(Nc) :
            if i in out[j] :
                WTlabels[i] = j # Setting WT labels
    # Plots
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'Clustering of the functional network of the human brain '\
        'using the Walktrap algorithm | $t$ = 6 and $N_c$ = 6'
    plt.title(title)
    nx.draw(G, node_color = WTlabels, labels = trueLabels, \
            with_labels = True, font_color = 'white')  
    ### ###
    
    ### Scoring ###
    t = 6
    for Nc in range(2,467) : # Scanning Nc
        out = main(t, Nc = Nc) # Computing WT communities
        score = 0
        for com in range(Nc) :
            # Computing score
            nodes = out[com] 
            propPlus = np.count_nonzero(trueLabelsArr[nodes] > 0) / trueLabelsArr[nodes].shape[0]
            propMins = 1 - propPlus
            propMax = np.max([propPlus,propMins])
            score += propMax
        score /= Nc
        if score == 1.0 :
            break
    print('Smallest Nc for which the score is 1.0 : ', Nc)
    ### ###
    
    ### t = 6 and Nc = 264 ###
    t = 6 
    Nc = 264 
    out = main(t, Nc = Nc)
    WTlabels = np.zeros(N) # Labels computed through WT clustering 
    # Adapting data
    for i in range(N) :
        for j in range(Nc) :
            if i in out[j] :
                WTlabels[i] = j # Setting WT labels
    # Plot
    plt.figure(figsize=(16, 12), dpi=80)
    title = 'Clustering of the functional network of the human brain '\
        'using the Walktrap algorithm | $t$ = 6 and $N_c$ = 264'
    plt.title(title)
    nx.draw(G, node_color = WTlabels, labels = trueLabels, \
            with_labels = True, font_color = 'white')
    ### ###

    return None 

if __name__=='__main__':
    analyzeWTdist(t = 6) 
    analyzeFD()





    
