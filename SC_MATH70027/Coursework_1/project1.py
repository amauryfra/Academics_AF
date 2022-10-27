""" Your college id here: 01258326
    Template code for project 1, contains 4 functions:
    code1: complete function for question 1
    test_code1: function to be completed for question 1
    createTable: complete function for question 2
    findAA: function to be completed for question 2
"""

def code1(L,x,istart=0,iend=-1000,N0=2):
    """function for question 1
    Input:
    L: an N-element list of integers arranged in non-decreasing order
    x,istart,iend: integers
    N0: a non-negative integer

    Output: an integer
    """

    if iend==-1000: iend = len(L)-1

    if istart>iend: return -1000

    if iend-istart<N0:
        for ind in range(istart,iend+1):
            if L[ind]==x: return ind
        return -1000
    else:
        imid = int(0.5*(istart+iend))
        if x==L[imid]:
            return imid
        elif x < L[imid]:
            iend = imid-1
            return code1(L,x,istart,iend,N0=N0)
        else:
            istart = imid+1
            return code1(L,x,istart,iend,N0=N0)
        
def denoise(result) :
    """
    This function uses the linear least squares regularization technique seen in 
    Optimization module (week 4) to denoise the walltimes that have been computed 
    on random lists. 

    Parameters
    ----------
    result : a p-dimensionnal numpy array containing the computed walltimes

    Returns
    -------
    result_denoised : a p-dimensionnal numpy array containing the denoised walltimes

    """
    import numpy as np
    
    p = len(result) # Retreiving size of data to denoise
    L = np.column_stack((np.identity(p-1) + np.diag([-1 for i in range(p-2)], k=1) , \
                         np.zeros(p-1))) # Building matrix involved in the penalization term
    L[p-2,p-1] = -1
    I = np.identity(p)
    lbda = 200 # Coefficient of penalization term 
        
    result_denoised = np.dot(np.linalg.inv(I + lbda * np.dot(L.T,L)), result)
    # Computing (I + lbda L.T L)^-1 b, where b is the noisy result vector 
        
    return(result_denoised)


def test_code1(inputs=0.5):
    """Question 1: investigate trends in wall time of code1 function.
    Use the variables inputs and outputs if/as needed.
    You may import modules within this function as needed, please do not import
    modules elsewhere without permission.
    """
    import time
    import numpy as np
    from numpy import random
    random.seed(3141592653)
    import matplotlib.pyplot as plt
    
    results = {'binary' : [], 'linear' : [], 'hybrid' : []} # Initializing dictionnary \
        # which will contain our computation times
    
    for N in range(3,1000) :
    
        N0 = int(inputs * N) # N0 as a fraction of N
        
        for test in range(25) : # Averaging over several random test lists to capture \
            # typical behavior 
            
            hybrid = []
            linear = []
            binary = []
    
            L = np.sort(random.randint(1000,size=N)) # Generating a sorted list of N random \
            # integers smaller than 1000
            x = random.randint(1000) # Generating a random target integer smaller than 1000
        
            t1 = time.time() # Start time 
            _ = code1(L,x,istart=0,iend=-1000,N0=N0) # Computing the search for hybrid version 
            t2 = time.time() # End time
            hybrid.append(t2-t1) # Adding the results to the corresponding list 
        
            t1 = time.time() # Start time 
            _ = code1(L,x,istart=0,iend=-1000,N0=N) # Computing the search for linear version 
            t2 = time.time() # End time
            linear.append(t2-t1) # Adding the results to the corresponding list 
            
            t1 = time.time() # Start time 
            _ = code1(L,x,istart=0,iend=-1000,N0=0) # Computing the search for binary version 
            t2 = time.time() # End time
            binary.append(t2-t1) # Adding the results to the corresponding list 
            
            
        results['hybrid'].append(np.mean(hybrid)) # Adding the average results 
        results['linear'].append(np.mean(linear)) # to the corresponding list
        results['binary'].append(np.mean(binary)) 
        
    outputs = results # Walltimes of code1

    # Plotting with denoising
    plt.figure()
    Xaxis = np.array([n for n in range(3,1000)])
    plt.plot(Xaxis,denoise(results['binary']), label='binary search')
    plt.plot(Xaxis,denoise(results['linear']), label='linear search')
    plt.plot(Xaxis,denoise(results['hybrid']), label='hybrid search')
    plt.yscale('log')
    #plt.xlim([0,150])
    plt.legend(loc="best")
    plt.xlabel('$N$')
    plt.ylabel('Walltime')
    title = 'Denoised walltimes of the different considered searching algorithms for $N_0$ = ' \
        + str(inputs) + ' x $N$'
    plt.title(title)
    plt.show()

    return outputs



def createTable():
	"""Function for question 2. Return dictionary providing mapping from codons
    to amino acids. "_" indicates that codon does not correspond to any amino acid.
	"""
	table = {
		'ATA':'i', 'ATC':'i', 'ATT':'i', 'ATG':'m',
		'ACA':'t', 'ACC':'t', 'ACG':'t', 'ACT':'t',
		'AAC':'n', 'AAT':'n', 'AAA':'k', 'AAG':'k',
		'AGC':'s', 'AGT':'s', 'AGA':'r', 'AGG':'r',
		'CTA':'l', 'CTC':'l', 'CTG':'l', 'CTT':'l',
		'CCA':'p', 'CCC':'p', 'CCG':'p', 'CCT':'p',
		'CAC':'h', 'CAT':'h', 'CAA':'q', 'CAG':'q',
		'CGA':'r', 'CGC':'r', 'CGG':'r', 'CGT':'r',
		'GTA':'b', 'GTC':'b', 'GTG':'b', 'GTT':'b',
		'GCA':'a', 'GCC':'a', 'GCG':'a', 'GCT':'a',
		'GAC':'d', 'GAT':'d', 'GAA':'e', 'GAG':'e',
		'GGA':'g', 'GGC':'g', 'GGG':'g', 'GGT':'g',
		'TCA':'s', 'TCC':'s', 'TCG':'s', 'TCT':'s',
		'TTC':'f', 'TTT':'f', 'TTA':'l', 'TTG':'l',
		'TAC':'o', 'TAT':'o', 'TAA':'_', 'TAG':'_',
		'TGC':'c', 'TGT':'c', 'TGA':'_', 'TGG':'j',
	}
	return table


def heval(L,c2b, Base=20,Prime=97):
    """Convert list L to base-10 number mod Prime where Base specifies the base of L
        Function taken from lecture notes and modified.
    """
    f=0
    for l in L[:-1]:
            f = Base*(c2b[l]+f)
            h = (f + (c2b[L[-1]])) % Prime
    return h

def findAA(S,L_p):
    """Question 2: Complete function to find amino acid patterns in
    amino acid sequence, S
    Input:
        S: String corresponding to an amino acid sequence
        L_p: List of p length-3m strings. Each string corresponds to a gene
        sequence
    Output:
        L_out: List of lists containing locations of amino-acids in S.
        L_out[i] should be a list of integers containing all locations in S at
        which the amino acid sequence corresponding to the gene sequence in L_p[i] occur.
        If one or more codons in L_p[i] do not correspond to an amino acid (such
        that '_' would appear in the corresponding amino acid sequence), set
        L_out[i]=-1000.
    """
    c2b = {
        'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4,
        'f' : 5, 'g' : 6, 'h' : 7, 'i' : 8, 'j' : 9,
        'k' : 10, 'l' : 11, 'm' : 12, 'n' : 13, 'o' : 14, 
        'p' : 15, 'q' : 16, 'r' : 17, 's' : 18, 't' :19
        }
    
    #use/modify the code below as needed
    n = len(S)
    p = len(L_p)
    k = len(L_p[0])
    m = k // 3
    L_out = [[] for i in range(p)]
    T = createTable()
    
    L_AA = ['' for i in range(p)] # Here we store the amino acids we are looking for in S
    
    prime = 97 # Prime for remainder
    
    bm = (20**m) % prime # Computing those parameters only once !
    hi0 = heval(S[:m], c2b = c2b, Base = 20, Prime = prime) 

    
    for i in range(p) : # Going linearly through the gene sequences
    
    
        
        ## We first translate the gene sequences to the corresponding amino acid sequences ##
        for j in range(0,k,3) : # Going 3 by 3 over L_p[i] codons
            aminoAcid = T[L_p[i][j:j+3]] # Looking in the table for the corresponding 
            # amino acid
            if aminoAcid != '_' : # Verifying the codon is relevant 
                L_AA[i] += aminoAcid # Concatenating the strings
            else :
                L_out[i] =[-1000] # As required in the subject 
                break

        # L_p[i] has been translated to L_AA[i] at this stage
        # We have the amino acid sequence corresponding to L_p[i]
        
        
        
        ## We now search for the location of the L_AA[i] amino acids in S ##
        if len(L_AA[i]) == m : # Only searching if all codons were coding for amino acids 
        
            ## We use the Rabin Karp algorithm and the code provided in the lecture slides
            # We first compute hashes of L_AA[i] and the first m elements of S and further 
            # compare the characters if the hashes correspond in first place
            ind=0
            hp = heval(L_AA[i],c2b = c2b, Base = 20, Prime = prime) 
            imatch=[]
            hi = 1 * hi0
            if hi == hp : # Hashes match
                if S[:m] == L_AA[i] : # Python string comparison 
                    imatch.append(ind)
            
            
            for ind in range(1,n - m + 1) :
                # Update rolling hash
                hi = (20 * hi - int(c2b[S[ind-1]]) * bm + int(c2b[S[ind-1+m]])) % prime
                
                if hi == hp : # If hashes match, check if strings match
                    if S[ind:ind+m] == L_AA[i] : 
                        imatch.append(ind)
            L_out[i] = imatch
                
    return L_out #please do not modify

#%matplotlib inline

if __name__=='__main__':
    #When "run project1" is used in the terminal, the condition above evaluates
    #to True and the code below will run. If "import p1soln" is used, the
    #condition evaluates to False, and the code below will not run.

    print('')
    ## Question 1 ##
    print('Starting computing figures of question 1')
    inputs=[0,0.25,0.5,0.75,1]
    for inp in inputs :
        outputs = test_code1(inp)
    print('All figures computed')
    print('')

    ###########################################################################

    ## Question 2 ##
    #Sample code for loading example sequence for question 2 (uncomment and use as needed)
    print('Starting tests related to question 2')
    infile = open('Sexample.txt','r')
    S = infile.read()
    infile.close()
    
    # Testing findAA
    T = createTable()
    bases = ['A','T','C','G']
    from numpy import random
    random.seed(662607015)
    for test in range(10) : # Testing 10 times
        L_p = ['' for g in range(10)] # 10 genes sequences
        for i in range(10) : # Creating the 10 genes
            baseIndex = random.randint(4, size = 3 * 3) # m = 3
            for ind in baseIndex :
                L_p[i] += bases[ind] # Generating random L_p
        L_out = findAA(S,L_p) # Computing the location of sequences
        i = -1
        for indexList in L_out :
            i+=1
            if indexList and indexList[0] != -1000 : # The sequence is well in S
                for index in indexList :
                    # Asserting that the corresponding amino acid is coded by 
                    # a codon that is in L_p
                    assert L_p[i][0:3] in [codon for codon in T if T[codon] == S[index]]
                    assert L_p[i][3:6] in [codon for codon in T if T[codon] == S[index+1]]
                    assert L_p[i][6:9] in [codon for codon in T if T[codon] == S[index+2]]

    print('All examples passed the tests')
    print('')
        




