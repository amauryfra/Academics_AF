"""Scientific Computation Project 3, part 2
CID : 01258326
"""
import numpy as np
import matplotlib.pyplot as plt

def solvePDE(Nx=200,Nt=4000,T=400,alpha=3,display=False):
    """PDE solver for part 2, question 1
    Input:

    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    alpha: model parameter
    display: if true, a contour plot of u is generated

    Output:
    x: spatial grid, size Nx
    t: times at which solution is computed, size Nt+u
    u,v: Nt+1 x Nx arrays containing solution
    """

    from scipy.integrate import solve_ivp
    #set up grid and needed parameters
    R = 0.5
    l = 2*np.pi/np.sqrt(1-R**2)
    fac = 50//l
    L = fac*l

    x = np.linspace(0,1,Nx+1)
    x = x[:-1]
    n = np.arange(Nx/2+1)
    k = 2*np.pi*n
    ik = 1j*k/L
    k2m = -k**2/L**2

    #set initial condition
    fac = 2*np.pi/l
    u0 = R*np.cos(fac*x)
    v0 = R*np.sin(fac*x)
    np.random.seed(1)
    ak = 1j*np.random.rand(Nx//2+1)
    ak[0]=0
    ak[-5:]=0
    a = np.fft.irfft(ak)
    u0 = u0 + 0.01*a

    def RHS(t,f):
        """
        RHS of PDEs called by solve_IVP
        """
        n = f.size
        u,v = f[:n//2],f[n//2:]
        a = 1-u**2-v**2
        b = 2+a
        uk =np.fft.rfft(u)
        vk = np.fft.rfft(v)
        du = np.fft.irfft(ik*uk)
        dv = np.fft.irfft(ik*vk)
        d2u = np.fft.irfft(k2m*uk)
        d2v = np.fft.irfft(k2m*vk)
        df = np.zeros_like(f)
        df[:n//2] = d2u + a*u - b*v + 4*du
        df[n//2:] = d2v +b*u + a*v -alpha*dv
        return df

    #compute solution
    Y0=np.zeros(2*Nx)
    Y0[:Nx]=u0
    Y0[Nx:]=v0
    t = np.linspace(0,T,Nt+1)
    sol = solve_ivp(RHS,[t[0],t[-1]],Y0,method='BDF',t_eval=t,rtol=1e-6,atol=1e-6)
    print("sol.success=",sol.success)
    f = sol.y.T
    u = f[:,:Nx]
    v = f[:,Nx:]
    print("finished simulation")

    if display:
        plt.figure()
        plt.contour(x,t,u,20)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Computed solution for u')

    return x,t,u,v


def analyzePDE():
    """
    Part 2, question 1
    Add input/output as needed

    """

    ## alpha = 0.1 ###
    x,t,u,v = solvePDE(Nx=200,Nt=4000,T=400,alpha = 0.1, display=False) # Computing u
    x0 = 10
    iskip = 15
    u = u[iskip:,:]
    u = u[:,x0]

    # Peaks | Taken from lecture slides
    du = np.diff(u)
    d2u = du[:-1]*du[1:]
    ind = np.argwhere(d2u<0) # Index of extrema
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title(r'Function $t\rightarrow u(10,t)$, for $\alpha = 0.1$')
    plt.xlabel('t')
    plt.ylabel('$u(10,t)$')
    plt.plot(t[iskip:],u)
    plt.plot(t[ind[0::2]+iskip+1],u[ind[0::2]+1],'x')

    # FFT | Taken from lecture slides
    c = np.fft.fft(u)
    c = np.fft.fftshift(c)/u.shape[0]
    k = np.arange(-u.shape[0]/2, u.shape[0]/2)
    plt.figure(figsize=(16, 12), dpi=80)
    plt.plot(k,np.abs(c))
    plt.title(r'Fourier coefficients for signal $u$, computed for $\alpha = 0.1$')
    plt.xlabel('Mode number $k$')
    plt.ylabel('$|c_k|$')



    ## alpha = 3 ###
    x,t,u,v = solvePDE(Nx=200,Nt=4000,T=400,alpha = 3, display=False) # Computing u
    x0 = 10
    iskip = 15
    u = u[iskip:,:]
    u = u[:,x0]

    # Peaks | Taken from lecture slides
    du = np.diff(u)
    d2u = du[:-1]*du[1:]
    ind = np.argwhere(d2u<0) # Index of extrema
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title(r'Function $t\rightarrow u(10,t)$, for $\alpha = 3$')
    plt.xlabel('t')
    plt.ylabel('$u(10,t)$')
    plt.plot(t[iskip:],u)
    plt.plot(t[ind[0::2]+iskip+1],u[ind[0::2]+1],'x')

    # FFT | Taken from lecture slides
    c = np.fft.fft(u)
    c = np.fft.fftshift(c)/u.shape[0]
    k = np.arange(-u.shape[0]/2, u.shape[0]/2)
    plt.figure(figsize=(16, 12), dpi=80)
    plt.plot(k,np.abs(c))
    plt.title(r'Fourier coefficients for signal $u$, computed for $\alpha = 3$')
    plt.xlabel('Mode number $k$')
    plt.ylabel('$|c_k|$')



    ## alpha = 1.6 ###
    x,t,u,v = solvePDE(Nx=200,Nt=4000,T=400,alpha = 1.6, display=False) # Computing u
    x0 = 10
    iskip = 15
    u = u[iskip:,:]
    u = u[:,x0]

    # Peaks | Taken from lecture slides
    du = np.diff(u)
    d2u = du[:-1]*du[1:]
    ind = np.argwhere(d2u<0) # Index of extrema
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title(r'Function $t\rightarrow u(10,t)$, for $\alpha = 1.6$')
    plt.xlabel('t')
    plt.ylabel('$u(10,t)$')
    plt.plot(t[iskip:],u)
    plt.plot(t[ind[0::2]+iskip+1],u[ind[0::2]+1],'x')



    ## alpha = 0.1 ###
    x,t,u,v = solvePDE(Nx=200,Nt=4000,T=400,alpha = 0.1, display=False) # Computing u
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title(r'Function $x\rightarrow u(x,t_0)$, for $\alpha = 0.1$')
    plt.xlabel('x')
    plt.ylabel('$u(x,t_0)$')
    plt.plot(x,u[150,:], label = 't = 150')
    plt.plot(x,u[100,:], label = 't = 100')
    plt.plot(x,u[50,:], label = 't = 50')
    plt.legend(loc = 'best')


    ## alpha = 3 ###
    x,t,u,v = solvePDE(Nx=200,Nt=4000,T=400,alpha = 3, display=False) # Computing u
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title(r'Function $x\rightarrow u(x,t_0)$, for $\alpha = 3$')
    plt.xlabel('x')
    plt.ylabel('$u(x,t_0)$')
    plt.plot(x,u[150,:], label = 't = 150')
    plt.plot(x,u[100,:], label = 't = 100')
    plt.plot(x,u[50,:], label = 't = 50')
    plt.legend(loc = 'best')


    # Attempting Bifurcation graph
    alphas = np.linspace(start = 0, stop = 10, num = 600) # alphas tested
    iskip = 15
    x0 = 10 # Fixing x
    chaos = {} # Storing results

    for index, alpha in enumerate(alphas) :
        print(index) # DELETE
        x,t,u,v = solvePDE(Nx=200,Nt=4000,T=400,alpha = alpha, display=False) # Computing u

        # Finding all peaks | Taken from lecture slides
        u = u[iskip:,:]
        u = u[:,x0]
        du = np.diff(u)
        d2u = du[:-1]*du[1:]
        ind = np.argwhere(d2u<0) # Index of extrema
        extremas = u[ind+1]

        chaos[str(index)] = extremas # Storing

    # Plots
    plt.figure(figsize=(16, 12), dpi=80)
    plt.title(r'Bifurcation graph related to $t\rightarrow u(x_0,t)$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Extremum values')
    plt.xlim([0,10])
    for key in chaos.keys() :
        alpha = alphas[int(key)]
        extremums = chaos[key]
        #extremums = np.unique(extremums.round(decimals=2))
        extremums = np.unique(np.floor(50*extremums)/50)
        plt.scatter(alpha * np.ones(extremums.shape[0]),extremums.reshape(extremums.shape[0],),\
                    s= 15, marker = 'x', color = 'black')

    return None



def fd4(u,h):
    """
    Part 2, question 2
    Input:
        u: n x m array whose first derivative will be computed along each column
        h: grid spacing
    Output:
        du: first derivative of u computed with 4th-order fd scheme
            (2nd-order near the boundaries)
    """

    du = np.zeros_like(u) #modify as needed

    #set parameters
    a = 1/12
    b = 2/3
    hinv = 1/h

    #compute derivative
    du[2:-2,:] = a*(u[:-4,:]-u[4:,:])+b*(u[3:-1,:]-u[1:-3,:]) #4th-order centered
    du[1,:] = 0.5*(u[2,:]-u[0,:]) #2nd-order centered
    du[-2,:] = 0.5*(u[-1,:]-u[-3,:]) #2nd-order centered
    du[0,:] = 0 #derivatives are zero at the boundaries
    du[-1,:] = 0
    du = hinv*du

    return du

def implicitFD(u,h):
    """
    Part 2, question 2
    Input:
        u: n x m array whose first derivative will be computed
        h: grid spacing
    Output:
        du: first derivative of u computed with implicit fd scheme
    """
    from scipy.linalg import solve_banded

    # Coefficients
    alpha = 0.435181352
    a = 1.551941906
    aDiv2 = a / 2
    b = 0.361328195
    bDiv4 = b / 4
    c = -0.042907397
    cDiv6 = c / 6

    # Coefficients in first and last rows
    alpha2 = 3
    a2 = -2.8333333333333335
    b2 = 1.5
    c2 = 1.5
    d2 = -0.16666666666666666

    N = u.shape[0]

    # Tridiagonal matrix | Effective storage
    Teff = np.zeros((3,N))
    array = alpha * np.ones(N-1)
    array[N-2] = 0
    array[N-3] = 0
    array[0] = 0
    array[1] = alpha2
    array[2] = alpha2
    Teff[0,1:] = array
    Teff[1,:] = np.ones(N)
    array = alpha * np.ones(N-1)
    array[0] = 0
    array[1] = 0
    array[N-2] = 0
    array[N-3] = alpha2
    array[N-4] = alpha2
    Teff[2,:N-1] = array

    # Banded matrix
    B = np.zeros((N,N))
    B[1:N-1,1:N-1] += np.diag(aDiv2 * np.ones(N-3), k = 1)
    B[1:N-1,1:N-1] += np.diag(bDiv4 * np.ones(N-4), k = 2)
    B[1:N-1,1:N-1] += np.diag(cDiv6 * np.ones(N-5), k = 3)
    B[1:N-1,1:N-1] += np.diag(-aDiv2 * np.ones(N-3), k = -1)
    B[1:N-1,1:N-1] += np.diag(-bDiv4 * np.ones(N-4), k = -2)
    B[1:N-1,1:N-1] += np.diag(-cDiv6 * np.ones(N-5), k = -3)
    B[N-3,N-2] = 0
    B[1,2] = b2
    B[2,3] = b2
    B[N-2,N-4] = -c2
    B[N-3,N-5] = -c2
    B[3,0] = -cDiv6
    B[N-2,N-3] = -d2
    B[N-3,N-4] = -d2
    B[2,1] = 0
    B[N-2,N-3] = -b2
    B[N-3,N-4] = -b2
    B[1,4] = d2
    B[2,5] = d2
    B[N-4,N-1] = cDiv6
    B[1,1] = a2
    B[2,2] = a2
    B[N-2,N-2] = -a2
    B[N-3,N-3] = -a2
    B[1,3] = c2
    B[2,4] = c2
    B[N-2,N-5] = -d2
    B[N-3,N-6] = -d2

    # Right hand side
    RHS = (1/h) * np.dot(B,u)
    uPrime = solve_banded((1,1),Teff,RHS) # Banded solve - Could use TDMA instead

    return uPrime



if __name__=='__main__':
    out = solvePDE(display=True) #call solvePDE and display contour plot of solution
