"""
Imperial College London
MSc Applied Mathematics
This code has been written as part of the MSc project 'Deep Neural Networks
for Real-time Trajectory Planning'
Author : Amaury FRANCOU - CID: 01258326
Supervisor : Dr Dante KALISE
"""

# Imports
import numpy as np
from simulator import simulate
from datasetGenerator import parameterDraw
from IVPsolver import solve

##############################################################################
# Settings
method = 'Euler' # RK4 or Euler
modelNumber = 37 # Model to use

q0 = np.zeros(5)
qf = np.array([0,0,1,0,0])

printState = True
printControls = True

drawParameters = True # Draw q0 and qf in the same manner as dataset generator
maxDist = 6.0 # Max XZ distance between q0 and qf
onlyUp = False # Only accept upwards XZ trajectories



                        #############################

### Euler ####
Nmax = 150 # Number of points along trajectory
h = 2.5 * 1e-3 # Time step

eps = 1e-4 # Tolerance on state

targetXZ = False
targetQ = False
printCircle = False


### RK4 ###
Tmax = 2.0
#Nmax = 100 # Number of points along trajectory
maxStep = 1e-4


##############################################################################



def getInitialFinalStates(maxDist = maxDist, onlyUp = onlyUp) :
    """
    This function samples initial and final states in the same manner as in
    the dataset generator.

    Parameters
    ----------
    None

    Returns
    -------
    q0 : 5-dimensional numpy array - the initial state vector
    qf : 5-dimensional numpy array - the final state vector

    """

    # Sampling until condition is verified
    condition = True
    while condition :
        # Drawing parameters
        c, q0, tf = parameterDraw()

        # Solving IVP
        N = 100
        sol = solve(q0, c, tf, N = N, extractControls = False)
        qf = sol.y[:5,N-1]

        # Computing XZ distance
        diff = qf - q0
        xzDiff = np.array([diff[0], diff[2]])
        if onlyUp :
            if np.linalg.norm(xzDiff) < maxDist and diff[2] > 0 :
                condition = False
        else :
            if np.linalg.norm(xzDiff) < maxDist :
                condition = False

    return q0, qf


if __name__ == '__main__' :

    np.set_printoptions(precision=3)

    if drawParameters :
        q0, qf = getInitialFinalStates()

    converged, qStorage, uStorage = simulate(modelNumber = modelNumber, method = method,\
                                             qf = qf, q0 = q0,\
                                                 Nmax = Nmax, h = h, \
                                 eps = eps, Tmax = Tmax, maxStep = maxStep,\
                                     targetXZ = targetXZ, targetQ = targetQ, \
                                     printCircle = printCircle,
                                 printState = printState, printControls = printControls)
    print('')
    print('Converged : ', converged)
    print('')
