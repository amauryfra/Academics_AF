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
from IVPsolver import *
from numpy import random
from math import pi
import uuid
import json
from progress.bar import FillingCirclesBar
import time

# Computing only once
twoPi = 2 * pi
piOn2 = pi / 2
threePiOn2 = (3 * pi) / 2

# Initializing random number generator
random.seed(1008*1996)

def parameterDraw() :
    """
    This function performs the parameters draw for the trajectory generation.

    Parameters
    ----------
    None

    Returns
    -------
    c : 4-dimensional numpy array - the costate constants
    q0 : 5-dimensional numpy array - the initial state vector
    tf : float - the final time value
    """

    # Generating c
    c = random.normal(loc = 0, scale = 200, size = 4)

    # Generating q0
    x0 = random.normal(loc = 0, scale = 4.0)
    xDot0 = random.normal(loc = 0, scale = 4.0)
    z0 = random.normal(loc = 0, scale = 4.0)
    zDot0 = random.normal(loc = 0, scale = 4.0)
    # Better chances having the UAV not upside down at start
    possibleTheta0 = [random.uniform(low = 0, high = piOn2), \
                      random.uniform(low = threePiOn2, high = twoPi), \
                          random.uniform(low = piOn2, high = threePiOn2)]
    theta0 = random.choice(possibleTheta0, p = [0.325, 0.325, 0.35])
    q0 = np.array([x0, xDot0, z0, zDot0, theta0])

    # Generating tf
    tf = random.lognormal(mean = 0.75, sigma = 0.20)

    return c, q0, tf


def qfLocation(q0, qf) :
    """
    This function returns the location of the terminal state position (xf,zf)
    with respect to the initial position (x0,z0).

    Parameters
    ----------
    q0 : 5-dimensional numpy array - the initial state vector
    qf : 5-dimensional numpy array - the final state vector

    Returns
    -------
    location : string - the terminal state position
    """
    if qf[0] >= q0[0] :
        if qf[2] >= q0[2] :
            return 'upper right'
        else :
            return 'lower right'
    else :
        if qf[2] >= q0[2] :
            return 'upper left'
        else :
            return 'lower left'

def distribInit() :
    """
    This function initializes the distribution dictionnary of final positions
    with respect to corresponding initial positions.

    Parameters
    ----------
    None

    Returns
    -------
    distrib : dictionnary - the initialized distribution of final positions
        with respect to corresponding initial positions.
    """

    return {'upper left' : {'num' : 0, 'full' : False},
            'lower left' : {'num' : 0, 'full' : False},
            'upper right' : {'num' : 0, 'full' : False},
            'lower right' : {'num' : 0, 'full' : False}}


def qfAdmissible(q0, qf, maxDist, distrib, maxPerLoc) :
    """
    This function returns True if the final state is admissible regarding
    our arbitrary conditions.

    Parameters
    ----------
    q0 : 5-dimensional numpy array - the initial state vector
    qf : 5-dimensional numpy array - the final state vector
    maxDist : float - the maximum distance authorized between the initial
        and final positions
    distrib : dictionnary - the current distribution of final positions
        with respect to corresponding initial positions
    maxPerLoc : integer - the maximum final positions per location

    Returns
    -------
    admissibility : boolean - set to True if the given qf is admissible in the dataset
    """

    # Testing distance
    diff = qf - q0
    xzDiff = np.array([diff[0], diff[2]])
    if np.linalg.norm(xzDiff) > maxDist :
        return False

    # Testing location
    loc = qfLocation(q0, qf)
    if distrib[loc]['full'] :
        return False
    else :
        distrib[loc]['num'] += 1
        if distrib[loc]['num'] == maxPerLoc :
            distrib[loc]['full'] = True
        return True



def generateBatch(targetBatchSize, N, maxDist) :
    """
    This function generates a batch of trajectories and controls, computed using
    (2.18).

    Parameters
    ----------
    targetBatchSize : integer - the number of trajectories and controls samples
        in the batch, must be divisible by 4
    N : integer - the number of evaluation points
    maxDist : float - the maximum distance authorized between the initial
        and final positions

    Returns
    -------
    None
    """

    # Divisibility by 4 required
    targetBatchSize = targetBatchSize - targetBatchSize % 4

    # Initialization
    currentBatchSize = 0
    distrib = distribInit()
    batch = {}

    # Evenly distributed final positions with respect to initial positions
    maxPerLoc = int(targetBatchSize / 4)

    # Progress bar
    bar = FillingCirclesBar('Processing current batch', max = targetBatchSize)

    # Generating batch
    while currentBatchSize < targetBatchSize :

        # Drawing parameters
        c, q0, tf = parameterDraw()

        # Solving IVP
        sol, u = solve(q0, c, tf, N, extractControls = True)
        qf = sol.y[:5,N-1]

        # Verifying if qf is admissible
        if qfAdmissible(q0, qf, maxDist, distrib, maxPerLoc) :

            currentBatchSize += 1
            bar.next()
            trajId = str(uuid.uuid4().int)[:12] # Unique id for each trajectory

            # Storing values
            batch[trajId] = {}
            batch[trajId]['q'] = sol.y.tolist()
            batch[trajId]['u'] = u.tolist()
            batch[trajId]['c'] = c.tolist()


    # Save as you go
    batchId = str(uuid.uuid4().int)[:6]
    fileDir = '/.../.../.../.../' \
    '.../trajectoryBatches.nosync/'
    with open(fileDir + batchId + '.json', 'w') as file:
        json.dump(batch, file)

    bar.finish()
    return

# Settings
nbBatches = 250
targetBatchSize = 100
N = 100
maxDist = 6.0


def generateDataset(nbBatches = nbBatches, targetBatchSize = targetBatchSize, \
                    N = N, maxDist = maxDist) :
    """
    This function generates the synthetic trajectories and controls dataset.

    Parameters
    ----------
    nbBatches : integer - the number of batches required
    targetBatchSize : integer - the number of trajectories and controls samples
        in the batch, must be divisible by 4
    N : integer - the number of evaluation points
    maxDist : float - the maximum distance authorized between the initial
        and final positions

    Returns
    -------
    None
    """

    print('')
    for batchNb in range(nbBatches):
        print('----------------------------------------')
        print('Batch number ' + str(batchNb + 1) + '/' + str(nbBatches))
        t0 = time.time()
        # Generate batch
        generateBatch(targetBatchSize, N, maxDist)
        t1 = time.time()
        print('Completed in ' + str(t1-t0)[:5] +'s')

    print('........................................')
    print('........................................')
    print('Dataset generated')
    print('')


if __name__ == '__main__' :
    generateDataset()
