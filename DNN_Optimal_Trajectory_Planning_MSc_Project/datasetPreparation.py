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
from os import listdir
from os.path import isfile, join
import json
from math import pi
from progress.bar import ChargingBar
from sklearn.model_selection import train_test_split
import copy


def encodeTheta(theta, pi = pi) :
    """
    This function encodes the angle theta for use as input of the neural network.

    Parameters
    ----------
    theta : float - the angle

    Returns
    -------
    sHat : float - the first encoded number
    sCheck : float - the second encoded number

    """

    return np.cos(theta), np.sin(theta)

def prepareDataset(datasetPath) :
    """
    This function prepares the dataset in order to be ready for use in tensorflow
    model.

    Parameters
    ----------
    datasetPath : string - the absolute path to the dataset files

    Returns
    -------
    X_train : Nx5-dimensional numpy array - the states used for training
    X_test : Nx5-dimensional numpy array - the states used for testing
    Y_train : Nx2-dimensional numpy array - the controls used for training
    Y_test : Nx2-dimensional numpy array - the controls used for testing

    """

    # Getting batch file names
    batchList = [fileName for fileName in listdir(datasetPath) if isfile(join(datasetPath, fileName))]
    batchList.remove('.DS_Store')
    batchFiles = [datasetPath + '/' + batchName for batchName in batchList]

    # Computing only once
    twoPi = 2 * pi

    # Retrieval of the number of points in each trajectories
    batchOne = open(batchFiles[0])
    batchOneData = json.load(batchOne)
    firstTrajId = next(iter(batchOneData))
    N = np.array(batchOneData[firstTrajId]['q']).shape[1]

    # Total size of dataset
    lenBatchList = len(batchList)
    pointsPerTraj = int((N-1)*N/2)
    trajPerFile = len(batchOneData)
    totalPoints = pointsPerTraj * trajPerFile * lenBatchList
    print('Total number of points in dataset : ', totalPoints)

    # Preparing storage
    qStorage = np.zeros(shape = (totalPoints,6))
    uStorage = np.zeros(shape = (totalPoints,2))

    # Progress bar
    print('...........................................')
    print('...........................................')
    print('Loading data')
    bar = ChargingBar('Loading data', max = lenBatchList)

    for fileNum, batchFile in enumerate(batchFiles) :

        # Batch loading
        batch = open(batchFiles[0])
        batchData = json.load(batch)
        bar.next()

        for batchNum, trajId in enumerate(batchData) :

            # For stacking
            startAt = 0 + batchNum * pointsPerTraj + fileNum * trajPerFile * pointsPerTraj
            stopAt = N-1 + batchNum * pointsPerTraj + fileNum * trajPerFile * pointsPerTraj

            # Extracting arrays of trajectory
            q = np.array(batchData[trajId]['q'])[:5]
            u = np.array(batchData[trajId]['u'])

            # Augmentation procedure
            for k in range(N-1) :
                k+=1

                # Subtrajectory
                uSubtraj = copy.deepcopy(u[:,:N-k])
                qSubtraj = copy.deepcopy(q[:,:N-k+1])

                # Last state in subtrajectory
                qf = copy.deepcopy(q[:,N - k])
                qf = qf.reshape((5,1))
                qSubtraj = qf - qSubtraj # Considering the difference with qf
                qSubtraj = qSubtraj[:,:N-k]
                qSubtraj[4,:] = np.mod(qSubtraj[4,:], twoPi) # Theta in [0,2pi)

                # Encoding
                cosTheta, sinTheta = encodeTheta(qSubtraj[4,:])
                qSubtrajAugm = np.row_stack((qSubtraj[:4,:],cosTheta))
                qSubtrajAugm = np.row_stack((qSubtrajAugm,sinTheta))

                # Shaping the data for tensorflow
                qSubtrajAugm = qSubtrajAugm.T
                uSubtraj = uSubtraj.T

                # Adding to numpy storage
                qStorage[startAt:stopAt, :] = qSubtrajAugm
                uStorage[startAt:stopAt, :]  = uSubtraj

                # Updating
                startAt = stopAt
                stopAt += (N-k-1)

                # Flushing variables
                del uSubtraj, qSubtraj, qf, cosTheta, sinTheta, qSubtrajAugm

    bar.finish()

    # Splitting train and test sets with shuffling
    X_train, X_test, Y_train, Y_test = \
        train_test_split(qStorage, uStorage, test_size = 0.15, shuffle = True)


    print('Data loaded')
    print('...........................................')
    print('...........................................')

    assert X_train.shape[0] + X_test.shape[0] == totalPoints

    print('Total number of training points : ', X_train.shape[0])
    print('Total number of testing points : ', X_test.shape[0])

    return X_train, X_test, Y_train, Y_test
