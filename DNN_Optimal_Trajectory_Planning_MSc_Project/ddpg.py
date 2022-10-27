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
from math import pi
import tensorflow as tf
from dynamicalSolver import fDynamics
from buffer import BasicBuffer_b
from trainNN import controlsActivation
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import model_from_json
from datasetPreparation import encodeTheta
import copy
import os
# Constants
uTmax = 14
uTmin = 0.8
uRmax = 4.6
twoPi = 2 * pi
g = 9.81

def qfDistanceMeasure(q, qf) :
    """
    This function samples initial and final states in the same manner as in
    the dataset generator.

    Parameters
    ----------
    q : 5-dimensional numpy array - the current state vector
    qf : 5-dimensional numpy array - the final state vector

    Returns
    -------
    g : float - a measure of the distance to the final state

    """

    qhat = qf - q
    qhat[4] = qhat[4] % twoPi
    qhat[4] = np.square(np.arctan2(np.sin(qhat[4]), np.cos(qhat[4])))

    return np.linalg.norm(qhat)**2


def getReward(qNext, q, u ,qf, h, alpha, beta, gamma) :
    """
    This function computes the reward of the DDPG method.

    Parameters
    ----------
    qNext : 5-dimensional numpy array - the next state vector
    q : 5-dimensional numpy array - the current state vector
    u : 2-dimensional numpy array - the control vector
    qf : 5-dimensional numpy array - the final state vector
    h : float - the time step used
    alpha : float - reward parameter
    beta : float - reward parameter
    gamma : float - reward parameter


    Returns
    -------
    r : float - the instant reward

    """
    dist = qfDistanceMeasure(q, qf)
    newDist = qfDistanceMeasure(qNext, qf)
    sign = np.sign(dist - newDist)

    if sign >= 0 :
        r = alpha * np.abs(1 / newDist) - beta * np.linalg.norm(u)**2
    else :
        r = 0 - gamma * np.linalg.norm(u)**2
    return r

def nextState(q, u, h) :
    """
    This function computes the next state of the quadrotor based on the Euler
    method.

    Parameters
    ----------
    q : 5-dimensional numpy array - the current state vector
    u : 2-dimensional numpy array - the control vector
    h : float - the time step used

    Returns
    -------
    qNew : 5-dimensional numpy array - the next state vector

    """
    qNew = copy.deepcopy(q)
    qNew += h * fDynamics(t = -1, q = q, u = u)

    return qNew

def isArrived(q, qf, eps) :
    """
    This function returns True if the vehicle has arrived close to its final
    state within the given tolerance.

    Parameters
    ----------
    q : 5-dimensional numpy array - the current state vector
    qf : 5-dimensional numpy array - the final state vector
    eps : float - the tolerance of closeness to required final state

    Returns
    -------
    isArrived : boolean - True if the quadrotor has arrived to qf

    """

    return (qfDistanceMeasure(q, qf) < eps)


def predictControl(q, qf, piModel, noiseScale) :
    """
    This function computes controls using the policy network and adds random
    noise for DRL exploration.

    Parameters
    ----------
    q : 5-dimensional numpy array - the current state vector
    qf : 5-dimensional numpy array - the final state vector
    piModel : keras object - the trained neural network
    noiseScale : 2-dimensional numpy array - scale parameters for random noise

    Returns
    -------
    u : 2-dimensional numpy array - the control vector

    """

    qhat = qf - q
    qhat[4] = qhat[4] % twoPi
    cosTheta, sintheta = encodeTheta(qhat[4])
    qhatAugm = np.concatenate((qhat[:4],np.array([cosTheta, sintheta])))

    u = piModel.predict(np.array([qhatAugm]), verbose = 0)[0]
    u += np.multiply(noiseScale, np.random.randn(2))

    u[0] = np.clip(u[0], uTmin, uTmax)
    u[1] = np.clip(u[1], -uRmax, uRmax)

    return u

q0 = np.zeros(5)
qf = np.array([0,0,0,0,3.1415])

def ddpg(trainingNumber, modelNumber = 2, numberEpisodes = 100, \
         q0 = q0, qf = qf, h = 1 * 1e-3, eps = 1e-2, maxEpisodeLength = 10000, \
         startSteps = 500, noiseScale = np.array([1.25,1.00]), batchSize = 32, discount = 0.99, \
             decayFactor = 0.99, alpha = 100, beta = 50, gamma = 50) :
    """
    This function performs the deep deterministic policy gradient method, loading a
    previously trained quadrotor controller.

    Parameters
    ----------
    trainingNumber : int - the current training number
    modelNumber : int - the pre-trained model number to use
    numberEpisodes : int - the number of episodes to perform
    q0 : 5-dimensional numpy array - the initial state vector
    qf : 5-dimensional numpy array - the final state vector
    h : float - the time step used
    eps : float - the tolerance of closeness to required final state
    maxEpisodeLength : int - maximum number of iteration per episode
    startSteps : int - the number of start steps to reach before adding noise
        to controls for exploration
    noiseScale : 2-dimensional numpy array - scale parameters for random noise
    batchSize : int - size of batch
    discount : float - discount factor for future Q values
    decayFactor : float - decay factor for networks update
    alpha : float - reward parameter
    beta : float - reward parameter
    gamma : float - reward parameter


    Returns
    -------
    rewards : M-dimensional numpy array - the rewards across training
    qLosses : N-dimensional numpy array - loss of Q network across training
    piLosses : N-dimensional numpy array - loss of Pi network across training

    """

    #############################################################################
    # Loading model Pi
    modelNumber = str(modelNumber)
    pathToModel = 'DNN_Models/training_' + modelNumber
    json_file = open(pathToModel + '/model' + modelNumber + '.json', 'r')
    loaded_model_json = json_file.read()
    piModel = model_from_json(loaded_model_json, \
                              custom_objects={'controlsActivation': Activation(controlsActivation)})
    piModel.load_weights(pathToModel + '/model' + modelNumber + '.h5')
    # Loading model piTarget
    piTarget = model_from_json(loaded_model_json, \
                               custom_objects={'controlsActivation': Activation(controlsActivation)})
    piTarget.load_weights(pathToModel + '/model' + modelNumber + '.h5')
    json_file.close()


    # Creating model Q
    Q = tf.keras.Sequential()
    Q.add(tf.keras.layers.Input(shape  = 5 + 2))
    for layer in range(5) :
        Q.add(tf.keras.layers.Dense(units = 32, activation='relu'))
    Q.add(tf.keras.layers.Dense(units = 1, activation = None))
    # Creating model Qtarget
    Qtarget = tf.keras.Sequential()
    Qtarget.add(tf.keras.layers.Input(shape  = 5 + 2))
    for layer in range(5) :
        Qtarget.add(tf.keras.layers.Dense(units = 32, activation='relu'))
    Qtarget.add(tf.keras.layers.Dense(units = 1, activation = None))

    #############################################################################

    # Replay buffer
    replay_buffer = BasicBuffer_b(size = int(1e6), obs_dim = 5, act_dim = 2)

    # For network training
    piModel_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    Q_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)


    #############################################################################

    # Episodes and training
    rewards = []
    qLosses = []
    piLosses = []
    numSteps = 0

    for episodeNb in range(numberEpisodes) :

        # Initialization
        q = copy.deepcopy(q0)
        episodeRewards = 0
        episodeLength = 0
        arrived = False

        while not (arrived or (episodeLength == maxEpisodeLength)) :

            if numSteps == startSteps + 1 :
              print('........... Starting using physics-informed model .............')


            if numSteps > startSteps :
                u = predictControl(q, qf, piModel, noiseScale)
            else :
                uT = np.random.normal(loc = 7.4, scale = 3.3)
                uT =  np.clip(uT, uTmin, uTmax)
                uR = np.random.normal(loc = 0, scale = 2.3)
                uR = np.clip(uR, -uRmax, uTmax)
                u = np.array([uT, uR])

            numSteps += 1

            # Next step in navigation
            qNext = nextState(q, u, h)
            qNext[4] = qNext[4] % twoPi
            reward = getReward(qNext, q, u ,qf, h, alpha, beta, gamma)
            arrived = isArrived(q, qf, eps)


            episodeRewards += reward
            episodeLength += 1

            # Ignore arrived if time horizon reached
            if episodeLength == maxEpisodeLength :
                arrivedStorage = False
            else :
                arrivedStorage = arrived

            # Store navigation to replay buffer
            replay_buffer.push(q, u, reward, qNext, arrivedStorage)

            # Moving on
            q = copy.deepcopy(qNext)

        # Perform the gradient descent/ascent updates
        for _ in range(episodeLength) :

            # Sampling from buffer
            States, Controls, Rewards, NextStates, ArrivalStatus = \
                replay_buffer.sample(batchSize)
            States = np.asarray(States, dtype=np.float32)
            Controls = np.asarray(Controls, dtype=np.float32)
            Rewards = np.asarray(Rewards, dtype=np.float32)
            NextStates = np.asarray(NextStates, dtype=np.float32)
            ArrivalStatus = np.asarray(ArrivalStatus, dtype=np.float32)
            StatesTensor = tf.convert_to_tensor(States)

            # Optimizating pi
            with tf.GradientTape() as tape2 :
                StatesHat = qf - States
                cosTheta, sinTheta = encodeTheta(StatesHat[:,4])
                StatesHat = StatesHat[:,:4]
                StatesHat = np.column_stack((StatesHat,cosTheta))
                StatesHat =  np.column_stack((StatesHat,sinTheta))
                EvaluatedControls = piModel(StatesHat)
                args = tf.keras.layers.concatenate([StatesTensor, EvaluatedControls], axis=1)
                Qval = Q(args)
                piLoss =  -tf.reduce_mean(Qval)
                piGrad = tape2.gradient(piLoss, piModel.trainable_variables)
            array = np.random.normal(size=6)
            piModel_optimizer.apply_gradients(zip(piGrad, piModel.trainable_variables))
            piLosses.append(piLoss)

            # Optimizating Q
            with tf.GradientTape() as tape :
                NextStatesHat = qf - NextStates
                cosTheta, sinTheta = encodeTheta(NextStatesHat[:,4])
                NextStatesHat = NextStatesHat[:,:4]
                NextStatesHat = np.column_stack((NextStatesHat,cosTheta))
                NextStatesHat =  np.column_stack((NextStatesHat,sinTheta))
                nextControls = piTarget(NextStatesHat)
                args = np.concatenate((NextStates, nextControls), axis=1)
                QtargetVals = Rewards + discount * (1 - ArrivalStatus) * Qtarget(args)
                args2 = np.concatenate((States, Controls), axis=1)
                Qvals = Q(args2)
                Qloss = tf.reduce_mean((Qvals - QtargetVals)**2)
                Qgrad = tape.gradient(Qloss, Q.trainable_variables)
            Q_optimizer.apply_gradients(zip(Qgrad, Q.trainable_variables))
            qLosses.append(Qloss)

            # Updating Q
            QtWeights = np.array(Qtarget.get_weights(), dtype = object)
            QWeights = np.array(Q.get_weights(), dtype = object)
            QFinalWeights = decayFactor * QtWeights + (1 - decayFactor) * QWeights
            Qtarget.set_weights(QFinalWeights)

            # Updating pi
            piTargetWeights = np.array(piTarget.get_weights(), dtype = object)
            piWeights = np.array(piModel.get_weights(), dtype = object)
            piFinalWeights = decayFactor * piTargetWeights + (1 - decayFactor) * piWeights
            piTarget.set_weights(piFinalWeights)

        if episodeNb == 0 :
            print('')
        print("Episode : ", episodeNb + 1, "Reward : ", '{:,}'.format(episodeRewards),\
              'Episode length : ', episodeLength, 'Last state : ', q)

        rewards.append(episodeRewards)

    ###################################################################################
    # Saving model and metrics
    trainingNumber = str(trainingNumber)
    path = 'DRL_Models/' + 'training_' + trainingNumber
    os.mkdir(path)
    filePath = 'DRL_Models/' + 'training_' + trainingNumber + '/'

    model_json = piTarget.to_json()
    with open(filePath + 'piModel' + trainingNumber + '.json', 'w') as json_file :
        json_file.write(model_json)
    # Save weights to HDF5
    piTarget.save_weights(filePath + 'piModel' + trainingNumber + '.h5')

    model_json = Qtarget.to_json()
    with open(filePath + 'Qmodel' + trainingNumber + '.json', 'w') as json_file :
        json_file.write(model_json)
    # Save weights to HDF5
    Qtarget.save_weights(filePath + 'Qmodel' + trainingNumber + '.h5')
    print("Saved models to HDD")
    print('')

    return rewards, qLosses, piLosses

if __name__ == "__main__" :

    trainingNumber = 9
    modelNumber = 2
    numberEpisodes = 1500

    maxEpisodeLength = 200 # 3.56 seconds
    startSteps = maxEpisodeLength * 150
    noiseScale = np.array([0.15 * (uTmax-uTmin), 0.15 * 2 * uRmax])

    alpha = 10000
    beta = 1
    gamma = 10


    discount = 0.95 # 0.99
    decayFactor = 0.95 # 0.99
    batchSize = 32

    h = 3.5 * 1e-2
    eps = 1e-2
    q0 = np.zeros(5)
    qf = np.array([0,0,0,0,3.1415])
    print('')
    rewards, qLosses, piLosses  = ddpg(trainingNumber, modelNumber, numberEpisodes, \
         q0, qf, h, eps, maxEpisodeLength, \
         startSteps, noiseScale, batchSize, discount, \
             decayFactor, alpha, beta, gamma)
