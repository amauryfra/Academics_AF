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
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
import os
from datasetPreparation import prepareDataset
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import model_from_json

input_shape = (6,) # Inputs are state vectors

# Constants
uTmax = 14
uTmin = 0.8
uRmax = 4.6

##############################################################################
# Choose dataset old/new
whichDataset = 'old'
# Settings
training_number = 47
batch_size = 128
numberEpochs = 50
numberNeurons = 900
patience = 3
learning_rate = 1e-3
# Load previous model
loadModel = False
modelNumber = 34
##############################################################################


def controlsActivation(x, uTmin = uTmin, uTmax = uTmax, uRmax = uRmax) :
    """
    This function performs the activation for the output layer of the neural
    network based on the controls limits using a custom sigmoid function.

    Parameters
    ----------
    x  : Mx2-dimensional numpy array - the pre-activations for the output layer
        over all the batch
    uTmin : float - the minimal admissible thrust control
    uTmax : float - the maximal admissible thrust control
    uRmax : float - the maximal admissible torque control

    Returns
    -------
    activations : Mx2-dimensional numpy array - the activations for the
        output layer over all the batch
    """

    # Neurons
    n0 = x[:,0:1] # Shape is (batch_size, 1)
    n1 = x[:,1:2]

    # Neuron 0 provides uT
    x0 = ((uTmax-uTmin) * K.sigmoid(n0) + uTmin)
    # Neuron 1 provides uR
    x1 = uRmax * (2 * K.sigmoid(n1) - 1)

    return K.concatenate([x0,x1], axis = -1)


# Defining our tensorflow neural network model
model = Sequential([ # Tensorflow model
        Input(shape = input_shape), # Fully-connected layer
        Dense(numberNeurons, activation = 'relu'),
        Dropout(0.25),
        Dense(numberNeurons, activation = 'relu'),
        Dropout(0.25),
        Dense(numberNeurons, activation = 'relu'),
        Dropout(0.25),
        Dense(numberNeurons, activation = 'relu'),
        Dropout(0.25),
        Dense(numberNeurons, activation = 'relu'),
        Dense(2, activation = controlsActivation) # Output layer
    ])


# Using stochastic gradient descent
learning_rate = learning_rate
opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
# Load previous weights
if loadModel :
    modelNumber = str(modelNumber)
    pathToModel = 'DNN_Models/training_' + modelNumber
    json_file = open(pathToModel + '/model' + modelNumber + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,\
                           custom_objects={'controlsActivation': Activation(controlsActivation)})
    model.load_weights(pathToModel + '/model' + modelNumber + '.h5')
# Using mean squared error loss
model.compile(loss = 'mse', optimizer = opt, metrics = ['mse', 'mae', 'mape'])


# Saving model weights along the way
training_number = str(training_number)
filePath = "Trained Models/training_" + training_number +'/'
checkpoint_path = filePath + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                 save_weights_only = True,
                                                 verbose = 1)

# Perform early stopping
es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patience)

datasetPath = '/.../trajectoryBatches_' + whichDataset + '.nosync'

if __name__ == '__main__' :

    print('')
    # Loading data
    X_train, X_test, Y_train, Y_test = prepareDataset(datasetPath)
    print('')

    # Training the neural network
    model.summary()
    print('')
    print('Starting training')
    print('.......................................')
    print('.......................................')
    with tf.device('/device:GPU:0'):
      metrics = model.fit(x = X_train, y = Y_train, \
                          batch_size = batch_size, epochs = numberEpochs, verbose = 1,
                          validation_data = (X_test, Y_test), shuffle = True, \
                              callbacks = [es_callback,cp_callback])
    print('.......................................')
    print('.......................................')
    print('End of training')
    print('')

    # Saving model and metrics
    model_json = model.to_json()
    with open(filePath + "model" + training_number + ".json", "w") as json_file :
        json_file.write(model_json)
    # Save weights to HDF5
    model.save_weights(filePath + "model" + training_number + ".h5")
    print("Saved model to HDD")
    np.save(filePath + 'metricsTheoretical' + training_number + '.npy', metrics.history)
    print("Saved metrics to HDD")
    print('')
