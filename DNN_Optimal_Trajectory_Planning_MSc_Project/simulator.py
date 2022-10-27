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
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Activation
from dynamicalSolver import eulerSolveIVP, RK4solveIVP
from matplotlib.lines import Line2D
from trainNN import controlsActivation


def simulate(modelNumber, method, qf, q0 = np.zeros(5), Nmax = 500, h = 2.5 * 1e-3, eps = 1e-3,
             Tmax = 5.0, maxStep = 1e-3, targetXZ = False, targetQ = False, printCircle = False, printState = False, \
                 printControls = False) :
    """
    This function simulates a quadrotor flight, performed using the controller
    synthesized with the trained neural network.

    Parameters
    ----------
    modelNumber : int - the model number which to use
    method : string - preferred method
    qf : 5-dimensional numpy array - the final state vector
    q0 : 5-dimensional numpy array - the initial state vector
    Nmax : integer - the maximum number of time iterations before stopping
    h : float - the time step used
    eps : float - the tolerance of closeness to required final state
    Tmax : float - the maximum time on which to compute the trajectory
    targetXZ : boolean - Set to True if the end point is chosen to minimize
        the distance with the position input (xf,zf)
    targetQ : boolean - Set to True if the end point is chosen to minimize
        the distance with the state input qf
    printCircle : boolean - If set to True, prints an error circle with a radius
        corresponding to the XZ distance to the target endpoint
    printState : boolean - if set to true, prints the difference of states qhat
    printControls : boolean - if set to true, prints the controls u

    Returns
    -------
    converged : boolean - the variable is set to True if the final state has been reached
                            within the tolerance
    qStorage : 5xM-dimensional numpy array - the state vectors across time
    uStorage : 2xM-dimensional numpy array - the control vector across time

    """


    ##########################################################################


    #### Loading model ####
    modelNumber = str(modelNumber)
    pathToModel = 'DNN_Models/training_' + modelNumber
    json_file = open(pathToModel + '/model' + modelNumber + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,\
                           custom_objects={'controlsActivation': Activation(controlsActivation),\
                                           'Activation': Activation(controlsActivation)})
    model.load_weights(pathToModel + '/model' + modelNumber + '.h5')


    ##########################################################################


    #### Simulating trajectory ####

    if method == 'Euler' :
        qStorage, uStorage, converged, errArray, errXZarray  = \
            eulerSolveIVP(model = model, q0 = q0, qf = qf, Nmax = Nmax, h = h, eps = eps, \
                          printState = printState, printControls = printControls)
    if method == 'RK4' :
        qStorage = RK4solveIVP(model = model, q0 = q0, qf = qf, Tmax = Tmax, Nmax = Nmax, \
                               maxStep = maxStep, extractControls = True, printState = printState, \
                                   printControls = printControls)
        converged = True
        uStorage = None


    ##########################################################################


    #### Chosing end point ####
    if not converged :
        if targetXZ :
            tBestXZ = np.argmin(errXZarray)
            qStorage = qStorage[:,:tBestXZ]
        if targetQ :
            tBestq = np.argmin(errArray)
            qStorage = qStorage[:,:tBestq]


    ##########################################################################


    #### Plot ####
    fig, ax = plt.subplots(figsize=(9.75,12.25))


    # Initial and final points
    plt.plot(q0[0],q0[2],'bo', label ='$\mathbf{q}_0$', markersize = 4)
    plt.text(q0[0]-0.070, q0[2]-.035, 'Initial $\mathbf{q}_0$', fontsize = 12)
    plt.plot(qf[0],qf[2],'o', label = 'Target $\mathbf{q}_f$', markersize = 4, color = 'black')
    plt.text(qf[0]-.085, qf[2]+.02, 'Target $\mathbf{q}_f$', fontsize = 12)
    plt.plot(qStorage[0,-1], qStorage[2,-1], 'ro', label = 'Observed final point', markersize = 4)
    plt.text(qStorage[0,-1]+.03, qStorage[2,-1]+.03, 'Observed final point', fontsize = 12)

    # XZ Trajectory
    X = qStorage[0,:]
    Z = qStorage[2,:]
    plt.plot(X,Z)
    plt.xlabel("$x'$ (m)", fontsize = 15)
    plt.ylabel("$z'$ (m)", fontsize = 15)

    # Error circle
    xzf = np.array([qf[0],qf[2]])
    xzaf = np.array([qStorage[0,-1],qStorage[2,-1]])
    err = np.linalg.norm(xzf-xzaf)
    if printCircle :
        circle = plt.Circle((qf[0],qf[2]), err, fill = False, ls = '--')
        plt.text(qf[0]-err-0.005,qf[2]-err-0.005, 'Error on $(x_f,z_f)$ : ' + str(err)[:4] + 'm')
        plt.gca().add_patch(circle)

    # Parameters
    N = qStorage.shape[1]
    rate = int(0.15 * N)
    length = 0.5
    lengthOn2 = length / 2
    height = 0.07

    # Colormap
    cmap = matplotlib.cm.get_cmap('jet')

    # Target Final quadrotor drawing
    thetaf = np.rad2deg(qf[4])
    quadrotorX = np.array([qf[0] - lengthOn2, qf[0] - lengthOn2, \
                           qf[0] + lengthOn2, qf[0] + lengthOn2])
    quadrotorZ = np.array([qf[2] + height, qf[2], qf[2], qf[2] + height])
    rotate = matplotlib.transforms.Affine2D().rotate_deg_around(qf[0], qf[2], -thetaf)
    quadrotor = Line2D(quadrotorX, quadrotorZ, linewidth = 1.37,  \
                       drawstyle = 'steps-mid', color = 'black', linestyle = '--')
    quadrotor.set_transform(rotate + ax.transData)
    plt.gca().add_line(quadrotor)


    # Actual final quadrotor drawing
    thetafActual = np.rad2deg(qStorage[4,-1])
    quadrotorX = np.array([qStorage[0,-1] - lengthOn2, qStorage[0,-1] - lengthOn2, \
                           qStorage[0,-1] + lengthOn2, qStorage[0,-1] + lengthOn2])
    quadrotorZ = np.array([qStorage[2,-1] + height, qStorage[2,-1], qStorage[2,-1], \
                           qStorage[2,-1] + height])
    rotate = matplotlib.transforms.Affine2D().rotate_deg_around(qStorage[0,-1], qStorage[2,-1],\
                                                                -thetafActual)
    quadrotor = Line2D(quadrotorX, quadrotorZ, linewidth = 1.37, drawstyle = 'steps-mid',\
                       color = cmap((N-1)*h))
    quadrotor.set_transform(rotate + ax.transData)
    plt.gca().add_line(quadrotor)


    # Drawing quadrotor orientation
    for i in range(N) :

        if i % rate == 0 :

            xNow = qStorage[0,i]
            zNow = qStorage[2,i]
            thetaNow = np.rad2deg(qStorage[4,i])
            quadrotorX = np.array([xNow - lengthOn2, xNow - lengthOn2, xNow + lengthOn2, xNow + lengthOn2])
            quadrotorZ = np.array([zNow + height, zNow, zNow, zNow + height])
            rotate = matplotlib.transforms.Affine2D().rotate_deg_around(xNow, zNow, -thetaNow)
            quadrotor = Line2D(quadrotorX, quadrotorZ, linewidth = 1.37, drawstyle = 'steps-mid', \
                               color = cmap(i * h))
            quadrotor.set_transform(rotate + ax.transData)
            plt.gca().add_line(quadrotor)

    # Colorbar
    norm = matplotlib.colors.Normalize(vmin = 0,vmax = (N-1) * h)
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    cbar = plt.colorbar(sm, orientation = "horizontal")
    cbar.set_label('$t$ (s)', fontsize = 15)

    # Title
    plt.title('Quadrotor trajectory starting at $\mathbf{q}_0$' \
              + ' = ' + np.array2string(q0) + ', \n targeting endpoint $\mathbf{q}_f$ = ' \
                   + np.array2string(qf)  +r'$^\top$' + '.' + '\n')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('equal')
    plt.show()

    return converged, qStorage, uStorage
