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
from scipy.integrate import solve_ivp


# Setting constants
g = 9.81
uTmax = 14
uTmin = 0.8
uRmax = 4.6


def aFunc(t, y, c) :
    """
    This function computes the value of the modulating function a at time t,
    according to (2.12).

    Parameters
    ----------
    t : float - the time value
    y : 6-dimensional numpy array - the augmented state vector
    c : 4-dimensional numpy array - the costate constants
    Returns
    -------
    a : float - the modulating function a at time t
    """
    return - 0.5 * ((c[1] - c[0] * t) * np.sin(y[4]) + (c[3] - c[2] * t) * np.cos(y[4]))

def aFuncVec(tf, N, y, c) :
    """
    This function computes the value of the modulating function a,
    according to (2.12), in a vectorized fashion.

    Parameters
    ----------
    tf : float - the final time value
    N : integer - the number of evaluation points
    y : 6xN-dimensional numpy array - the augmented state vectors on [0,tf]
    c : 4-dimensional numpy array - the costate constants
    Returns
    -------
    aVec : N-dimensional numpy array - the modulating function a on [0,tf]
    """

    time = np.linspace(0,tf,N)
    return - 0.5 * ((c[1] - c[0] * time) * np.sin(y[4,:]) \
                   + (c[3] - c[2] * time) * np.cos(y[4,:]))


def uTcontrol(a, uTmin = uTmin, uTmax = uTmax) :
    """
    This function computes the thrust control as defined in (2.13).

    Parameters
    ----------
    a : float - the modulating function value a(t)
    uTmin : float - the minimal admissible thrust control
    uTmax : float - the maximal admissible thrust control

    Returns
    -------
    uTcontrol : float - the computed thrust control
    """
    return min(max(a, uTmin), uTmax)


def uTcontrolVec(aVec, uTmin = uTmin, uTmax = uTmax) :
    """
    This function computes the thrust controls as defined in (2.13),
    in a vectorized fashion.

    Parameters
    ----------
    aVec  : N-dimensional numpy array - the modulating function a on [0,tf]
    uTmin : float - the minimal admissible thrust control
    uTmax : float - the maximal admissible thrust control

    Returns
    -------
    uTcontrolVec : N-dimensional numpy array - the computed thrust controls on [0,tf]
    """
    return np.clip(aVec, uTmin, uTmax)


def uRcontrol(b, uRmax = uRmax) :
    """
    This function computes the torque control as defined in (2.16).

    Parameters
    ----------
    b : float - the modulating function value b(t)
    uRmax : float - the maximal admissible torque control

    Returns
    -------
    uRcontrol : float - the computed torque control
    """
    return min(max(b, -uRmax), uRmax)


def uRcontrolVec(bVec, uRmax = uRmax) :
    """
    This function computes the torque control as defined in (2.16),
    in a vectorized fashion.

    Parameters
    ----------
    bVec : N-dimensional numpy array - the modulating function value b on [0,tf]
    uRmax : float - the maximal admissible torque control

    Returns
    -------
    uRcontrolvec : N-dimensional numpy array - the computed torque controls on [0,tf]
    """
    return np.clip(bVec, -uRmax, uRmax)


def solve(q0, c, tf, N, extractControls = False) :
    """
    This function solves the initial value problem defined in (2.18) for a target
    final time tf.

    Parameters
    ----------
    q0 : 5-dimensional numpy array - the initial state vector
    c : 4-dimensional numpy array - the costate constants
    tf : float - the final time value
    N : integer - the number of evaluation points
    extractControls - boolean - if set to True the function returns the controls

    Returns
    -------
    sol : scipy OdeSolution instance - sol.y contains the values of the solution
    u : 2xN-dimensional numpy array - the controls on [0,tf]
    """

    def fDynamics(t, y, c = c) :
        """
        This function computes the derivative of the augmented state vector according
        to (2.18).

        Parameters
        ----------
        t : float - the time value
        y : 6-dimensional numpy array - the augmented state vector
        c : 4-dimensional numpy array - the costate constants

        Returns
        -------
        f : 6-dimensional numpy array - the augmented state vector derivative
        """

        # Computing controls
        uT = uTcontrol(a = aFunc(t = t, y = y, c = c))
        uR = uRcontrol(b = y[5])

        return np.array([y[1], uT * np.sin(y[4]), y[3], uT * np.cos(y[4]) - g, uR, \
                          0.5 * uT * ((c[1] - c[0] * t) * np.cos(y[4])  \
                                      - (c[3] - c[2] * t) * np.sin(y[4]))])

    # Initial value of augmented state vector
    y0 = np.append(q0,0)
    # Scipy IVP solver using explicit Runge-Kutta method of order 5(4)
    sol = solve_ivp(fDynamics, [0, tf], y0, method = 'RK45', max_step = 1e-2, \
                    t_eval = np.linspace(0,tf,N))

    # Controls extraction
    if extractControls :
        # uT
        aVec = aFuncVec(tf = tf, N = N, y = sol.y, c = c)
        print(aVec)
        uTcontrols = uTcontrolVec(aVec)
        # uR
        uRcontrols = uRcontrolVec(bVec = sol.y[5,:])
        # u = [uT, uR]
        u = np.row_stack((uTcontrols,uRcontrols))

        return sol, u

    return sol
