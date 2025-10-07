#!/usr/bin/env python3
'''
Doc String
'''
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def solve_heat(x_stop = 1., t_stop = 0.2, dx = 0.2, dt = 0.02, c2 = 1):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    FILL THIS OUT AND DO NOT FORGET

    c2: Float
        c^2, the square of the diffusion coeffiecient
    Returns
    -------
    x, t: 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace * nTime
    '''
    # Get grid sizes
    N = int(t_stop / dt)
    M = int(x_stop / dx)

    # Set up space and time grid
    t = np.linspace(0, t_stop, N)
    x = np.linspace(0, x_stop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    U[:,0] = 4*x - 4*x**2

    #Get our "r" coefficient
    r = c2 * (dt/dx**2)

    # Solve our equation 
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])

    # Return our pretty solution to the caller
    return t, x, U
