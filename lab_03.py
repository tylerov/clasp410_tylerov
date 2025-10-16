#!/usr/bin/env python3
'''
Doc String
'''
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Solution to problem 10.3 from fink/matthews as a nested list:
sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
[0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
[0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
[0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
[0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
[0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
[0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
[0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
[0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
[0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
[0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]
# Convert to an array and transpose it to get correct ordering:
sol10p3 = np.array(sol10p3).transpose()


def solve_heat(x_stop = 1., t_stop = 0.2, dx = 0.02, dt = 0.0002, c2 = 1):
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

    # Check our stability criterion:
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
       raise ValueError(f'DANGER: dt={dt} > dt_max = {dt_max}.')

    # Get grid sizes ( + 1 to include "0" as well)
    N = int(t_stop / dt) + 1
    M = int(x_stop / dx) + 1

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

def solve_heat_Neumann(x_stop = 1., t_stop = 0.2, dx = 0.02, dt = 0.0002, c2 = 1):
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

    # Check our stability criterion:
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
       raise ValueError(f'DANGER: dt={dt} > dt_max = {dt_max}.')

    # Get grid sizes ( + 1 to include "0" as well)
    N = int(t_stop / dt) + 1
    M = int(x_stop / dx) + 1

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
        U[0, j + 1] = U[1, j + 1]
        U[M - 1, j + 1] = U[ M - 2, j + 1]


    # Return our pretty solution to the caller
    return t, x, U

def plot_heatsolve(t, x, U, title = None, **kwargs):
    '''
    Plot the 2D solution for the 'solve_heat' function.

    Parameters
    ----------
    x, t: 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace * nTime
    title: str, set to None

    Returns
    -------
    fig, ax : Matplotlib figure and axes objects
        The figure and axes of the plot
    cbar : Matplotlib color bar object
        The color bar on the final plot
    '''
    # Check our kwargs for defaults
    # Set defaults cmap to hot
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Create and configure figure and axes:
    fig, ax = plt.subplots(1, 1, figsize = (8,8))

    # Add contour to axes
    contour = plt.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Add labels to stuff
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    fig.tight_layout()
    plt.show()
    return fig, ax, cbar