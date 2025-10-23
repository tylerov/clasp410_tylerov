#!/usr/bin/env python3
'''
Throughout the course of this report, the idea of forward difference solvers
and how they can be applied to permafrost will be examined. Our example city 
that will be used to examine and model their permafrost behavior will be from 
Kangerlussuaq, Greenland. Throughout, we will see how different initial average 
temperatures at the surface will affect the permafrost underground and how 
different time steps, Δx, and rates of diffusion affect how stable our model is. 

Hypotheses:
1) Based on typical thermal diffusivity values for permafrost, what is the depth 
   of the active layer and permafrost layer?
2) With an initial condition of 0 degrees Celsius, how long does it take for the 
   ground to reach a steady state?
3) How does adding a 0.5, 1, and 3 degrees Celsius shift in Kangerlussuaq’s climate 
   curve affect the depth and thickness of the active and permafrost layers?

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

1. To validate my code for problem 1; 
    a) run lab_03.py
    b) type into terminal:
        i) t, x, U = solve_heat()
        ii) Array printed should be same as the one in Dan's Report
2. To get figures 1 and 2;
    a) run lab_03.py
    b) type into terminal:
        i) t, x, U = solve_heat_Kanger()
        ii) plot_heatsolve(t, x, U, vmin = -25, vmax = 25)
3. To reproduce figure 3;
    a) run lab_03.py
    b) type into terminal:
        i) t, x, U = solve_heat_Kanger(upperbound = 5.5)
        ii) plot_heatsolve(t, x, U, vmin = -25, vmax = 25)
4. To reproduce figure 4;
    a) run lab_03.py
    b) type into terminal:
        i) t, x, U = solve_heat_Kanger(upperbound = 6)
        ii) plot_heatsolve(t, x, U, vmin = -25, vmax = 25)
5. To reproduce figure 5;
    a) run lab_03.py
    b) type into terminal:
        i) t, x, U = solve_heat_Kanger(upperbound = 8)
        ii) plot_heatsolve(t, x, U, vmin = -25, vmax = 25)

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

# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
                     10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()

    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

def solve_heat(x_stop = 1, t_stop = 0.2, dx = 0.2, dt = 0.02, c2 = 1,
               lowerbound = 0, upperbound = 0):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    x_stop: Floating point, defaults to 1
        Telling our forward difference solver when to stop locationally
    t_stop: Floating point, defaults to 0.2
        Telling our forward difference solver when to stop given a t
    dx: Floating point, defaults to 0.2
        Our spatial step
    dt: Floating point, defaults to 0.02
        Our time step
    c2: Floating point, defaults to 1
        c^2, the square of the diffusion coefficient
    lowerbound: Float, defaults to 0
        Setting boundary conditions
    upperbound: Float, defaults to 0
        Setting boundary conditions

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

    # Get grid sizes (+ 1 to include "0" as well)
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
        # Apply Neumann BC's
        if lowerbound is None:
            U[0, j + 1] = U[1, j + 1]
        elif callable(lowerbound):
            U[0, j + 1] = lowerbound(t[j + 1])
        else: 
            U[0, j + 1] = lowerbound
        if upperbound is None:
            U[-1, j + 1] = U[-2, j + 1]
        elif callable(upperbound):
            U[-1, j + 1] = upperbound(t[j + 1])
        else:
            U[-1, j + 1] = upperbound
    
    # Print [U] to verify code
    print([U])
    # Return our pretty solution to the caller
    return t, x, U

def solve_heat_Kanger(x_stop = 100, t_stop = 36500, dx = 0.5, dt = 5, c2 = 0.25,
                      lowerbound = temp_kanger, upperbound = 5):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    x_stop: Floating point, defaults to 1
        Telling our forward difference solver when to stop locationally
    t_stop: Floating point, defaults to 0.2
        Telling our forward difference solver when to stop given a t
    dx: Floating point, defaults to 0.2
        Our spatial step
    dt: Floating point, defaults to 0.02
        Our time step
    c2: Floating point, defaults to 1
        c^2, the square of the diffusion coefficient
    lowerbound: Float, defaults to 0
        Setting boundary conditions
    upperbound: Float, defaults to 0
        Setting boundary conditions

    Returns
    -------
    x, t: 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace * nTime
    '''

    # C unit conversion from mm^2/s to m^2/day
    c2_updated = c2 * 0.0864

    # Check our stability criterion:
    dt_max = dx**2 / (2*c2_updated)
    if dt > dt_max:
       raise ValueError(f'DANGER: dt={dt} > dt_max = {dt_max}.')

    # Get grid sizes (+ 1 to include "0" as well)
    N = int(t_stop / dt) + 1
    M = int(x_stop / dx) + 1

    # Set up space and time grid
    t = np.linspace(0, t_stop, N)
    x = np.linspace(0, x_stop, M)

    # Create solution matrix
    U = np.zeros([M, N])

    #Get our "r" coefficient
    r = c2_updated * (dt/dx**2)

    # Solve our equation 
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])
        # Apply Neumann BC's
        if lowerbound is None:
            U[0, j + 1] = U[1, j + 1]
        elif callable(lowerbound):
            U[0, j + 1] = lowerbound(t[j + 1])
        else: 
            U[0, j + 1] = lowerbound
        if upperbound is None:
            U[-1, j + 1] = U[-2, j + 1]
        elif callable(upperbound):
            U[-1, j + 1] = upperbound(t[j + 1])
        else:
            U[-1, j + 1] = upperbound
    
    # Print [U] to verify code
    print([U])

    # Return our pretty solution to the caller
    return t, x, U

def plot_heatsolve(t, x, U, dt = 5, **kwargs):
    '''
    Plot the 2D solution for the 'solve_heat' function.
    Plot the temperature profile for Kangerlussuaq
    Parameters
    ----------
    x, t: 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace * nTime

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
        kwargs['cmap'] = 'seismic'

    # Create and configure figure and axes:
    fig, ax = plt.subplots(1, 1, figsize = (8,8))

    # Add contour to axes
    contour = plt.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Add labels to stuff
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($days$)')
    ax.set_ylabel('Depth ($m$)')
    ax.set_title('Ground Temp of Kangerlussuaq, Greenland')
    plt.gca().invert_yaxis()

    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = U[:, loc:].min(axis=1)
    summer = U[:, loc:].max(axis=1)

    #Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))

    # Plot the winter and summer temperatures
    ax2.plot(winter, x, label='Winter')
    ax2.plot(summer, x, '--', label='Summer')
    ax2.set_xlabel(r'Temperature ($^{\circ}C$)')
    ax2.set_ylabel('Depth ($m$)')
    ax2.set_title('Temperature Profile')

    # Invert y axis to get 0m at the top
    plt.gca().invert_yaxis()

    # Plot this black line to give us permafrost layer
    ax2.axvline(0, c = 'black', ls = '--')
    
    fig.tight_layout()
    plt.show()
    return fig, ax, cbar
