#!/usr/bin/env python3
''' 
Doc String - Description of what we are doing
'''
'''
TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

1. Shown in Code

2. Run "n_layer_atmosphere()" given any N layers. 

3. Run question3() to get graph
3_2. Run question3_2() to get graph

4. Run "n_layer_atmosphere_Q4(51)"

5. Run "question5(5)"
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Physical Constants
sigma = 5.67e-8 # Units: W/m^2/k^-4

def n_layer_atmosphere(nlayers, epsilon = 1., albedo = 0.33, s0 = 1350.):
    
    '''
    Parameters
    ----------
    nlayers: Floating point
         A value based on how atmospheres is given 
    epsilon: Floating Point, defaults to 1
         A value that indicates our emissivity throughout atmospheres
    albedo: Floating Point, defaults to 0.33
         A value based on reflectivity of Earth's surface.
    s0: Floating Point, defaults to 1350
         The total solar irradiance that the Earth's surface absorbs, in W/m^2.  

    Returns
    -------
    final_Temp: Array
         Temperature starting at the surface, and multiple temperatures are given
         based on N layers of atmospheres. 
    '''

    # Create an array of cofficients an N+1 x N+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:           # Handle Diagonal
                if j > 0:
                    A[i, j] = -2 # Diagonals
                else: 
                    A[i, j] = -1 # Diagonals (First row is special)
            else:
                A[i, j] = (epsilon ** (i > 0)) * ((1 - epsilon) ** (np.abs(j - i) - 1))
    print(A)

    # Making sure the surface temps are different than the rest of the atmospheres
    b[0] = -0.25 * s0 * (1 - albedo) 

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!   

    print(fluxes) 

    # Turn fluxes into temperature
    final_Temp = (fluxes / epsilon / sigma) ** (1/4)
    # Epsilon has no effect on the surface
    final_Temp[0] = ((fluxes[0]) / (sigma)) ** (1/4)

    return final_Temp

def question3(): 
    '''
    Plotting Surface Temperature vs Emissivity in a function for clarity when grading. 
    '''
    temps_array = []
    epsilon_range = np.linspace(0.01, 1, 100)
    for i in epsilon_range:
        plotted_temps = n_layer_atmosphere(nlayers = 1, epsilon = i)
        temps_array.append(plotted_temps[0])

    plt.figure(figsize = (8 , 5))
    plt.plot(epsilon_range, temps_array, color = 'darkgreen')
    plt.axhline(288, color = 'red', linestyle = '--', label = 'Observed Earth Temp (288 K)')
    plt.xlabel('Atmospheric Emissivity')
    plt.ylabel('Surface Temperature')
    plt.title('Surface Temperature vs. Emissivity (Single Layer Atmosphere)')
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def question3_2(nlayers = 5, epsilon = 0.255, albedo = 0.33, s0 = 1350):

    '''
    Plotting Temperature vs Altitude in a function for clarity when grading. 
    '''

    fig, ax = plt.subplots(figsize = (10,8))
    altitude = np.arange(0, (nlayers + 1) * 10, 10)
    altitude_temps = n_layer_atmosphere(nlayers, epsilon)
    ax.plot(altitude_temps, altitude, color = 'red')
    ax.set_title('Temperature vs Altitude for a 5 Layer Atmosphere when epsilon=0.255')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)
    plt.show()

def n_layer_atmosphere_Q4(nlayers, epsilon = 1., albedo = 0.6, s0 = 2600.):
    
    '''
    Parameters
    ----------
    nlayers: Floating point
         A value based on how atmospheres is given 
    epsilon: Floating Point, defaults to 1
         A value that indicates our emissivity throughout atmospheres
    albedo: Floating Point, defaults to 0.6
         A value based on reflectivity of Earth's surface.
    s0: Floating Point, defaults to 2600
         The total solar irradiance that the Earth's surface absorbs, in W/m^2.  

    Returns
    -------
    final_Temp: Array
         Temperature starting at the surface, and multiple temperatures are given
         based on N layers of atmospheres. 
    '''

    # Create an array of cofficients an N+1 x N+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:           # Handle Diagonal
                if j > 0:
                    A[i, j] = -2 # Diagonals
                else: 
                    A[i, j] = -1 # Diagonals (First row is special)
            else:
                A[i, j] = (epsilon ** (i > 0)) * ((1 - epsilon) ** (np.abs(j - i) - 1))
    print(A)

    b[0] = -0.25 * s0 * (1 - albedo) 

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!   

    print(fluxes) 

    # Turn fluxes into temperature
    final_Temp = (fluxes / epsilon / sigma) ** (1/4)
    final_Temp[0] = ((fluxes[0]) / (sigma)) ** (1/4)

    return final_Temp

def n_layer_atmosphere_Q5(nlayers = 5, epsilon = 0.5, albedo = 0, s0 = 1350.):
    
    '''
    Parameters
    ----------
    nlayers: Floating point, defaults to 5
         A value based on how atmospheres is given 
    epsilon: Floating Point, defaults to 0.5
         A value that indicates our emissivity throughout atmospheres
    albedo: Floating Point, defaults to 0
         A value based on reflectivity of Earth's surface.
    s0: Floating Point, defaults to 1350
         The total solar irradiance that the Earth's surface absorbs, in W/m^2.  

    Returns
    -------
    final_Temp: Array
         Temperature starting at the surface, and multiple temperatures are given
         based on N layers of atmospheres. 
    '''

    # Create an array of cofficients an N+1 x N+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:           # Handle Diagonal
                if j > 0:
                    A[i, j] = -2 # Diagonals
                else: 
                    A[i, j] = -1 # Diagonals (First row is special)
            else:
                A[i, j] = (epsilon ** (i > 0)) * ((1 - epsilon) ** (np.abs(j - i) - 1))
    print(A)

    b[0] = 0
    # Changing top layer to absorb all incoming solar irradiance. 
    b[-1] = -0.25 * s0 * (1 - albedo)

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!   

    print(fluxes) 

    # Turn fluxes into temperature
    final_Temp = (fluxes / epsilon / sigma) ** (1/4)
    final_Temp[0] = ((fluxes[0]) / (sigma)) ** (1/4)

    return final_Temp

def question5(nlayers, epsilon = 0.5, albedo = 0.33, s0 = 1350):
    '''
    Plotting Temperature vs Altitude in a function for clarity when grading. 
    '''
    fig, ax = plt.subplots(figsize=(10,8))

    nuke = np.arange(0, (nlayers + 1) * 10, 10)
    nuke_temps = n_layer_atmosphere_Q5()
    ax.plot(nuke_temps, nuke, color = 'red')
    ax.set_title('Temperature vs Altitude for a 5 layer Atmosphere given an emissivity')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Altitude (km)')
    ax.grid(True)
    plt.show()
