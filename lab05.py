#!/usr/bin/env python3

'''
Throughout the course of this lab report, we will be uncovering the truth on if the entirety
of earth was covered in ice 650-700 million years ago. There have been many hypotheses about 
whether or not it would even be possible for earth to become a complete “snowball.” This 
hypothesis is hard to believe because this would mean the entire earth is glaciated, which would 
mean there is no exposed liquid water or land. Getting out of this state has been said to be 
“impossible” by many scientists. Throughout this report, we will discuss whether or not the 
formation of snowball Earth is possible and stable. 

HYPOTHESES
----------
1. Validate model with Figure 1 in Dan's lab report
2. Reproduce warm-Earth equilibrium. Report findings. 
3. What is the equilibrium solution for a “hot” Earth?
4. What is the equilibrium solution for a “cold” Earth?
5. What is the equilibrium solution for a “flash freeze” Earth?
6. How do the three equilibrium solutions above compare to one another? What does it tell us about 
   the stability of snowfall vs. warm Earth solutions?
7. Does the snowball Earth hypothesis represent an equilibrium solution that is stable? What does 
   the plot tell us about the stability of the different equilibria? 
8. Given the large range of γ and considering the historical variation of So, is the idea of Snowball 
   Earth a valid hypothesis?

TO REPRODUCE THE PLOTS IN MY REPORT, DO THIS:
---------------------------------------------
Figure 1: 
    1. Run lab05.py
    2. Type 'problem1()' into terminal
Figure 2: 
    1. Run lab05.py
    2. Type 'problem2()' into terminal
Figure 3:
    1. Run lab05.py
    2. Type 'problem3()' into terminal
Figure 4:
    1. run lab05.py
    2. Type 'problem4()' into terminal

'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

radearth = 6357000.     # Earth radius in meters
mxdlyr = 50.            # Depth of mixed layer (m)
sigma = 5.67e-8         # Steffan Boltzman Constant
C = 4.2e6               # Heat capacity of water
rho = 1020              # Density of seat water (kg/m^3)

def gen_grid(npoints=18):
    '''
    Create an evenly spaced latitudinal grid wit 'npoints' cell centers.
    Grid will always run from zero to 180 as the edges of the grid. This
    means that the first grid point will be 'dLat/2' from 0 degrees and the 
    last point will be '180 - dLat/2'. 

    Parameters:
    -----------
    npoints: int, defaults to 18
        Number of grid points to create
    
    Returns:
    --------
    dLat: float
        Grid spacing in latitude (degrees)
    lats: numpy array
        Locations of all grid cell centers.
    '''
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.

    return dlat, lats

def temp_warm(lats_in):
    '''
    Create a temperature profile for real life "warm" earth.
    Parameters
    ----------
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
    Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])

    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_grid(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def temp_hot(lats_in):
    '''
    Create a temperature profile for possible "hot" earth.
    Parameters
    ----------
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
    Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_hot = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                       60, 60, 60, 60, 60, 60, 60, 60])

    # Get base grid:
    npoints = T_hot.size
    dlat, lats = gen_grid(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_hot, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def temp_cold(lats_in):
    '''
    Create a temperature profile for possible "cold" earth.
    Parameters
    ----------
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
    Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_cold = np.array([-60, -60, -60, -60, -60, -60, -60, -60, -60, -60,
                       -60, -60, -60, -60, -60, -60, -60, -60])

    # Get base grid:
    npoints = T_cold.size
    dlat, lats = gen_grid(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_cold, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.
    Parameters
    ----------
    S0 : float
    Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
    Latitudes to output insolation. Following the grid standards set in
    the diffusion program, polar angle is defined from the south pole.
    In other words, 0 is the south pole, 180 the north.
    Returns
    -------
    insolation : numpy array
    Insolation returned over the input latitudes.
    '''
    # Constants:
    max_tilt = 23.5 # tilt of earth in degrees
    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)
    # Daily rotation of earth reduces solar constant by distributing the sun
    # energy all along a zonal band
    dlong = 0.01 # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)
    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]
    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.
    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365
    return insolation

def snowball_earth(nlat = 18, tfinal = 10000., dt = 1.0, lam = 100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr = False,
                   apply_insol = False, solar = 1370., albice = 0.6, albgnd = 0.3):
    ''' 
    Solve the snowball Earth problem. 

    Parameters:
    -----------
    nlat: int, defaults to 18
        Number of latitude cells.
    tfinal: int or float, defaults to 10,000
        Time length of simulation in years
    dt: int or float, defaults to 1.0
        Size of time step in years
    lam: float, defaults to 100
        Set ocean diffusivity
    emiss: float, defaults to 1.0
        Set emissivity of Earth
    init_cond: function, float, or array
        Set the initial condition of the smulation. If a function is given,
        it must take latitudes as inputs and return temperature as a function
        of lat. Otherwise, the given values are used as is. 
    apply_spherecorr: Bool, defaults to False
        Apply spherical correction term
    apply_insol: Bool, defaults to False
        Apply insolation term
    solar: float, defaults to 1370
        Set level of solar forcing in W/m^2
    albice: float, defaults to 0.6
        Albedo of ice
    albgnd: float, defaults to 0.3
        Albedo of ground
    
    Returns:
    ---------
    Lats: numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        and 180 is north.
    Temp: Numpy array
        Temperature as a function of latitude
    '''
    # Set up grid:
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units
    dy = np.pi * radearth / nlat

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set time step to seconds:
    dt = dt * 365 * 24 * 3600
    print(lats)
    # Create insolation:
    insol = insolation(solar, lats)
    print(insol)
    
    # Create Temp array; set initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond
    
    # Create our k matrix
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary Conditions
    K[0, 1], K[-1, -2] = 2, 2

    # Units!
    K *= 1/dy**2

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array
    Axz = np.pi * ((radearth + 50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)

    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Create and invert our L matrix
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10 # Sea water freezes at ten below
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # Solve!
    for istep in range(nsteps):
        # Update albedo:
        loc_ice = Temp <= -10 # Sea water freezes at ten below
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            spherecorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp)*dAxz
        else:
            spherecorr = 0

        # Apply radiative insolation term
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho * C * mxdlyr)
        
        # Advance Solution
        Temp = np.matmul(Linv, Temp + spherecorr)

    return lats, Temp

def snowball_earth_dynamic(nlat = 18, tfinal = 10000., dt = 1.0, lam = 100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr = False,
                   apply_insol = False, solar = 1370., albice = 0.6, albgnd = 0.3):
    ''' 
    Solve the snowball Earth problem with a dynamic albedo 

    Parameters:
    -----------
    nlat: int, defaults to 18
        Number of latitude cells.
    tfinal: int or float, defaults to 10,000
        Time length of simulation in years
    dt: int or float, defaults to 1.0
        Size of time step in years
    lam: float, defaults to 100
        Set ocean diffusivity
    emiss: float, defaults to 1.0
        Set emissivity of Earth
    init_cond: function, float, or array
        Set the initial condition of the smulation. If a function is given,
        it must take latitudes as inputs and return temperature as a function
        of lat. Otherwise, the given values are used as is. 
    apply_spherecorr: Bool, defaults to False
        Apply spherical correction term
    apply_insol: Bool, defaults to False
        Apply insolation term
    solar: float, defaults to 1370
        Set level of solar forcing in W/m^2
    albice: float, defaults to 0.6
        Albedo of ice
    albgnd: float, defaults to 0.3
        Albedo of ground
    
    Returns:
    ---------
    Lats: numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        and 180 is north.
    Temp: Numpy array
        Temperature as a function of latitude
    '''
    # Parameters for smooth dynamic albedo
    Tcrit = -10.0    # temperature where ice/ground transition occurs
    dT = 3.0         # width of transition zone
    
    # Smooth fractional ice function
    def ice_fraction_from_temp(T):
        return 1.0 / (1.0 + np.exp((T - Tcrit) / dT))

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    dy = np.pi * radearth / nlat

    nsteps = int(tfinal / dt)
    dt = dt * 365 * 24 * 3600   # convert years to seconds

    # Insolation
    insol = insolation(solar, lats)

    # Initial temperature
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp = np.array(init_cond, dtype=float)

    # Diffusion matrix K
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    K[0,1] = K[-1,-2] = 2
    K *= 1/dy**2

    # First-derivative operator B
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0,:] = B[-1,:] = 0

    # Cell area and derivative
    Axz = np.pi * ((radearth + 50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz)

    # Invert (I - dt*K*lam)
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # INITIAL dynamic albedo
    ice_frac = ice_fraction_from_temp(Temp)
    albedo = albgnd * (1 - ice_frac) + albice * ice_frac

    # ---- TIME STEPPING ----
    for istep in range(nsteps):

        # Update fractional ice and albedo (MINIMAL CHANGE)
        ice_frac = ice_fraction_from_temp(Temp)
        albedo = albgnd * (1 - ice_frac) + albice * ice_frac

        # Spherical correction
        if apply_spherecorr:
            spherecorr = (lam * dt) / (4 * Axz * dy**2) * np.matmul(B, Temp) * dAxz
        else:
            spherecorr = 0

        # Radiative balance (uses the new dynamic albedo)
        if apply_insol:
            radiative = (1 - albedo) * insol - emiss * sigma * (Temp + 273.15)**4
            Temp += dt * radiative / (rho * C * mxdlyr)

        # Diffusion step
        Temp = np.matmul(Linv, Temp + spherecorr)

    return lats, Temp

def snowball_earth_gamma(nlat = 36, tfinal = 10000., dt = 1.0, lam = 100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr = False, gamma = 1.0,
                   apply_insol = False, solar = 1370., albice = 0.6, albgnd = 0.3):
    ''' 
    Solve the snowball Earth problem with a dynamic albedo, and add our solar multiplier
    (gamma) 

    Parameters:
    -----------
    nlat: int, defaults to 18
        Number of latitude cells.
    tfinal: int or float, defaults to 10,000
        Time length of simulation in years
    dt: int or float, defaults to 1.0
        Size of time step in years
    lam: float, defaults to 100
        Set ocean diffusivity
    emiss: float, defaults to 1.0
        Set emissivity of Earth
    init_cond: function, float, or array
        Set the initial condition of the smulation. If a function is given,
        it must take latitudes as inputs and return temperature as a function
        of lat. Otherwise, the given values are used as is. 
    apply_spherecorr: Bool, defaults to False
        Apply spherical correction term
    gamma: float, defaults to 1
        Size of solar multiplier
    apply_insol: Bool, defaults to False
        Apply insolation term
    solar: float, defaults to 1370
        Set level of solar forcing in W/m^2
    albice: float, defaults to 0.6
        Albedo of ice
    albgnd: float, defaults to 0.3
        Albedo of ground
    
    Returns:
    ---------
    Lats: numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        and 180 is north.
    Temp: Numpy array
        Temperature as a function of latitude
    '''
    # Parameters for smooth dynamic albedo
    Tcrit = -10.0    # temperature where ice/ground transition occurs
    dT = 3.0         # width of transition zone
    
    # Smooth fractional ice function
    def ice_fraction_from_temp(T):
        return 1.0 / (1.0 + np.exp((T - Tcrit) / dT))

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    dy = np.pi * radearth / nlat

    nsteps = int(tfinal / dt)
    dt = dt * 365 * 24 * 3600 # convert years to seconds

    # Insolation (featuring solar multiplier)
    insol = gamma * insolation(solar, lats)

    # Initial temperature
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp = np.array(init_cond, dtype=float)

    # Diffusion matrix K
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    K[0,1] = K[-1,-2] = 2
    K *= 1/dy**2

    # First-derivative operator B
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0,:] = B[-1,:] = 0

    # Cell area and derivative
    Axz = np.pi * ((radearth + 50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz)

    # Invert (I - dt*K*lam)
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # INITIAL dynamic albedo
    ice_frac = ice_fraction_from_temp(Temp)
    albedo = albgnd * (1 - ice_frac) + albice * ice_frac

    # ---- TIME STEPPING ----
    for istep in range(nsteps):

        # Update fractional ice and albedo (MINIMAL CHANGE)
        ice_frac = ice_fraction_from_temp(Temp)
        albedo = albgnd * (1 - ice_frac) + albice * ice_frac

        # Spherical correction
        if apply_spherecorr:
            spherecorr = (lam * dt) / (4 * Axz * dy**2) * np.matmul(B, Temp) * dAxz
        else:
            spherecorr = 0

        # Radiative balance (uses the new dynamic albedo)
        if apply_insol:
            radiative = (1 - albedo) * insol - emiss * sigma * (Temp + 273.15)**4
            Temp += dt * radiative / (rho * C * mxdlyr)

        # Diffusion step
        Temp = np.matmul(Linv, Temp + spherecorr)

    return lats, Temp

def problem1():
    '''
    Create solution figure for problem 1. Also validate our code 
    qualitatively.
    '''

    # Get warm Earth initial condition
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms
    lats, temp_diff = snowball_earth()
    lats, temp_sphere = snowball_earth(apply_spherecorr=True)
    lats, temp_all = snowball_earth(apply_spherecorr = True, apply_insol = True,   
                                    albice=0.3)

    # Create a fancy plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats - 90, temp_init, label = 'Initial condition')
    ax.plot(lats - 90, temp_diff, label = 'Diffusion only')
    ax.plot(lats - 90, temp_sphere, label = 'Diffusion + Spherical Corr.')
    ax.plot(lats - 90, temp_all, label = 'Diffusion + Spherical Corr. + Radiative')
    # Customize 
    ax.set_title('Solution after 10,000 years')
    ax.set_ylabel(r'Temp (${\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc = 'best')
    plt.show()

def problem2():
    '''
    Create solution figure for problem 2. Also validate our code 
    qualitatively.
    '''

    # Get warm Earth initial condition
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms
    lats, temp_diff = snowball_earth()
    lats, temp_sphere = snowball_earth(apply_spherecorr=True)
    lats, temp_all = snowball_earth(apply_spherecorr = True, apply_insol = True,   
                                    albice=0.3, lam = 55., emiss = 0.72)

    # Create a fancy plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats - 90, temp_init, label = 'Initial condition')
    ax.plot(lats - 90, temp_diff, label = 'Diffusion only')
    ax.plot(lats - 90, temp_sphere, label = 'Diffusion + Spherical Corr.')
    ax.plot(lats - 90, temp_all, label = 'Diffusion + Spherical Corr. + Radiative')
    # Customize 
    ax.set_title('Solution after 10,000 years')
    ax.set_ylabel(r'Temp (${\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc = 'best')
    plt.show()

def problem3():
    '''
    Create solution figure for problem 3. 
    '''
    # dynamic-albedo experiments for hot and cold
    lats, T_hot = snowball_earth_dynamic(init_cond=temp_hot,
                                  apply_insol=True,  lam = 55., emiss = 0.72, 
                                  apply_spherecorr=True)

    lats, T_cold = snowball_earth_dynamic(init_cond=temp_cold,
                                  apply_insol=True,  lam = 55., emiss = 0.72,
                                  apply_spherecorr=True)

    # flash freeze: constant albedo = 0.6
    lats, T_flash = snowball_earth_dynamic(albice=0.6, albgnd=0.6,  # forces constant albedo
                                  apply_insol=True, lam = 55., emiss = 0.72,
                                  apply_spherecorr=True)

    # plot our data
    plt.figure(figsize=(10,5))
    temp_warm_curve = temp_warm(lats)
    plt.plot(lats-90, temp_warm_curve, label="Warm curve (initial for flash-freeze)")
    plt.plot(lats-90, T_hot,   label="Hot start (dynamic albedo)")
    plt.plot(lats-90, T_cold,  label="Cold start (dynamic albedo)")
    plt.plot(lats-90, T_flash, label="Flash freeze (constant albedo=0.6)")
    plt.xlabel("Latitude (deg)")
    plt.ylabel("Equilibrium Temperature (°C)")
    plt.title("Dynamic Albedo Snowball Earth Equilibria")
    plt.legend()
    plt.grid()
    plt.show()

    # Means for precise values for my report
    print("Hot-start equilibrium mean temp:", np.mean(T_hot))
    print("Cold-start equilibrium mean temp:", np.mean(T_cold))
    print("Flash-freeze equilibrium mean temp:", np.mean(T_flash))

def problem4():
    '''
    Create solution figure for problem 4. 
    '''
    # Make array for steps in Gamma
    gammas_forward = np.arange(0.40, 1.401, 0.05)
    gammas_backward = np.arange(1.35, 0.39, -0.05)
    
    # Define Arrays
    mean_forward = []
    mean_backward = []

    # Set initial temperature
    init_temp = np.ones(36) * (-60)

    # Loop for forward gamma 
    for g in gammas_forward:
        lats, T = snowball_earth_gamma(init_cond=lambda x: init_temp.copy(), gamma=g,
                                       apply_insol = True, tfinal = 200, dt = 1/120,
                                       lam = 55., emiss = 0.72)
        mean_forward.append(np.mean(T))
        init_temp = T.copy()

    init_temp = T.copy()
     
    # Loop for backward gamma
    for g in gammas_backward:
        lats, T = snowball_earth_gamma(init_cond=lambda x: init_temp, gamma=g, 
                                       apply_insol = True, tfinal = 200, dt = 1/120,
                                       lam = 55., emiss = 0.72)
        mean_backward.append(np.mean(T))
        init_temp = T.copy()

    # Plot our data
    plt.figure(figsize=(10,5))
    plt.plot(gammas_forward, mean_forward, 'o-', label='Forward sweep')
    plt.plot(gammas_backward, mean_backward, 'o-', label='Backward sweep')
    plt.xlabel("Solar multiplier γ")
    plt.ylabel("Global mean temperature (°C)")
    plt.title("Hysteresis in Snowball Earth vs Solar Forcing")
    plt.grid()
    plt.legend()
    plt.show()

    return gammas_forward, mean_forward, gammas_backward, mean_backward

def test_functions():
    '''
    Test our functions
    '''
    print('Test gen_grid')
    print('For npoints = 5:')
    dlat_correct, lats_correct = (36.0, np.array([18., 54., 90., 126., 162.,]))
    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else: 
        print('\tFAILED')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {gen_grid(5)}")


