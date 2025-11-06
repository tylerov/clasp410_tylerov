#!/usr/bin/env python3
'''
Can solar forcing be accounted for climate change? In this code, 
we will show measured temperature as their anomalies from 1900 to 
2000, and then show expected temperatures given our T_E equation, 
and then compare and make conclusions on our hypothesis. 
'''
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def solve_temp(albedo, solar_irr, sigma = 5.67e-8):

    ''' This function returns the temperature of the atmosphere given
    an albedo and a value for the solar irradiance. 

    Parameters
    ----------
    albedo: Floating point
        A value based on how much albedo the surface is given
    solar_irr: Floating Point
        A value of how much solar irradiance is given
    sigma: Floating Point, defaults to 5.67e-8
        Stefan-Boltzmann constant (5.67 * 10^-8) 

    Returns
    -------
    T_earth: Floating Point
        Temperature of Earth's surface given a certain albedo and a 
        certain solar irradiance
    '''

    T_earth = (((1 - albedo) * solar_irr) / (2 * sigma)) ** (1/4)

    return T_earth

def verify_code():
    ''' 
    Verify that our implementation is correct
    '''
    T_real = 288
    T_earth_real = solve_temp(0.33, 1350)
    print('Target solution is: ', T_real)
    print('Numerical solution is: ', T_earth_real)
    print('Difference in solutions is: ', T_real - T_earth_real)

# Years in an array
year = np.array([1900, 1950, 2000])
# Measured Solar forcing in an array
s0 = np.array([1365, 1366.5, 1368])
# Measured temperature anomaly in an array
t_anom = np.array([-0.4, 0, 0.4])

# Equation to find expected temperature given a solar forcing
T_solar = solve_temp(0.33, s0)
# Measured temperature given a solar forcing
T_measured = T_solar[1] + t_anom

# Plot that shows us trends in measured vs. expected Temperatures
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(year, T_measured, label = 'Measured Temperature')
ax.plot(year, T_solar, label = 'Solar forcing predicted temperature')
ax.legend(loc = 'best')
ax.set_title('Can Solar Forcing Account for Climate Change?')
ax.set_xlabel('Year')
ax.set_ylabel('Temperature (K)')
fig.tight_layout()