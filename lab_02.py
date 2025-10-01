#!/usr/bin/env python3
'''
In this lab, we will be discussing different solutions to approximate predator-prey and competition models.
We will use Euler’s method and an 8th order Range-Kutta equation to approximate these models with the highest
accuracy possible. In this lab, we will be exploring how 2 different species interact with one another, 
and how competition and predator-prey relationships work through models. 
Throughout the course of this lab report, we will be attempting to create conclusions about different hypotheses. 
Here they are;
1. How are the behavior of the Lotka-Volterra equations dependent on the initial conditions and four coefficients?

2. How does the performance of the Euler method solver compare to the 8th-order DOP853 method for both sets of equations?

3. How do the initial conditions and coefficient values affect the final result and general behavior of the two species?

4. How do the initial conditions and coefficient values affect the final result and general behavior of the two species? 
   What new information did you gain from the phase diagrams plotted?

--------------------------------------------------------
TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

1. To verify my code works for questions 1 and 2, and to replicate the graph in Dan's lab report, run 
    'verification_plot()'
2. To get phase diagrams for question 3, run
    'phase_diagram()'

This should be all. If you'd like to change any default values, you're more than welcome to by changing the values
each of the 'euler_solve' and 'solve_rk8" default values in my 'verification_plot' or 'phase_diagram' functions. 
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time
    N : two-element list
            The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    # Returning each time derivative for N1 and N2
    return dN1dt, dN2dt

def dNdt_pred_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    Parameters
    ----------
    t : float
        The current time
    N : two-element list
            The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]

    #Returning each time derivative for N1 and N2
    return dN1dt, dN2dt

def euler_solve(func, N1_init = 0.5, N2_init = 0.5, dT = 0.1, 
                t_final = 100.0):
    
    '''
    Solve an ordinary diffeq using Euler's Method.

    Parameters
    ----------
    func: function, defaults to dNdt_comp()
        A function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init: float, defaults to 0.5
        Our initial population for species 1
    N2_init: float, defaults to 0.5
        Our initial population for species 2
    dT: float, defaults to 0.1
        Each individual time step
    t_final: float, defualts to 100
        The time where we end our simulation

    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    
    time = np.arange(0, t_final, dT)
    N1 = np.zeros(time.size) 
    N1[0] = N1_init
    N2 = np.zeros(time.size) 
    N2[0] = N2_init

    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]])
        N1[i] = N1[i - 1] + dT * dN1
        N2[i] = N2[i - 1] + dT * dN2

    return N1, N2, time 


def solve_rk8(func, N1_init = 0.5, N2_init = 0.5, dT = 10, t_final = 100.0,
              a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp

    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
                       args=[a, b, c, d], method='DOP853', 
                       max_step=dT)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # Return values to caller.
    return N1, N2, time

def verification_plot():
    fig, axes = plt.subplots(1,2, figsize=[12.5, 10])
    N1, N2, time = euler_solve(func = dNdt_comp, N1_init = 0.3, 
                               N2_init = 0.6, dT = 1)
    axes[0].plot(time, N1, label = 'N1')
    axes[0].plot(time, N2, label = 'N2')
    N1_rk8, N2_rk8, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.3,
                                         N2_init = 0.6, dT = 1)
    axes[0].plot(time_rk8, N1_rk8, '--', label = 'N1_rk8')
    axes[0].plot(time_rk8, N2_rk8, '--', label = 'N2_rk8')
    axes[0].legend()
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Population/Carrying Capacity')
    axes[0].set_title('Lotka-Volterra Competition Model')

    N1_0, N2_0, time_0 = euler_solve(func = dNdt_pred_prey, N1_init = 0.3,
                                     N2_init = 0.6, dT = 0.05)
    axes[1].plot(time_0, N1_0, label = 'N1')
    axes[1].plot(time_0, N2_0, label = 'N2')
    N1_rk8_0, N2_rk8_0, time_rk8_0 = solve_rk8(func = dNdt_pred_prey, 
                                               N1_init = 0.3, N2_init = 0.6, 
                                               dT = 0.05)
    axes[1].plot(time_rk8_0, N1_rk8_0, '--', label = 'N1_rk8')
    axes[1].plot(time_rk8_0, N2_rk8_0, '--', label = 'N2_rk8')
    axes[1].legend()
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Population/Carrying Capacity')
    axes[1].set_title('Lotka-Volterra Predator-Prey Model') 
    fig.tight_layout()
    plt.show()

def phase_diagram():
    fig, ax = plt.subplots(1, 1, figsize=[10,8])
    N1, N2, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3, N2_init = 0.6, 
                         dT = 0.05)
    N1_rk8, N2_rk8, time_rk8 = solve_rk8(func = dNdt_pred_prey, N1_init = 0.3, 
                                         N2_init = 0.6, dT = 0.05)
    ax.plot(N1, N2, label = 'Predator vs Prey')
    ax.plot(N1_rk8, N2_rk8, '--', label = 'Predator vs Prey RK8')
    ax.legend()
    ax.set_xlabel('Prey Population')
    ax.set_ylabel('Predator Population')
    ax.set_title('Predator vs Prey Model')
    fig.tight_layout()
    plt.show()