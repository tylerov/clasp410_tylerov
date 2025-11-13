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

Firstly, run my code, then
1. To reproduce figure 1, and to replicate the graph in Dan's lab report, run 'verification_plot()'
2. To reproduce figure 2 in my report, run 'error_plots_comp_euler()'
3. To reproduce figure 3 in my report, run 'error_plots_comp_rk8()'
4. To reproduce figure 4 in my report, run 'error_plots_pred_prey_euler()'
5. To reproduce figure 5 in my report, run 'error_plots_pred_prey_rk8()'
6. To reproduce figure 6 in my report, run 'varying_initial_conditions_comp()'
7. To reproduce figure 7 in my report, run 'varying_initial_conditions_pred_prey()'
8. To reproduce figure 8 in my report, run 'phase_diagram()'

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
    '''
    This functions simply returns a plot verifying my data with the plot in
    Dan's lab report.
    '''
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

def error_plots_comp_euler():
    '''
    This functions simply returns a plot of errors using Euler's method,
    for the competition model
    '''
    fig, ax = plt.subplots(1,1, figsize=[12.5, 10])
    N1_init, N2_init, time = euler_solve(func = dNdt_comp, N1_init = 0.3, 
                                         N2_init = 0.6, dT = 1)
    ax.plot(time, N1_init, label = 'N1_init')
    ax.plot(time, N2_init, label = 'N2_init')

    N1_dthalf, N2_dthalf, time = euler_solve(func = dNdt_comp, N1_init = 0.3, 
                                             N2_init = 0.6, dT = 0.5)
    ax.plot(time, N1_dthalf, label = 'N1 DT Halved')
    ax.plot(time, N2_dthalf, label = 'N2 DT Halved')

    N1_double, N2_double, time = euler_solve(func = dNdt_comp, N1_init = 0.3, 
                                         N2_init = 0.6, dT = 2)
    ax.plot(time, N1_double, label = 'N1 DT Doubled')
    ax.plot(time, N2_double, label = 'N2 DT Doubled')

    N1_quadruple, N2_quadruple, time = euler_solve(func = dNdt_comp, N1_init = 0.3, 
                                             N2_init = 0.6, dT = 4)
    ax.plot(time, N1_quadruple, label = 'N1 DT Quadrupled')
    ax.plot(time, N2_quadruple, label = 'N2 DT Quadrupled')
    ax.legend()
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Population/Carrying Capacity')
    ax.set_title('Lotka-Volterra Competition Model')
    plt.show()

def error_plots_comp_rk8():
    '''
    This functions simply returns a plot of errors using Range-Kutta's method, 
    for the competition model
    '''
    fig, ax = plt.subplots(1,1, figsize=[12.5, 10])
    N1_rk8_init, N2_rk8_init, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.3,
                                                   N2_init = 0.6, dT = 1)
    ax.plot(time_rk8, N1_rk8_init, label = 'N1_rk8_init')
    ax.plot(time_rk8, N2_rk8_init, label = 'N2_rk8_init')

    N1_rk8_dthalf, N2_rk8_dthalf, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.3,
                                                       N2_init = 0.6, dT = 0.5)
    ax.plot(time_rk8, N1_rk8_dthalf, label = 'N1_rk8 DT Halved')
    ax.plot(time_rk8, N2_rk8_dthalf, label = 'N2_rk8 DT Halved')
    N1_rk8_double, N2_rk8_double, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.3,
                                                   N2_init = 0.6, dT = 2)
    ax.plot(time_rk8, N1_rk8_double, '--', label = 'N1_rk8 DT Doubled', color = 'pink')
    ax.plot(time_rk8, N2_rk8_double, label = 'N2_rk8 DT Doubled')

    N1_rk8_quadruple, N2_rk8_quadruple, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.3,
                                                       N2_init = 0.6, dT = 4)
    ax.plot(time_rk8, N1_rk8_quadruple, '--', label = 'N1_rk8 DT Quadrupled')
    ax.plot(time_rk8, N2_rk8_quadruple, label = 'N2_rk8 DT Quadrupled')
    ax.legend()
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Population/Carrying Capacity')
    ax.set_title('Lotka-Volterra Competition Model')
    plt.show()

def error_plots_pred_prey_euler():
    '''
    This functions simply returns a plot of errors using Euler's method, 
    for the predator-prey models
    '''
    fig, ax = plt.subplots(1,1, figsize=[12.5, 10])
    N1_init, N2_init, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3,
                                         N2_init = 0.6, dT = 0.05)
    ax.plot(time, N1_init, label = 'N1_init')
    ax.plot(time, N2_init, label = 'N2_init')

    N1_halved, N2_halved, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3,
                                         N2_init = 0.6, dT = 0.025)
    ax.plot(time, N1_halved, label = 'N1 DT Halved')
    ax.plot(time, N2_halved, label = 'N2 DT Halved')

    N1_double, N2_double, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3,
                                         N2_init = 0.6, dT = 0.1)
    ax.plot(time, N1_double, label = 'N1 DT Double')
    ax.plot(time, N2_double, label = 'N2 DT Double')

    N1_quad, N2_quad, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3,
                                         N2_init = 0.6, dT = 0.2)
    ax.plot(time, N1_quad, label = 'N1 DT Quadruple')
    ax.plot(time, N2_quad, label = 'N2 DT Quadruple')
    ax.legend()
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Population/Carrying Capacity')
    ax.set_title('Lotka-Volterra Predator-Prey Model') 
    fig.tight_layout()
    plt.show()

def error_plots_pred_prey_rk8():
    '''
    This functions simply returns a plot of errors using Range-Kutta's method, 
    for the predator-prey models
    '''
    fig, ax = plt.subplots(1,1, figsize=[12.5, 10])
    N1_rk8_init, N2_rk8_init, time_rk8 = solve_rk8(func = dNdt_pred_prey, 
                                               N1_init = 0.3, N2_init = 0.6, 
                                               dT = 0.05)
    ax.plot(time_rk8, N1_rk8_init, label = 'N1_rk8_init')
    ax.plot(time_rk8, N2_rk8_init, label = 'N2_rk8_init')

    N1_rk8_halved, N2_rk8_halved, time_rk8 = solve_rk8(func = dNdt_pred_prey, 
                                               N1_init = 0.3, N2_init = 0.6, 
                                               dT = 0.025)
    ax.plot(time_rk8, N1_rk8_halved, label = 'N1_rk8 Halved')
    ax.plot(time_rk8, N2_rk8_halved, label = 'N2_rk8 Halved')

    N1_rk8_double, N2_rk8_double, time_rk8 = solve_rk8(func = dNdt_pred_prey, 
                                               N1_init = 0.3, N2_init = 0.6, 
                                               dT = 0.1)
    ax.plot(time_rk8, N1_rk8_double, label = 'N1_rk8 Double')
    ax.plot(time_rk8, N2_rk8_double, label = 'N2_rk8 Double')

    N1_rk8_quad, N2_rk8_quad, time_rk8 = solve_rk8(func = dNdt_pred_prey, 
                                               N1_init = 0.3, N2_init = 0.6, 
                                               dT = 0.2)
    ax.plot(time_rk8, N1_rk8_quad, label = 'N1_rk8 Quadrupled')
    ax.plot(time_rk8, N2_rk8_quad, label = 'N2_rk8 Quadrupled')
    ax.legend()
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Population/Carrying Capacity')
    ax.set_title('Lotka-Volterra Predator-Prey Model') 
    fig.tight_layout()
    plt.show()

def varying_initial_conditions_comp():
    '''
    This functions simply returns a plot of different initial conditions
    using Euler's method, for the competition models
    '''
    fig, axes = plt.subplots(1,2, figsize=[12.5, 10])
    N1_init, N2_init, time = euler_solve(func = dNdt_comp, N1_init = 0.3, 
                               N2_init = 0.6, dT = 1)
    axes[0].plot(time, N1_init, label = 'N1_init')
    axes[0].plot(time, N2_init, label = 'N2_init')
    N1_half, N2_half, time = euler_solve(func = dNdt_comp, N1_init = 0.15, 
                               N2_init = 0.3, dT = 1)
    axes[0].plot(time, N1_half, label = 'N1 Pop. Halved')
    axes[0].plot(time, N2_half, label = 'N2 Pop. Halved')
    N1_2, N2_2, time = euler_solve(func = dNdt_comp, N1_init = 0.2, 
                               N2_init = 0.4, dT = 1)
    axes[0].plot(time, N1_2, label = 'N1 Pop. = 0.2')
    axes[0].plot(time, N2_2, label = 'N2 Pop. = 0.4')
    axes[0].legend()
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Population/Carrying Capacity')
    axes[0].set_title('Lotka-Volterra Competition Model')

    N1_rk8_init, N2_rk8_init, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.3,
                                                   N2_init = 0.6, dT = 1)
    axes[1].plot(time_rk8, N1_rk8_init, label = 'N1_rk8_init')
    axes[1].plot(time_rk8, N2_rk8_init, label = 'N2_rk8_init')

    N1_rk8_half, N2_rk8_half, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.15,
                                                       N2_init = 0.3, dT = 1)
    axes[1].plot(time_rk8, N1_rk8_half, label = 'N1_rk8 Pop. Halved')
    axes[1].plot(time_rk8, N2_rk8_half, label = 'N2_rk8 Pop. Halved')
    N1_rk8_2, N2_rk8_2, time_rk8 = solve_rk8(func = dNdt_comp, N1_init = 0.2,
                                                   N2_init = 0.4, dT = 1)
    axes[1].plot(time_rk8, N1_rk8_2, label = 'N1_rk8 Pop. = 0.2')
    axes[1].plot(time_rk8, N2_rk8_2, label = 'N2_rk8 Pop. = 0.4')
    axes[1].legend()
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Population/Carrying Capacity')
    axes[1].set_title('Lotka-Volterra Competition Model') 
    fig.tight_layout()
    plt.show()

def varying_initial_conditions_pred_prey():
    '''
    This functions simply returns a plot of different initial conditions
    using Euler's method, for the predator-prey models
    '''
    fig, axes = plt.subplots(1,2, figsize=[12.5, 10])
    N1_init, N2_init, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3, 
                                         N2_init = 0.6, dT = 1)
    axes[0].plot(time, N1_init, label = 'N1_init')
    axes[0].plot(time, N2_init, label = 'N2_init')
    N1_half, N2_half, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.15, 
                                         N2_init = 0.3, dT = 1)
    axes[0].plot(time, N1_half, label = 'N1 Pop. Halved')
    axes[0].plot(time, N2_half, label = 'N2 Pop. Halved')
    N1_2, N2_2, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.2, 
                                   N2_init = 0.4, dT = 1)
    axes[0].plot(time, N1_2, label = 'N1 Pop. = 0.2')
    axes[0].plot(time, N2_2, label = 'N2 Pop. = 0.4')
    axes[0].legend()
    axes[0].set_xlabel('Time (years)')
    axes[0].set_ylabel('Population/Carrying Capacity')
    axes[0].set_title('Lotka-Volterra Predator_Prey Model')

    N1_rk8_init, N2_rk8_init, time_rk8 = solve_rk8(func = dNdt_pred_prey, N1_init = 0.3,
                                                   N2_init = 0.6, dT = 1)
    axes[1].plot(time_rk8, N1_rk8_init, label = 'N1_rk8_init')
    axes[1].plot(time_rk8, N2_rk8_init, label = 'N2_rk8_init')

    N1_rk8_half, N2_rk8_half, time_rk8 = solve_rk8(func = dNdt_pred_prey, N1_init = 0.15,
                                                   N2_init = 0.3, dT = 1)
    axes[1].plot(time_rk8, N1_rk8_half, label = 'N1_rk8 Pop. Halved')
    axes[1].plot(time_rk8, N2_rk8_half, label = 'N2_rk8 Pop. Halved')
    N1_rk8_2, N2_rk8_2, time_rk8 = solve_rk8(func = dNdt_pred_prey, N1_init = 0.2,
                                             N2_init = 0.4, dT = 1)
    axes[1].plot(time_rk8, N1_rk8_2, label = 'N1_rk8 Pop. = 0.2')
    axes[1].plot(time_rk8, N2_rk8_2, label = 'N2_rk8 Pop. = 0.4')
    axes[1].legend()
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Population/Carrying Capacity')
    axes[1].set_title('Lotka-Volterra Predator_Prey Model') 
    fig.tight_layout()
    plt.show()

def phase_diagram():
    '''
    This functions simply returns a plot of different initial conditions
    using Euler's method, for the competition models, as phase diagrams. 
    '''
    fig, ax = plt.subplots(1, 1, figsize=[10,8])
    N1, N2, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.3, N2_init = 0.6, 
                               dT = 0.05)
    N1_rk8, N2_rk8, time_rk8 = solve_rk8(func = dNdt_pred_prey, N1_init = 0.3, 
                                         N2_init = 0.6, dT = 0.05)
    ax.plot(N1, N2, label = 'Initial Conditions')
    ax.plot(N1_rk8, N2_rk8, '--', label = 'Initial Conditions RK8')
    N1_2, N2_2, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.2, N2_init = 0.4, 
                                   dT = 0.05)
    ax.plot(N1_2, N2_2, label = 'N1 Pop. = 0.2, N2 Pop. = 0.4')
    N1_1, N2_1, time = euler_solve(func = dNdt_pred_prey, N1_init = 0.35, N2_init = 0.7, 
                                             dT = 0.05)
    ax.plot(N1_1, N2_1, label = 'N1 Pop. = 0.35, N2 Pop. = 0.7')
    ax.legend()
    ax.set_xlabel('Prey Population')
    ax.set_ylabel('Predator Population')
    ax.set_title('Predator vs Prey Model')
    fig.tight_layout()
    plt.show()