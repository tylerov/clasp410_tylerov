#!/usr/bin/env python3
'''
Doc String

The monte carlo represents the real world in a way that the average 
physical processes may be a good way of being represented, but if winds 
are extremely high or easily catchable shrubs, then this model does not work
very well. 
'''

import numpy as np
import matplotlib.pyplot as plt


def forest_fire(nstep = 4, isize = 3, jsize = 3, pspread = 1.0):
    '''
    Create a forest fire

    Parameters:
    nstep: int, defaults to 4
        Set size of forest in x and y direction, respectively
    isize, jsize: int, defaults to 3
        Set size of forest in x and y direction, respectively
    pspread: float, defaults to 1.0
        Set chance that fire can spread in any direction from 0 to 1
    '''

    # Creating a forest and making all spots have trees.
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Set initial fire to center [NEED TO UPDATE THIS FOR LAB]:
    forest[0, isize//2, jsize//2] = 3

    # Loop through time to advance our fire
    for k in range(nstep - 1):
        # Search every spot that is on fire and spread fire as needed.
        for i in range(isize):
            for j in range(jsize):
                # Are we on fire?
                if forest[k, i, j] != 3:
                    continue
                # Spread fire in each direction
                # Spread "up" (i to i-1)
                if (pspread > rand()) and (forest[k, i - 1, j] == 2) and (i > 0):
                    forest[k + 1, i - 1, j] = 3
                # Spread down
                # Spread east
                # Spread west
    return forest



