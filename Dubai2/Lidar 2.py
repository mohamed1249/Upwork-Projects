###############################################################################
# Written for DEWA Master's in Future Energy Systems and Technology Course 1
# Introduction to LiDAR Mapping
# Copyright 2021 Tarek Zohdi, Emre Mengi. All rights reserved
###############################################################################
"""
Simulation initializes rays downwards. Increment rays in time, and once the rays
hit the surface the reflect back. If the rays reach the original height, stop
simulation.

Keeps track of total reflections, and uses rays that return to sensor to perform
time-of-flight calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
from matplotlib import rc
from matplotlib import cm
###################################################################################################
############################### Lidar Constants and Parameters ####################################
###################################################################################################
A_ = 0.64                                       # surface amplitudes
c = 3e8                                         # speed of light
z0 = 4                                          # scanner radial distance from center of the surface
L1 = 1                                          # surface oscillation height
L2 = 1                                          # surface oscillation height
dt = 0.001 * z0/c                               # time step size
omega1 = 2                                      # surface frequency 1
omega2 = 1                                      # surface frequency 2
###################################################################################################
##################################### Simulation Functions ########################################
###################################################################################################

# Surface functions
def surfG(x1, x2, A):
    """
    Equation 1
    :param x1: x position of r
    :param x2: y position of r
    :param A: surface parameter\
    :return: x1 (height of surface)
    """
    return 2 + A * np.sin(2 * omega1 * np.pi * x1 / L1) * np.sin(2 * omega2 * np.pi * x2 / L2)
# Random Number of Rays Generator - generates 4 nRay values to be tested by the LiDAR sim
def rayGen():
    
    nRays = np.zeros(3)
    
    #define ray number bounds
    n1_lo = 40
    n1_hi = 60
    
    n2_lo = 400
    n2_hi = 600
    
    n3_lo = 900
    n3_hi = 1100
    
    n4_lo = 4500
    n4_hi = 5500
    
    #randomizing variables
    n1 = (n1_hi - n1_lo) * np.random.rand() + n1_lo
    n2 = (n2_hi - n2_lo) * np.random.rand() + n2_lo
    n3 = (n3_hi - n3_lo) * np.random.rand() + n3_lo
    n4 = (n4_hi - n4_lo) * np.random.rand() + n4_lo
    
    #randomize ray numbers around previously defined estimates
    nRays = np.array([n1,n2,n3,n4])
    
    nRays = nRays.astype(int)

    return nRays

nRays = rayGen()
print(nRays) #include the number of rays for your surface reconstruction plots
# Lidar simulation function
def lidarsim(nRays):
    """
    Lidar simulation
    :param surface: index for A[i]
    :return: list of ray positions throughout entire simulation
    """

    # Solution lists
    posTot = []
    timeTot = []

    # initialize ray positions and velocities
    xMax = 0.5 #domain limits
    xMin = -0.5
    yMax = 0.5
    yMin = -0.5
    
    # initialize ray positions and velocities
    phi1 = np.random.rand(nRays) - 0.5                                              # random variable 1     | +-0.5 |
    phi2 = np.random.rand(nRays) - 0.5                                              # random variable 2     | +-0.5 |
    r0 = np.array([phi1, phi2, z0 * np.ones(nRays)]).T                              # initial positions     |    m     |
    v0 = np.array([np.zeros(nRays), np.zeros(nRays), -c * np.ones(nRays)]).T        # ray velocity          |   m/s    |
    
    # Set current positions and velocities to initial conditions
    r = r0.copy()                                               # shallow copy to ensure full copy of array to memory 
    v = v0.copy()                                               # shallow copy to ensure full copy of array to memory
    # Gh = surfG(r[:, 0], r[:, 1], A_[0])  # surface plot points

    # tracking arrays
    active = np.ones(nRays, dtype=bool)                         # flag array to track "active" rays
    rtrn = np.zeros(nRays, dtype=bool)                          # flag array to track which rays are valid
    rfln = np.zeros(nRays)                                      # track number reflections

    time = 0                                                    # initialize time
    tj = np.zeros(nRays)                                        # initialize time-of-flight array
    n = [0,0,1]                                                 # initialize normal vector array
    thetai = np.zeros((nRays, 1))                               # initialize reflected angle array
    vjperp = np.zeros((nRays, 3))                               # initialize inbound normal velocity array
    vjref = np.zeros((nRays, 3))                                # initialize outgoing velocity array

    counter = 0
    while any(active):
        time += dt

        # Check if rays are below the surface
        idx = r[:, 2] <= surfG(r[:, 0], r[:, 1], A_)
        idx = np.where(np.logical_and(idx, active))[0]

        # For any rays that are CURRENTLY interacting with the surface, compute new velocities
        if idx.size != 0:
            # Update reflection counter array
            rfln[idx] += 1

            # compute normal vector for rays on surface

            vNorm = np.linalg.norm(v[idx, :], ord=2, axis=1)[:, np.newaxis]                     # should be -> c

            thetai[idx] = np.sum(v[idx, :] * n, axis=1)[:, np.newaxis] / (vNorm * 1)
            thetai[idx] = np.arccos(thetai[idx])

            # compute perpendicular component of velocity
            vjperp[idx, :] = vNorm * np.cos(thetai[idx]) * n                                    # equation 10
            vjref[idx, :] = v[idx, :] - 2 * vjperp[idx, :]                                      # equation 11
            v[idx, :] = vjref[idx, :]                                                           # update velocity vector

        """
        Here 'idx' is used as the array that stores the indices of the desired parameters. For criteria, 'idx' is 
        reset to new indices.  
        
        Check the following criteria:
        - if rays have returned to scanner
        - if rays have reflected more than once
        - if rays have reflected outside of domain
        
        """
        # check if rays have returned to scanner
        idx = r[:, 2] > z0
        idx = np.where(np.logical_and(idx, active))[0]
        active[idx] = False                                         # update flag array to deactivate returned arrays
        rtrn[idx] = True                                            # update return array to denote valid rays
        tj[idx] = time                                              # store time of flight

         # check if rays have reflected more than once
        idx = rfln[:] > 1                                           # for all rays that have reflected more than once
        idx = np.where(np.logical_and(idx, active))[0]              # check for reflected AND active rays
        active[idx] = False                                         # deactivate rays

        # check if rays have reflected outside of the domain

        idxX = np.where(np.logical_or(r[:, 0] < -0.5, r[:, 0] > 0.5))[0]   # indices of arrays outside of x domain & active
        idxY = np.where(np.logical_or(r[:, 1] < -0.5, r[:, 1] > 0.5))[0]   # indices of arrays outside of y domain & active
        active[idxX] = False                                            # deactivate rays outside of x domain
        active[idxY] = False                                            # deactivate rays outside of y domain

        # keep rays moving downward active even if they are outside the domain (since angled rays might start outside the limits)
        #idxV = np.where(v[:, 2] < 0)[0]
        #active[idxV] = True
        
        
        # Forward Euler Time Stepping Scheme
        idx = active == 1                                               # compute only active rays
        r[idx, :] += dt * v[idx, :]                                     # equation 17

        # store solutions (every 10 time steps) for animation
        if counter % 10 == 0:
            posTot.append(r[idx, :].copy())
            timeTot.append(time)
        counter += 1

    # # Compute geometric properties of all return rays

    # Compute distance traveled using time of flight
    tj = tj[:, np.newaxis]                                                                  # log total time-of-flight
    d1 = c * tj[rtrn] * 1/2

    # Compute point cloud positions
    rp = np.hstack((r0[rtrn, :2], z0 - d1))                                                 # equation 16

    # Point-cloud reconstruction plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(rp[:, 0], rp[:, 1], rp[:, 2], marker='o', s=25, color='red', edgecolor='grey')
    ax.set_title("Point Cloud of Lidar for %d Rays" % nRays)
    ax.set_xlabel('X (m)', fontsize=15)
    ax.set_ylabel('Y (m)', fontsize=15)
    ax.set_zlabel('Z (m)', fontsize=15)
    #ax.set_zlim((1.5, 4))
    ax.set_xlim((-0.75, 0.75))
    ax.set_ylim((-0.75, 0.75))
    # ax.set_zlim((1.5, 4))
    plt.savefig('point_cloud_reconstruction_A{}.tiff'.format(A_))
    plt.show()

    return posTot

###################################################################################################
##################################### Run Simulation ##############################################
###################################################################################################

for i in range(len(nRays)): #runs simulation and produces 4 plots corresponding to 4 nRays values
    posTot = lidarsim(nRays[i])
