# Standard set up for the other files
# Has some basic functions which get used everywhere else

import numpy as np
from numba import njit

# Select initial conditions
IC_type = "shu-osher"  # Options: "brio-wu", "shu-osher", "ryu-jones"
time_integral = "SSPRK4" # Options: "SSPRK4", "RK4"

# Constants
gamma = 1.4        # Ratio of specific heats; modified for certain problems
Nx = 256           # Number of grid points
L = 1.0            # Domain length
dx = L / (Nx - 1)  # Spatial step
CFL = 0.4          # Courant number
mu_0 = 1.0         # Permeability of free space

# Change gamma for the different problems
if IC_type == "brio-wu":
    gamma = 2.0
elif IC_type == "ryu-jones":
    gamma = 5.0/3.0
elif IC_type == "Gauss":
    L = 2.0

# Spatial grid
x = np.linspace(0, L, Nx)

# Very common functions
@njit
# Function to convert conservative -> primitive variables
def cons_to_prim(rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z):
    # Velocities
    u = mom_x / rho
    v = mom_y / rho
    w = mom_z / rho
    # Thermal pressure
    p = (gamma - 1) * (E - 0.5*rho*(u**2+v**2+w**2) - 0.5*(B_x**2+B_y**2+B_z**2))
    # Total pressure -> For the flux calculations
    # Adds in magnetic pressure
    p_t = p + 0.5*(B_x**2 + B_y**2 + B_z**2)
    return rho, u, v, w, p, p_t, B_x, B_y, B_z

@njit
# Function to convert primitive -> conservative variables
def prim_to_cons(rho, u, v, w, p, B_x, B_y, B_z):
    mom_x = rho * u
    mom_y = rho * v
    mom_z = rho * w
    E = p/(gamma - 1) + 0.5*rho*(u**2+v**2+w**2) + 0.5*(B_x**2+B_y**2+B_z**2)
    return rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z
