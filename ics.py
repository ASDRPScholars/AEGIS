import numpy as np
from config import Nx, gamma

def initialize_problems(IC_type, x):
    
    # Initialize all arrays; B_x will be set constant across the domain
    rho = np.zeros(Nx)    # Density
    p = np.zeros(Nx)      # Pressure
    u = np.zeros(Nx)      # X-Axis Velocity
    mom_x = np.zeros(Nx)  # X-Axis Momentum
    v = np.zeros(Nx)      # Y-Axis Velocity
    mom_y = np.zeros(Nx)  # Y-Axis Momentum
    w = np.zeros(Nx)      # Z-Axis Velocity
    mom_z = np.zeros(Nx)  # Z-Axis Momentum
    E = np.zeros(Nx)      # Total energy
    B_x = 0               # X-Axis Magnetic Field
    B_y = np.zeros(Nx)    # Y-Axis Magnetic Field
    B_z = np.zeros(Nx)    # Z-Axis Magnetic Field

    
    # Initial condition set up
    if IC_type == "brio-wu":
        # This is the Brio-Wu Shock-Tube Problem
        # Very basic benchmark for MHD codes

        shock_pos = 0.5 # Shock position given by the problem
        B_x = 0.75 # Constant aross the problem (this is the case for most 1D setups)

        # Left (of the shock) states
        rho_left = 1.0
        p_left = 1.0
        u_left = 0.0
        v_left = 0.0
        w_left = 0.0
        B_y_left = 1.0
        B_z_left = 0.0

        # Right (of the shock) states
        rho_right = 0.125
        p_right = 0.1
        u_right = 0.0
        v_right = 0.0
        w_right = 0.0
        B_y_right = -1.0
        B_z_right = 0.0

        rho = np.where(x < shock_pos, rho_left, rho_right)
        p = np.where(x < shock_pos, p_left, p_right)
        u = np.where(x < shock_pos, u_left, u_right)
        v = np.where(x < shock_pos, v_left, v_right)
        w = np.where(x < shock_pos, w_left, w_right)
        B_y = np.where(x < shock_pos, B_y_left, B_y_right)
        B_z = np.where(x < shock_pos, B_z_left, B_z_right)

        t_final = 0.1
        bc = "normal"

    elif IC_type == "ryu-jones":
        # This is the Ryu-Jones 2A shock-tube problem
        # This will be used to evaluate AEGIS over a more complex MHD problem

        shock_pos = 0.5 # Shock position given by the problem
        B_scale = 1.0/np.sqrt(4*np.pi) # Scale factor for the magnetic field
        B_x = 2.0*B_scale # Constant across the domain

        rho_l = 1.08
        u_l = 1.2
        v_l = 0.01
        w_l = 0.5
        B_y_l = 3.6*B_scale
        B_z_l = 2.0*B_scale
        p_l = 0.95

        rho_r = 1.0
        u_r = 0.0
        v_r = 0.0
        w_r = 0.0
        B_y_r = 4.0*B_scale
        B_z_r = 2.0*B_scale
        p_r = 1.0

        rho = np.where(x < shock_pos, rho_l, rho_r)
        u = np.where(x < shock_pos, u_l, u_r)
        v = np.where(x < shock_pos, v_l, v_r)
        w = np.where(x < shock_pos, w_l, w_r)
        p = np.where(x < shock_pos, p_l, p_r)
        B_y = np.where(x < shock_pos, B_y_l, B_y_r)
        B_z = np.where(x < shock_pos, B_z_l, B_z_r)

        t_final = 0.2
        bc = "normal"


    elif IC_type == "shu-osher":
        # This is the Shu-Osher shock-tube problems
        # I'll use this to see if the solver is working correctly when it comes to
        # going back to non-MHD solutions

        # Left state (post-shock)
        rho_l = 27/7
        p_l = 31/3
        u_l = 2.629369

        # Right state (pre-shock with oscillations)
        rho_r = 1.0 + 0.2 * np.sin(50 * x)  # Density with sine wave
        p_r = 1.0
        u_r = 0.0

        # Combine left and right states
        shock_pos = 0.1
        rho = np.where(x < shock_pos, rho_l, rho_r)
        p = np.where(x < shock_pos, p_l, p_r)
        u = np.where(x < shock_pos, u_l, u_r)

        t_final = 0.18
        bc = "normal"

    elif IC_type == "sod":
        # Classic Sod's shock tube
        # Part of the standard, non-MHD suite of problems to test the code over standard CFD
        
        rho[:] = 1.0
        rho[x >= 0.5] = 0.125  # Right side density
    
        p[:] = 1.0
        p[x >= 0.5] = 0.1  # Right side pressure
    
        u[:] = 0.0  # Velocity (zero everywhere)

        # No magnetic field or velocities in other directions, so initialized to 0 above
    
        t_final = 0.2  # Final time
        bc = "normal"

    # Convert to conservative variables
    # Total pressure
    p_t = p + 0.5*(B_x**2 + B_y**2 + B_z**2)
    # Total energy
    E = p/(gamma - 1) + 0.5*rho*(u**2+v**2+w**2) + 0.5*(B_x**2+B_y**2+B_z**2)
    # Momentums
    mom_x = rho * u
    mom_y = rho * v
    mom_z = rho * w

    return rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z, t_final, bc
