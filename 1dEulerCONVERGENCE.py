import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit

'''
Finished Convergence Plot!
'''

# Just to get it across everything
gamma = 1.4  # Ratio of specific heats
CFL = 0.4   # Courant number

# Select initial conditions
IC_type = "Gauss"  # Options: "sod", "shu-osher", "lax_tube", "Gauss"

# Create output directory if it doesn't exist
output_dir = "results_lf"
os.makedirs(output_dir, exist_ok=True)

def exact_solution(Nx_exact):
    rho, mom, E, t_final, dx, x = setUp(Nx_exact)

    # Time integration loop
    t = 0.0
    step = 0

    while t < t_final:

        print(f"[EXACT] Time: {t:.6f}, [EXACT] Timestep Num: {step}")
        # Convert conservative -> primitive
        rho, u, p = cons_to_prim(rho, mom, E)

        # Compute speed of sound
        c = np.sqrt(gamma * p / rho)

        # Compute time step based on CFL condition
        dt = min(CFL * dx / np.max(np.abs(u) + c), t_final - t)

        # Calculate all time steps
        rho, mom, E = WENO_solution(rho, mom, E, dt, dx)
        t += dt
        step += 1
    print("Exact solution finished!")
    return rho, mom, E, x

@njit
def first_order_solution(rho, mom, E, dt, dx):
    # I'm literally just going to copy paste from Chris' original LF code
    # Convert conservative -> primitive
    rho, u, p = cons_to_prim(rho, mom, E)

    # Compute fluxes at every cell
    F = compute_flux(rho, mom, E, p)

    # Lax-Friedrichs update: averaged neighbors + central flux difference
    rho[1:-1] = 0.5 * (rho[2:] + rho[:-2]) - dt / (2 * dx) * (F[0, 2:] - F[0, :-2])
    mom[1:-1] = 0.5 * (mom[2:] + mom[:-2]) - dt / (2 * dx) * (F[1, 2:] - F[1, :-2])
    E[1:-1]   = 0.5 * (E[2:]   + E[:-2])   - dt / (2 * dx) * (F[2, 2:] - F[2, :-2])

    # Apply boundary conditions (copy neighbor so waves pass through)
    rho, mom, E = apply_bcs(rho, mom, E)

    return rho, mom, E

@njit
def minmod3(a, b, c):
  # Minmod limiter used by the MC reconstruction.
  same_sign = (a * b > 0.0) & (a * c > 0.0)
  limited = np.sign(a) * np.minimum(
      np.abs(a),
      np.minimum(np.abs(b), np.abs(c))
  )
  return np.where(same_sign, limited, 0.0)

@njit
def reconstruct_MUSCL_MC(q):
  # 2nd-order MUSCL reconstruction with an MC slope limiter
  dq_left = q[1:-1] - q[:-2]
  dq_center = 0.5 * (q[2:] - q[:-2])
  dq_right = q[2:] - q[1:-1]

  slope = minmod3(
      2.0 * dq_left,
      dq_center,
      2.0 * dq_right
  )

  # Match the N-4 layout returned by reconstruct_5th_order.
  center = q[2:-2]
  slope = slope[1:-1]

  q_left = center - 0.5 * slope
  q_right = center + 0.5 * slope
  return q_left, q_right

@njit
def reconstruct_5th_order(u):
  uL = (2*u[:-4] - 13*u[1:-3] + 47*u[2:-2] + 27*u[3:-1] - 3*u[4:])/60
  uR = (2*u[4:] - 13*u[3:-1] + 47*u[2:-2] + 27*u[1:-3] - 3*u[:-4])/60
  return uR, uL

@njit
def reconstruct_4th_order(u):
  uL = (-u[:-4] + 7*u[1:-3] + 7*u[2:-2] - u[3:-1])/12
  uR = (-u[1:-3] + 7*u[2:-2] + 7*u[3:-1] - u[4:])/12
  return uL, uR

@njit
def shock_Detector(p):

  # This is an extended form of the JST edge sensor

  sensor_num = (p[:-4] - 4*p[1:-3] + 6*p[2:-2] - 4*p[3:-1] + p[4:])
  sensor_den = (p[:-4] + 4*p[1:-3] + 6*p[2:-2] + 4*p[3:-1] + p[4:])

  sensor = np.abs(sensor_num/sensor_den)

  return sensor

def interpolation(rho_L, rho_R, mom_L, mom_R, E_L, E_R):
    u_L = mom_L / rho_L
    u_R = mom_R / rho_R

    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)

    # Compute fluxes at every cell

    F_L = compute_flux(rho_L, mom_L, E_L, p_L)
    F_R = compute_flux(rho_R, mom_R, E_R, p_R)

    U_left = np.array([rho_R[:-1], mom_R[:-1], E_R[:-1]])
    U_right  = np.array([rho_L[1:],  mom_L[1:],  E_L[1:]])

    # Compute speed of sound for BOTH sides
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    F_half = np.zeros_like(U_left)

    # Variables look similar, but are diff in this case, sorry for poor naming

    uL = u_R[:-1]
    uR = u_L[1:]

    cL = c_R[:-1]
    cR = c_L[1:]

    pL = p_R[:-1]
    pR = p_L[1:]

    rhoL = rho_R[:-1]
    rhoR = rho_L[1:]

    FL = F_R[:, :-1]
    FR = F_L[:, 1:]

    EL = E_R[:-1]
    ER = E_L[1:]

    # Taken from the Toro book, pdf page 353

    S_L = np.minimum(uL - cL, uR - cR)
    S_R = np.maximum(uL + cL, uR + cR)

    S_star = (pR - pL + rhoL*uL*(S_L-uL) - rhoR*uR*(S_R-uR))/(
        rhoL*(S_L-uL) - rhoR*(S_R-uR)
    )

    # Setting up the D_star stuff

    D_star = np.zeros((3, len(S_star)))

    D_star[0, :] = 0
    D_star[1, :] = 1
    D_star[2, :] = S_star

    # Getting the star state fluxes

    F_star_L_num = S_star * (S_L*U_left - FL) + S_L*(pL+rhoL*(S_L-uL)*(S_star-uL))*D_star
    F_star_L_den = S_L - S_star

    F_star_R_num = S_star * (S_R*U_right - FR) + S_R*(pR+rhoR*(S_R-uR)*(S_star-uR))*D_star
    F_star_R_den = S_R - S_star

    F_star_L = F_star_L_num/F_star_L_den
    F_star_R = F_star_R_num/F_star_R_den

    left = (0 <= S_L)
    left_star = (S_L <= 0) & (0 <= S_star)
    right_star = (S_star <= 0) & (0 <= S_R)
    right = (0 >= S_R)

    F_half[:, left] = FL[:, left]
    F_half[:, right] = FR[:, right]

    F_half[:, left_star] = F_star_L[:, left_star]
    F_half[:, right_star] = F_star_R[:, right_star]

    return F_half

def setUp(spaceSteps):
    # Constants
    Nx = spaceSteps     # Number of grid points
    L = 2.0      # Domain length, either 1.0 or 2.0 depending on problem
    dx = L / (Nx - 1)  # Spatial step

    # Spatial grid
    x = np.linspace(0, L, Nx)

    # Initialize all arrays
    rho = np.zeros(Nx)  # Density
    p = np.zeros(Nx)    # Pressure
    u = np.zeros(Nx)    # Velocity
    E = np.zeros(Nx)    # Total energy
    mom = np.zeros(Nx)  # Momentum

    # Initial conditions
    if IC_type == "sod":
        # Sod shock tube
        rho[:] = 1.0
        rho[x >= 0.5] = 0.125  # Right side density

        p[:] = 1.0
        p[x >= 0.5] = 0.1  # Right side pressure

        u[:] = 0.0  # Velocity (zero everywhere)

        t_final = 0.2  # Final time


    elif IC_type == "shu-osher":
        # Shu-Osher problem
        # Left state (post-shock)
        rho_l = 3.857143
        p_l = 10.33333
        u_l = 2.629369

        # Right state (pre-shock with oscillations)
        rho_r = 1.0 + 0.2 * np.sin(5 * np.pi * x)  # Density with sine wave
        p_r = np.ones(Nx)
        u_r = np.zeros(Nx)

        # Combine left and right states
        shock_pos = 0.1
        rho = np.where(x < shock_pos, rho_l, rho_r)
        p = np.where(x < shock_pos, p_l, p_r)
        u = np.where(x < shock_pos, u_l, u_r)

        t_final = 0.18 # Final time
        # t_final = 0.17 # Modified bc of RAM constraints

    elif IC_type == "lax_tube":
        # Trying to implement the Lax Shock Tube problem
        # Just want to run more tests on this approach

        # Left state
        rho_l = 0.445
        p_l = 3.528
        u_l = 0.698

        # Right state
        rho_r = 0.5
        p_r = 0.571
        u_r = 0.0

        # Combining left and right states
        shock_pos = 0.5
        rho = np.where(x < shock_pos, rho_l, rho_r)
        p = np.where(x < shock_pos, p_l, p_r)
        u = np.where(x < shock_pos, u_l, u_r)

        t_final = 0.14

    elif IC_type == "Gauss":
        # This is for the convergence study

        rho = 1.0 + 0.1*np.exp(-((x - 1.0)**2)/(2*0.1**2))
        u = 1.0
        p = 1.0

        t_final = 0.5

    # Convert to conservative variables
    E = p / (gamma - 1) + 0.5 * rho * u**2  # Total energy
    mom = rho * u  # Momentum

    return rho, mom, E, t_final, dx, x

# Function to convert conservative -> primitive variables
@njit
def cons_to_prim(rho, mom, E):
    u = mom / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    return rho, u, p

# Function to convert primitive -> conservative variables
@njit
def prim_to_cons(rho, u, p):
    mom = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return rho, mom, E

# Function to compute fluxes
@njit
def compute_flux(rho, mom, E, p):
    F = np.zeros((3, len(rho)))
    F[0, :] = mom
    F[1, :] = mom**2 / rho + p
    F[2, :] = (E + p) * mom / rho
    return F

# Component-wise finite-difference WENO5-JS with global Lax–Friedrichs flux splitting and RK4.

#---------------------------------------------------------------------------------------------------

@njit
def weno5_plus(F):

    eps = 1e-6

    N = F.shape[1]

    Fpad = np.empty((3, N + 6))

    Fpad[:,3:-3] = F

    Fpad[:,0] = F[:,0]
    Fpad[:,1] = F[:,0]
    Fpad[:,2] = F[:,0]

    Fpad[:,-3] = F[:,-1]
    Fpad[:,-2] = F[:,-1]
    Fpad[:,-1] = F[:,-1]

    flux = np.zeros((3, N+1))

    for k in range(N+1):

        i = k + 2

        f0 = Fpad[:,i-2]
        f1 = Fpad[:,i-1]
        f2 = Fpad[:,i]
        f3 = Fpad[:,i+1]
        f4 = Fpad[:,i+2]

        # Candidate polynomials
        q0 = ( 2*f0 - 7*f1 + 11*f2)/6.0
        q1 = (-f1 + 5*f2 + 2*f3)/6.0
        q2 = ( 2*f2 + 5*f3 - f4)/6.0

        # Jiang-Shu smoothness indicators
        beta0 = (13.0/12.0)*(f0 - 2*f1 + f2)**2 \
              + 0.25*(f0 - 4*f1 + 3*f2)**2

        beta1 = (13.0/12.0)*(f1 - 2*f2 + f3)**2 \
              + 0.25*(f1 - f3)**2

        beta2 = (13.0/12.0)*(f2 - 2*f3 + f4)**2 \
              + 0.25*(3*f2 - 4*f3 + f4)**2

        # Linear weights
        d0 = 0.1
        d1 = 0.6
        d2 = 0.3

        alpha0 = d0 / (eps + beta0)**2
        alpha1 = d1 / (eps + beta1)**2
        alpha2 = d2 / (eps + beta2)**2

        asum = alpha0 + alpha1 + alpha2

        w0 = alpha0/asum
        w1 = alpha1/asum
        w2 = alpha2/asum

        flux[:,k] = w0*q0 + w1*q1 + w2*q2

    return flux


def weno5_minus(F):
    #negative moving flux (r to l) is shortcutted by flipping regular flux
    return np.flip(weno5_plus(np.flip(F, axis=1)), axis=1)

#dU/dt = RHS
#U(t) + Fx(U) = 0
#RHS = -Fx(U)
def compute_rhs(rho, mom, E, dx):
    U = np.vstack((rho, mom, E))

    rho_temp, u_temp, p_temp = cons_to_prim(rho, mom, E)

    #computes total euler flux
    F = compute_flux(rho_temp, mom, E, p_temp)

    c = np.sqrt(gamma * p_temp / rho_temp)
    #speed of sound estimate the max wave speed
    alpha = np.max(np.abs(u_temp) + c)

    #left to right & right to left fluxes
    F_plus = 0.5 * (F + alpha * U)
    F_minus = 0.5 * (F - alpha * U)

    #total flux is the sum of both fluxes
    flux = weno5_plus(F_plus) + weno5_minus(F_minus)


    RHS = np.zeros_like(U)

    #RHS_i = -dF/dx
    #RHS_i approximated by (flux(i+1/2)-flux(i-1/2))/dx
    RHS[:, 1:-1] = -(flux[:, 2:-1] - flux[:, 1:-2]) / dx

    return RHS

#---------------------------------------------------------------------------------------------------

# Okay now from here I will be integrating WENO into the architecture I currently have going on

def WENO_solution(rho, mom, E, dt, dx):

    # Lax-Friedrichs update: averaged neighbors + central flux difference
    U = np.vstack((rho, mom, E))

    U[0], U[1], U[2] = apply_bcs(U[0], U[1], U[2])

    #compute slope (RK stage 1)

    RHS1 = compute_rhs(U[0], U[1], U[2], dx)
    #uses forward euler to find the next U step
    #U(t) + Fx(U)
    U1 = U + 0.5 * dt * RHS1

    #applies boundary conditions on U1
    #0 gradient boundary
    U1[0], U1[1], U1[2] = apply_bcs(U1[0], U1[1], U1[2])

    #compute slope (RK stage 2) again using U1 that was just calculated
    RHS2 = compute_rhs(U1[0], U1[1], U1[2], dx)
    #uses RK stage 2 to factor 2 U steps ahead
    U2 = U + 0.5 * dt * RHS2

    # applies boundary conditions on U2
    U2[0], U2[1], U2[2] = apply_bcs(U2[0], U2[1], U2[2])

    #compute RK stage 3 again using the U2
    RHS3 = compute_rhs(U2[0], U2[1], U2[2], dx)

    # Creating the value to calculate the 4th time step
    U3 = U + dt * RHS3

    # Same BCs as before
    U3[0], U3[1], U3[2] = apply_bcs(U3[0], U3[1], U3[2])

    # Calculate RK stage 4 to factor in 4th order accuracy

    RHS4 = compute_rhs(U3[0], U3[1], U3[2], dx)

    #Calculate final RK stage 4 to factor in 4th order accuracy
    Unew = U + (dt / 6.0) * (RHS1 + 2.0 * RHS2 + 2.0 * RHS3 + RHS4)

    #updates conservative variables
    rho = Unew[0]
    mom = Unew[1]
    E = Unew[2]

    # Apply boundary conditions (keep it the same as the rest of the methods)

    rho, mom, E = apply_bcs(rho, mom, E)

    return rho, mom, E

def rhs(rho, mom, E, dt, blend):

    drho = np.zeros_like(rho)
    dmom = np.zeros_like(mom)
    dE = np.zeros_like(E)

    positive_floor = 1e-12
    center = slice(2, -2)

    U = np.vstack((rho, mom, E))

    # Convert conservative variables to primitive variables
    rho, u, p = cons_to_prim(rho, mom, E)

    shockArray_p = shock_Detector(p)
    shockArray_rho = shock_Detector(rho)

    shockArray = np.maximum(shockArray_p, shockArray_rho)

    # Fourth-order primitive-variable reconstruction

    rho_L_4, rho_R_4 = reconstruct_4th_order(rho)
    u_L_4, u_R_4 = reconstruct_4th_order(u)
    p_L_4, p_R_4 = reconstruct_4th_order(p)

    rho_L_4, mom_L_4, E_L_4 = prim_to_cons(
        rho_L_4, u_L_4, p_L_4
    )
    rho_R_4, mom_R_4, E_R_4 = prim_to_cons(
        rho_R_4, u_R_4, p_R_4
    )

    F_half4 = interpolation(
        rho_L_4, rho_R_4,
        mom_L_4, mom_R_4,
        E_L_4, E_R_4
    )

    # MUSCL-MC Reconstruction

    rho_L_MC, rho_R_MC = reconstruct_MUSCL_MC(rho)
    u_L_MC, u_R_MC = reconstruct_MUSCL_MC(u)
    p_L_MC, p_R_MC = reconstruct_MUSCL_MC(p)

    # This just an insurance against unphysical states
    rho_L_MC = np.where(
        rho_L_MC > positive_floor, rho_L_MC, rho[center]
    )
    rho_R_MC = np.where(
        rho_R_MC > positive_floor, rho_R_MC, rho[center]
    )
    p_L_MC = np.where(
        p_L_MC > positive_floor, p_L_MC, p[center]
    )
    p_R_MC = np.where(
        p_R_MC > positive_floor, p_R_MC, p[center]
    )

    rho_L_MC, mom_L_MC, E_L_MC = prim_to_cons(
        rho_L_MC, u_L_MC, p_L_MC
    )
    rho_R_MC, mom_R_MC, E_R_MC = prim_to_cons(
        rho_R_MC, u_R_MC, p_R_MC
    )

    F_half_MC = interpolation(
        rho_L_MC, rho_R_MC,
        mom_L_MC, mom_R_MC,
        E_L_MC, E_R_MC
    )

    # Fifth-order primitive-variable reconstruction

    rho_L_5, rho_R_5 = reconstruct_5th_order(rho)
    u_L_5, u_R_5 = reconstruct_5th_order(u)
    p_L_5, p_R_5 = reconstruct_5th_order(p)

    # Just a safety net on negative predicted values from the reconstruction

    rho_L_5 = np.where(
        rho_L_5 > positive_floor, rho_L_5, rho[center]
    )
    rho_R_5 = np.where(
        rho_R_5 > positive_floor, rho_R_5, rho[center]
    )
    p_L_5 = np.where(
        p_L_5 > positive_floor, p_L_5, p[center]
    )
    p_R_5 = np.where(
        p_R_5 > positive_floor, p_R_5, p[center]
    )

    rho_L_5, mom_L_5, E_L_5 = prim_to_cons(
        rho_L_5, u_L_5, p_L_5
    )
    rho_R_5, mom_R_5, E_R_5 = prim_to_cons(
        rho_R_5, u_R_5, p_R_5
    )

    F_half5 = interpolation(
        rho_L_5, rho_R_5,
        mom_L_5, mom_R_5,
        E_L_5, E_R_5
    )

    # First-order local Rusanov flux
    F = compute_flux(rho, mom, E, p)

    c = np.sqrt(gamma * p / rho)
    wave_speed = np.abs(u) + c

    a_face = np.maximum(
        wave_speed[2:-3],
        wave_speed[3:-2]
    )

    F_half1 = (
        0.5 * (F[:, 2:-3] + F[:, 3:-2])
        - 0.5 * a_face[None, :]
          * (U[:, 3:-2] - U[:, 2:-3])
    )

    # Fifth order in smooth regions
    F_half_hi = F_half5

    # Original fourth/first-order blend beside shocks
    F_half_lo = 0.75 * F_half_MC + 0.25 * F_half1

    # Convert the cell-centered sensor to one shared value at each face
    edge_sensor = np.maximum(
        shockArray[:-1],
        shockArray[1:]
    )

    U_temp = np.vstack((rho, mom, E))

    art_dissipation = (U[:, :-4] - 4*U[:, 1:-3] + 6*U[:, 2:-2] - 4*U[:, 3:-1] + U[:, 4:])

    art_dissipation = edge_sensor[None, :]*np.maximum(art_dissipation[:, :-1], art_dissipation[:, 1:])

    # Gaussian curve for the sensor scale

    sensor_scale = 0.02

    shockWeight = 1.0 - np.exp(
        -(edge_sensor / sensor_scale)**2
    )

    # Blend fluxes before taking their divergence
    F_half = (
        (1.0 - shockWeight) * F_half_hi
        + shockWeight * F_half_lo
    )

    drho[3:-3] = -(F_half[0, 1:] - F_half[0, :-1]) / dx
    dmom[3:-3] = -(F_half[1, 1:] - F_half[1, :-1]) / dx
    dE[3:-3] = -(F_half[2, 1:] - F_half[2, :-1]) / dx

    return drho, dmom, dE

@njit
def apply_bcs(rho, mom, E):
    # Applying BCs

    # Just going with simple 0 gradient in/outflows

    rho[:3] = rho[3]
    rho[-3:] = rho[-4]

    E[:3] = E[3]
    E[-3:] = E[-4]

    mom[:3] = mom[3]
    mom[-3:] = mom[-4]

    '''
    # Changing to periodic for the smooth sine wave problem
    rho[:3]  = rho[-6:-3]
    rho[-3:] = rho[3:6]

    mom[:3]  = mom[-6:-3]
    mom[-3:] = mom[3:6]

    E[:3]    = E[-6:-3]
    E[-3:]   = E[3:6]
    '''

    return rho, mom, E


def time_integration(rho, mom, E, blend, dt, dx):
    # Convert conservative -> primitive
    rho, u, p = cons_to_prim(rho, mom, E)

    # Compute speed of sound
    c = np.sqrt(gamma * p / rho)

    # Compute time step based on CFL condition
    # dt = CFL * dx / np.max(np.abs(u) + c)


    # k1 state
    rho0, mom0, E0 = apply_bcs(rho, mom, E)
    k1_rho, k1_mom, k1_E = rhs(rho0, mom0, E0, dt, blend)
    #print("k1 done")

    # k2 state
    rho2 = rho + 0.5 * dt * k1_rho
    mom2 = mom + 0.5 * dt * k1_mom
    E2   = E   + 0.5 * dt * k1_E
    rho2, mom2, E2 = apply_bcs(rho2, mom2, E2)
    k2_rho, k2_mom, k2_E = rhs(rho2, mom2, E2, dt/2, blend)
    #print("k2 done")

    # k3 state
    rho3 = rho + 0.5 * dt * k2_rho
    mom3 = mom + 0.5 * dt * k2_mom
    E3   = E   + 0.5 * dt * k2_E
    rho3, mom3, E3 = apply_bcs(rho3, mom3, E3)
    k3_rho, k3_mom, k3_E = rhs(rho3, mom3, E3, dt/2, blend)
    #print("k3 done")

    # k4 state
    rho4 = rho + dt * k3_rho
    mom4 = mom + dt * k3_mom
    E4   = E   + dt * k3_E
    rho4, mom4, E4 = apply_bcs(rho4, mom4, E4)
    k4_rho, k4_mom, k4_E = rhs(rho4, mom4, E4, dt, blend)
    #print("k4 done")

    # Final update
    rho = rho + dt * (k1_rho + 2*k2_rho + 2*k3_rho + k4_rho) / 6
    mom = mom + dt * (k1_mom + 2*k2_mom + 2*k3_mom + k4_mom) / 6
    E   = E   + dt * (k1_E   + 2*k2_E   + 2*k3_E   + k4_E) / 6

    rho, mom, E = apply_bcs(rho, mom, E)

    return rho, mom, E

def solution(rho, mom, E, t_final, dx, x):

    # Time integration loop
    t = 0.0
    step = 0

    rho_blend = rho.copy()
    mom_blend = mom.copy()
    E_blend = E.copy()

    rho_first = rho.copy()
    mom_first = mom.copy()
    E_first = E.copy()

    rho_WENO = rho.copy()
    mom_WENO = mom.copy()
    E_WENO = E.copy()

    while t < t_final:

        print(f"Time: {t:.6f}, Timestep Num: {step}")

        # Convert conservative -> primitive (just for negative check)
        rho_blend, u_blend, p_blend = cons_to_prim(rho_blend, mom_blend, E_blend)

        rho_first, u_first, p_first = cons_to_prim(rho_first, mom_first, E_first)

        rho_WENO, u_WENO, p_WENO = cons_to_prim(rho_WENO, mom_WENO, E_WENO)

        # Compute speed of sound (for each method)
        c_blend = np.sqrt(gamma * p_blend / rho_blend)
        c_first = np.sqrt(gamma * p_first / rho_first)
        c_WENO = np.sqrt(gamma * p_WENO / rho_WENO)

        # Compute time step based on CFL condition (for each method)
        dt_blend = CFL * dx / np.max(np.abs(u_blend) + c_blend)
        dt_first = CFL * dx / np.max(np.abs(u_first) + c_first)
        dt_WENO = CFL * dx / np.max(np.abs(u_WENO) + c_WENO)

        # Choose the smallest time step

        dt = min(dt_blend, dt_WENO, dt_first, t_final - t)

        # Check for unphysical states
        if np.any(rho_blend <= 0) or np.any(p_blend <= 0):
          print("[BLENDED] Unphysical starting state at time n", t, step)
          break

        if np.any(rho_WENO <= 0) or np.any(p_WENO <= 0):
          print("[WENO] Unphysical starting state at time n", t, step)
          break


        if np.any(rho_first <= 0) or np.any(p_first <= 0):
          print("[LAX] Unphysical starting state at time n", t, step)
          break

        # Time integration

        rho_blend, mom_blend, E_blend = time_integration(rho_blend, mom_blend, E_blend, True, dt, dx)

        rho_WENO, mom_WENO, E_WENO = WENO_solution(rho_WENO, mom_WENO, E_WENO, dt, dx)

        rho_first, mom_first, E_first = first_order_solution(rho_first, mom_first, E_first, dt, dx)

        # Convert to primitive variables to check for negative values
        rho_check_blend, u_check_blend, p_check_blend = cons_to_prim(rho_blend, mom_blend, E_blend)
        rho_check_WENO, u_check_WENO, p_check_WENO = cons_to_prim(rho_WENO, mom_WENO, E_WENO)
        rho_check_first, u_check_first, p_check_first = cons_to_prim(rho_first, mom_first, E_first)

        # Check for negative values on pressure and densty
        if np.any(rho_check_blend <= 0) or np.any(p_check_blend <= 0):
          print("Unphysical BLENDING final state at time", t, step)
          break

        if np.any(rho_check_WENO <= 0) or np.any(p_check_WENO <= 0):
          print("Unphysical WENO final state at time", t, step)
          break

        if np.any(rho_check_first <= 0) or np.any(p_check_first <= 0):
          print("Unphysical LAX final state at time", t, step)
          break


        t += dt
        step += 1

    # Save plot at current timestep
    plt.figure(figsize=(15, 5))


    # Convert to primitive variables for plotting
    rho_plot_blend, u_plot_blend, p_plot_blend = cons_to_prim(rho_blend, mom_blend, E_blend)
    rho_plot_WENO, u_plot_WENO, p_plot_WENO = cons_to_prim(rho_WENO, mom_WENO, E_WENO)
    rho_plot_first, u_plot_first, p_plot_first = cons_to_prim(rho_first, mom_first, E_first)

    # Density plot
    plt.subplot(1, 3, 1)
    plt.plot(x, rho_plot_blend, label=f"Blended Density (t={t:.3f})", color='r')
    plt.plot(x, rho_plot_WENO, label=f"WENO Density (t={t:.3f})", color='b')
    plt.plot(x, rho_plot_first, label=f"First Order Density (t={t:.3f})", color='y')
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Density Profile")
    plt.grid()
    plt.legend()

    # Velocity plot
    plt.subplot(1, 3, 2)
    plt.plot(x, u_plot_blend, label=f"Blended Velocity (t={t:.3f})", color='r')
    plt.plot(x, u_plot_WENO, label=f"WENO Velocity (t={t:.3f})", color='b')
    plt.plot(x, u_plot_first, label=f"First Order Velocity (t={t:.3f})", color='y')
    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.title("Velocity Profile")
    plt.grid()
    plt.legend()

    # Pressure plot
    plt.subplot(1, 3, 3)
    plt.plot(x, p_plot_blend, label=f"Blended Pressure (t={t:.3f})", color='r')
    plt.plot(x, p_plot_WENO, label=f"WENO Pressure (t={t:.3f})", color='b')
    plt.plot(x, p_plot_first, label=f"First Order Pressure (t={t:.3f})", color='y')
    plt.xlabel("x")
    plt.ylabel("Pressure")
    plt.title("Pressure Profile")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'step_{step:04d}.png'))
    plt.close()

    return rho_blend, mom_blend, E_blend, rho_WENO, mom_WENO, E_WENO, rho_first, mom_first, E_first

numCells = [32, 64, 128, 256, 512, 1024]

# Exact solution to compare against
rho_exact, mom_exact, E_exact, x_exact = exact_solution(10240)

errorL1_blend = np.zeros(len(numCells))
errorL1_WENO = np.zeros(len(numCells))
errorL1_first = np.zeros(len(numCells))
errorL2_blend = np.zeros(len(numCells))
errorL2_WENO = np.zeros(len(numCells))
errorL2_first = np.zeros(len(numCells))

# Setting up the comparison for the two methods
for cells in numCells:
    rho, mom, E, t_final, dx, x = setUp(cells)
    rho_blend, mom_blend, E_blend, rho_WENO, mom_WENO, E_WENO, rho_first, mom_first, E_first = solution(rho, mom, E, t_final, dx, x)

    rho_blend, u_blend, p_blend = cons_to_prim(rho_blend, mom_blend, E_blend)
    rho_WENO, u_WENO, p_WENO = cons_to_prim(rho_WENO, mom_WENO, E_WENO)
    rho_exact, u_exact, p_exact = cons_to_prim(rho_exact, mom_exact, E_exact)
    rho_first, u_first, p_first = cons_to_prim(rho_first, mom_first, E_first)

    # This whole step is just to line up the different indicies
    # Also we only calculate error over the inner cells, not the ghosts
    physical = slice(3, -3)

    rho_ref = np.interp(
        x[physical],
        x_exact[3:-3],
        rho_exact[3:-3]
    )

    # Getting L1 and L2 error for each of the methods

    errorL1_blend[numCells.index(cells)] = np.mean(np.abs(rho_blend[physical] - rho_ref))
    errorL1_WENO[numCells.index(cells)] = np.mean(np.abs(rho_WENO[physical] - rho_ref))
    errorL1_first[numCells.index(cells)] = np.mean(np.abs(rho_first[physical] - rho_ref))

    errorL2_blend[numCells.index(cells)] = np.sqrt(np.mean((rho_blend[physical] - rho_ref)**2))
    errorL2_WENO[numCells.index(cells)] = np.sqrt(np.mean((rho_WENO[physical] - rho_ref)**2))
    errorL2_first[numCells.index(cells)] = np.sqrt(np.mean((rho_first[physical] - rho_ref)**2))

# Plotting the error vs. number of cells

plt.figure(figsize=(10, 5))
plt.plot(numCells, errorL1_blend, label='Blended', color='r')
plt.plot(numCells, errorL1_WENO, label='WENO5', color='b')
plt.plot(numCells, errorL1_first, label='First Order', color='y')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Grid Points, N')
plt.ylabel('L1 Error')
plt.title('L1 Error vs. Total Grid Points')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, 'L1error_plot.png'))
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(numCells, errorL2_blend, label='Blended', color='r')
plt.plot(numCells, errorL2_WENO, label='WENO5', color='b')
plt.plot(numCells, errorL2_first, label='First Order', color='y')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Grid Points, N')
plt.ylabel('L2 Error')
plt.title('L2 Error vs. Total Grid Points')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, 'L2error_plot.png'))
plt.show()
plt.close()
