from numba import njit
import numpy as np
from config import gamma, dx, cons_to_prim, prim_to_cons
from flux import flux_solver, compute_flux

@njit
def reconstruct_5th_order(u):
    uL = (2*u[:-4] - 13*u[1:-3] + 47*u[2:-2] + 27*u[3:-1] - 3*u[4:])/60
    uR = (2*u[4:] - 13*u[3:-1] + 47*u[2:-2] + 27*u[1:-3] - 3*u[:-4])/60
    return uR, uL

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
def shock_Detector(q):

    # This is an extended form of the JST edge sensor

    sensor_num = np.abs(q[:-4] - 4*q[1:-3] + 6*q[2:-2] - 4*q[3:-1] + q[4:])
    sensor_den = (
        np.abs(q[:-4]) + np.abs(4*q[1:-3]) + np.abs(6*q[2:-2])
        + np.abs(4*q[3:-1]) + np.abs(q[4:])
        )

    # Epsilon scales so that it's always considered very small with respect
    # to the values it's being added to
    epsilon = 1e-12 * max(1.0, np.max(np.abs(q)))

    sensor = sensor_num/(sensor_den+epsilon)

    return sensor

@njit
def rhs(rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z):
    drho = np.zeros_like(rho)
    dmom_x = np.zeros_like(mom_x)
    dmom_y = np.zeros_like(mom_y)
    dmom_z = np.zeros_like(mom_z)
    dE = np.zeros_like(E)
    dB_y = np.zeros_like(B_y)
    dB_z = np.zeros_like(B_z)

    # Positivity floor used to trigger fallbacks
    positive_floor = 1e-12

    # Indexing stuff for later
    center = slice(2, -2)

    U = np.vstack((rho, mom_x, mom_y, mom_z, E, B_y, B_z))

    # Convert conservative variables to primitive variables
    rho, u, v, w, p, p_t, B_x, B_y, B_z = cons_to_prim(rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z)

    # Shock detector
    # Calculated over density, pressure, and non-constant magnetic fields
    # Pressure -> Shocks
    # Density -> Discontinuities in density not picked up by pressure (like contact discontinuities)
    # Magnetic Fields -> MHD-unique stuff, like Alfven discontinuities

    shockArray_p = shock_Detector(p)
    shockArray_rho = shock_Detector(rho)
    shockArray_B_y = shock_Detector(B_y)
    shockArray_B_z = shock_Detector(B_z)

    shockArray_standard = np.maximum(shockArray_p, shockArray_rho)
    shockArray_MHD = np.maximum(shockArray_B_y, shockArray_B_z)
    shockArray = np.maximum(shockArray_standard, shockArray_MHD)

    # MUSCL-MC Reconstruction

    rho_L_MC, rho_R_MC = reconstruct_MUSCL_MC(rho)
    u_L_MC, u_R_MC = reconstruct_MUSCL_MC(u)
    v_L_MC, v_R_MC = reconstruct_MUSCL_MC(v)
    w_L_MC, w_R_MC = reconstruct_MUSCL_MC(w)
    B_y_L_MC, B_y_R_MC = reconstruct_MUSCL_MC(B_y)
    B_z_L_MC, B_z_R_MC = reconstruct_MUSCL_MC(B_z)
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

    rho_L_MC, mom_x_L_MC, mom_y_L_MC, mom_z_L_MC, E_L_MC, B_x, B_y_L_MC, B_z_L_MC = prim_to_cons(
        rho_L_MC, u_L_MC, v_L_MC, w_L_MC, p_L_MC, B_x, B_y_L_MC, B_z_L_MC
    )

    rho_R_MC, mom_x_R_MC, mom_y_R_MC, mom_z_R_MC, E_R_MC, B_x, B_y_R_MC, B_z_R_MC = prim_to_cons(
        rho_R_MC, u_R_MC, v_R_MC, w_R_MC, p_R_MC, B_x, B_y_R_MC, B_z_R_MC
    )

    MUSCL_MC_interface_states = np.vstack((
        rho_L_MC, rho_R_MC,
        mom_x_L_MC, mom_x_R_MC,
        mom_y_L_MC, mom_y_R_MC,
        mom_z_L_MC, mom_z_R_MC,
        E_L_MC, E_R_MC,
        B_y_L_MC, B_y_R_MC,
        B_z_L_MC, B_z_R_MC
    ))

    F_half_MC = flux_solver(MUSCL_MC_interface_states, B_x)

    # Fifth-order primitive-variable reconstruction

    rho_L_5, rho_R_5 = reconstruct_5th_order(rho)
    u_L_5, u_R_5 = reconstruct_5th_order(u)
    v_L_5, v_R_5 = reconstruct_5th_order(v)
    w_L_5, w_R_5 = reconstruct_5th_order(w)
    B_y_L_5, B_y_R_5 = reconstruct_5th_order(B_y)
    B_z_L_5, B_z_R_5 = reconstruct_5th_order(B_z)
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

    rho_L_5, mom_x_L_5, mom_y_L_5, mom_z_L_5, E_L_5, B_x, B_y_L_5, B_z_L_5 = prim_to_cons(
        rho_L_5, u_L_5, v_L_5, w_L_5, p_L_5, B_x, B_y_L_5, B_z_L_5
    )
    rho_R_5, mom_x_R_5, mom_y_R_5, mom_z_R_5, E_R_5, B_x, B_y_R_5, B_z_R_5 = prim_to_cons(
        rho_R_5, u_R_5, v_R_5, w_R_5, p_R_5, B_x, B_y_R_5, B_z_R_5
    )

    fifth_order_interface_states = np.vstack((
        rho_L_5, rho_R_5,
        mom_x_L_5, mom_x_R_5,
        mom_y_L_5, mom_y_R_5,
        mom_z_L_5, mom_z_R_5,
        E_L_5, E_R_5,
        B_y_L_5, B_y_R_5,
        B_z_L_5, B_z_R_5
    ))

    F_half_5 = flux_solver(fifth_order_interface_states, B_x)

    # First-order local Rusanov flux
    F = compute_flux(rho, mom_x, mom_y, mom_z, E, p_t, B_x, B_y, B_z)

    # Compute speed of sound (adiabatic)
    c_s = np.sqrt(gamma * p / rho)

    # Compute Alfvenic speeds
    c_a = np.sqrt((B_x**2+B_y**2+B_z**2)/rho)
    c_ax = np.sqrt(B_x**2/rho)

    # Fastest magnetosonic wave
    c_f = np.sqrt(
        0.5*((c_a**2+c_s**2) + np.sqrt((c_a**2+c_s**2)**2-4*(c_s**2)*c_ax**2))
    )

    wave_speed = np.abs(u)+c_f

    a_face = np.maximum(
        wave_speed[2:-3],
        wave_speed[3:-2]
    )

    F_half_1 = (
        0.5 * (F[:, 2:-3] + F[:, 3:-2])
        - 0.5 * a_face[None, :]
          * (U[:, 3:-2] - U[:, 2:-3])
    )

    # Fifth order in smooth regions
    F_half_hi = F_half_5

    # Second/first-order flux blend
    F_half_lo = 0.75 * F_half_MC + 0.25 * F_half_1

    # Convert the cell-centered sensor to one shared value at each face
    edge_sensor = np.maximum(
        shockArray[:-1],
        shockArray[1:]
    )

    # Weibull cdf mapping for the weighting
    # Tuned values; these do not change between problems

    shape_parameter = 2
    scale_parameter = 0.02

    shockWeight = 1.0 - np.exp(
        -(edge_sensor / scale_parameter)**shape_parameter
    )

    # Blend fluxes before taking their divergence
    F_half = (
        (1.0 - shockWeight) * F_half_hi
        + shockWeight * F_half_lo
    )

    drho[3:-3] = -(F_half[0, 1:] - F_half[0, :-1]) / dx
    dmom_x[3:-3] = -(F_half[1, 1:] - F_half[1, :-1]) / dx
    dmom_y[3:-3] = -(F_half[2, 1:] - F_half[2, :-1]) / dx
    dmom_z[3:-3] = -(F_half[3, 1:] - F_half[3, :-1]) / dx
    dE[3:-3] = -(F_half[4, 1:] - F_half[4, :-1]) / dx
    dB_y[3:-3] = -(F_half[5, 1:] - F_half[5, :-1]) / dx
    dB_z[3:-3] = -(F_half[6, 1:] - F_half[6, :-1]) / dx

    return drho, dmom_x, dmom_y, dmom_z, dE, dB_y, dB_z