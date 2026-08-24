from numba import njit
import numpy as np
from config import gamma

@njit
# Function to compute fluxes analytically
def compute_flux(rho, mom_x, mom_y, mom_z, E, p_t, B_x, B_y, B_z):
    F = np.zeros((7, len(rho)))
    F[0, :] = mom_x
    F[1, :] = mom_x**2 / rho + p_t - B_x**2
    F[2, :] = mom_x*mom_y / rho - B_x*B_y
    F[3, :] = mom_x*mom_z / rho - B_x*B_z
    F[4, :] = (E + p_t)*mom_x / rho - B_x*(
        mom_x*B_x/rho + mom_y*B_y/rho + mom_z*B_z/rho
    )
    F[5, :] = mom_x*B_y/rho - mom_y*B_x/rho
    F[6, :] = mom_x*B_z/rho - mom_z*B_x/rho
    return F

@njit
def flux_solver(interface_states, B_x):
    # I'll just unpack these things to make it easier to work with for myself
    rho_L = interface_states[0]
    rho_R = interface_states[1]
    mom_x_L = interface_states[2]
    mom_x_R = interface_states[3]
    mom_y_L = interface_states[4]
    mom_y_R = interface_states[5]
    mom_z_L = interface_states[6]
    mom_z_R = interface_states[7]
    E_L = interface_states[8]
    E_R = interface_states[9]
    B_y_L = interface_states[10]
    B_y_R = interface_states[11]
    B_z_L = interface_states[12]
    B_z_R = interface_states[13]

    u_L = mom_x_L / rho_L
    u_R = mom_x_R / rho_R
    v_L = mom_y_L / rho_L
    v_R = mom_y_R / rho_R
    w_L = mom_z_L / rho_L
    w_R = mom_z_R / rho_R

    p_L = (gamma - 1) * (E_L - 0.5*rho_L*(u_L**2+v_L**2+w_L**2) - 0.5*(B_x**2+B_y_L**2+B_z_L**2))
    p_R = (gamma - 1) * (E_R - 0.5*rho_R*(u_R**2+v_R**2+w_R**2) - 0.5*(B_x**2+B_y_R**2+B_z_R**2))

    p_t_L = p_L + 0.5*(B_x**2 + B_y_L**2 + B_z_L**2)
    p_t_R = p_R + 0.5*(B_x**2 + B_y_R**2 + B_z_R**2)

    # Compute fluxes at every cell

    F_L = compute_flux(
        rho_L, mom_x_L, mom_y_L, mom_z_L, E_L, p_t_L, B_x, B_y_L, B_z_L
        )
    F_R = compute_flux(
        rho_R, mom_x_R, mom_y_R, mom_z_R, E_R, p_t_R, B_x, B_y_R, B_z_R
    )

    U_left = np.vstack(
        (rho_R[:-1], mom_x_R[:-1], mom_y_R[:-1], mom_z_R[:-1],
         E_R[:-1], B_y_R[:-1], B_z_R[:-1])
        )
    U_right  = np.vstack(
        (rho_L[1:],  mom_x_L[1:],  mom_y_L[1:],  mom_z_L[1:],
         E_L[1:],  B_y_L[1:],  B_z_L[1:])
        )

    # Compute speed of sound for BOTH sides
    # Compute speed of sound (adiabatic)
    c_s_R = np.sqrt(gamma * p_R / rho_R)

    # Compute Alfvenic speeds
    c_a_R = np.sqrt((B_x**2+B_y_R**2+B_z_R**2)/rho_R)
    c_ax_R = np.sqrt(B_x**2/rho_R)

    # Fastest magnetosonic wave
    c_f_R = np.sqrt(0.5*(
        (c_a_R**2+c_s_R**2) + np.sqrt((c_a_R**2+c_s_R**2)**2-4*(c_s_R**2)*c_ax_R**2)
        ))

    c_s_L = np.sqrt(gamma * p_L / rho_L)

    # Compute Alfvenic speeds
    c_a_L = np.sqrt((B_x**2+B_y_L**2+B_z_L**2)/rho_L)
    c_ax_L = np.sqrt(B_x**2/rho_L)

    # Fastest magnetosonic wave
    c_f_L = np.sqrt(0.5*(
        (c_a_L**2+c_s_L**2) + np.sqrt((c_a_L**2+c_s_L**2)**2-4*(c_s_L**2)*c_ax_L**2)
        ))

    # Realignment stuff
    cL = c_f_R[:-1]
    cR = c_f_L[1:]
    uL = u_R[:-1]
    uR = u_L[1:]
    vL = v_R[:-1]
    vR = v_L[1:]
    wL = w_R[:-1]
    wR = w_L[1:]
    FL = F_R[:, :-1]
    FR = F_L[:, 1:]
    rhoL = rho_R[:-1]
    rhoR = rho_L[1:]
    pL = p_R[:-1]
    pR = p_L[1:]
    ptL = p_t_R[:-1]
    ptR = p_t_L[1:]
    EL = E_R[:-1]
    ER = E_L[1:]
    B_yL = B_y_R[:-1]
    B_yR = B_y_L[1:]
    B_zL = B_z_R[:-1]
    B_zR = B_z_L[1:]

    SL = np.minimum(uL-cL, uR-cR)
    SR = np.maximum(uL+cL, uR+cR)

    A_L = rhoL*(SL-uL)
    A_R = rhoR*(SR-uR)

    SM = (A_R*uR-A_L*uL-ptR+ptL)/(A_R-A_L)
    p_star_t = (A_R*ptL-A_L*ptR+A_L*A_R*(uR-uL))/(A_R-A_L)

    # D = denominator, since it's common for many of the star states
    D_L = rhoL*(SL-uL)*(SL-SM) - B_x**2
    D_R = rhoR*(SR-uR)*(SR-SM) - B_x**2

    # Most of the star states
    rho_star_L = rhoL*(SL-uL)/(SL-SM)
    rho_star_R = rhoR*(SR-uR)/(SR-SM)
    u_star_L = SM
    u_star_R = SM
    v_star_L = vL - B_x*B_yL*(SM-uL)/D_L
    v_star_R = vR - B_x*B_yR*(SM-uR)/D_R
    w_star_L = wL - B_x*B_zL*(SM-uL)/D_L
    w_star_R = wR - B_x*B_zR*(SM-uR)/D_R
    B_y_star_L = (B_yL)*(rhoL*(SL-uL)**2-B_x**2)/D_L
    B_y_star_R = (B_yR)*(rhoR*(SR-uR)**2-B_x**2)/D_R
    B_z_star_L = (B_zL)*(rhoL*(SL-uL)**2-B_x**2)/D_L
    B_z_star_R = (B_zR)*(rhoR*(SR-uR)**2-B_x**2)/D_R

    # Setting stuff up for the E_star calculations
    uB_L = uL*B_x + vL*B_yL + wL*B_zL
    uB_R = uR*B_x + vR*B_yR + wR*B_zR
    uB_star_L = u_star_L*B_x + v_star_L*B_y_star_L + w_star_L*B_z_star_L
    uB_star_R = u_star_R*B_x + v_star_R*B_y_star_R + w_star_R*B_z_star_R

    # Set up star states for E
    E_star_L = (
        (SL-uL)*EL - ptL*uL + p_star_t*SM + B_x*(uB_L - uB_star_L)
    )/(SL-SM)
    E_star_R = (
        (SR-uR)*ER - ptR*uR + p_star_t*SM + B_x*(uB_R - uB_star_R)
    )/(SR-SM)

    S_star_L = SM - np.abs(B_x)/np.sqrt(rho_star_L)
    S_star_R = SM + np.abs(B_x)/np.sqrt(rho_star_R)

    # "staar" is spelled to denote double star by duplicating the "a"
    # Just used for shorthand
    # Used for denominator since it's common for a lot of these formulas
    D_staar = np.sqrt(rho_star_L) + np.sqrt(rho_star_R)

    u_staar = SM

    v_staar = (
        np.sqrt(rho_star_L)*v_star_L + np.sqrt(rho_star_R)*v_star_R
    + (B_y_star_R-B_y_star_L)*np.sign(B_x)
    )/D_staar

    w_staar = (
        np.sqrt(rho_star_L)*w_star_L + np.sqrt(rho_star_R)*w_star_R
    + (B_z_star_R-B_z_star_L)*np.sign(B_x)
    )/D_staar

    B_y_staar = (
        np.sqrt(rho_star_L)*B_y_star_R + np.sqrt(rho_star_R)*B_y_star_L
    + np.sqrt(rho_star_L*rho_star_R)*(v_star_R-v_star_L)*np.sign(B_x)
    )/D_staar

    B_z_staar = (
        np.sqrt(rho_star_L)*B_z_star_R + np.sqrt(rho_star_R)*B_z_star_L
    + np.sqrt(rho_star_L*rho_star_R)*(w_star_R-w_star_L)*np.sign(B_x)
    )/D_staar

    # Setting stuff up for double star state Energy calculations
    uB_staar = u_staar*B_x + v_staar*B_y_staar + w_staar*B_z_staar

    # Energy double star state calculation
    E_staar_L = E_star_L - np.sign(B_x)*np.sqrt(rho_star_L)*(uB_star_L - uB_staar)
    E_staar_R = E_star_R + np.sign(B_x)*np.sqrt(rho_star_R)*(uB_star_R - uB_staar)

    # Putting all of the variables into the conservative variable vector
    U_star_L = np.vstack((
        rho_star_L, rho_star_L*u_star_L, rho_star_L*v_star_L, rho_star_L*w_star_L,
        E_star_L, B_y_star_L, B_z_star_L))
    U_star_R = np.vstack((
        rho_star_R, rho_star_R*u_star_R, rho_star_R*v_star_R, rho_star_R*w_star_R,
        E_star_R, B_y_star_R, B_z_star_R))
    U_staar_L = np.vstack((
        rho_star_L, rho_star_L*u_staar, rho_star_L*v_staar, rho_star_L*w_staar,
        E_staar_L, B_y_staar, B_z_staar))
    U_staar_R = np.vstack((
        rho_star_R, rho_star_R*u_staar, rho_star_R*v_staar, rho_star_R*w_staar,
        E_staar_R, B_y_staar, B_z_staar))

    # Moment of truth: calculating all the intermediate fluxes
    F_star_L = np.empty_like(FL)
    F_star_R = np.empty_like(FR)
    F_staar_L = np.empty_like(FL)
    F_staar_R = np.empty_like(FR)

    # Operations are done entirely with vectors
    # Calculated using the Rankine–Hugoniot jump condition
    F_star_L[:, :] = (
        FL[:, :] + SL[None, :]*(U_star_L[:, :] - U_left[:, :])
    )
    F_star_R[:, :] = (
        FR[:, :] + SR[None, :]*(U_star_R[:, :] - U_right[:, :])
    )
    F_staar_L[:, :] = (
        F_star_L[:, :] + S_star_L[None, :]*(U_staar_L[:, :] - U_star_L[:, :])
    )
    F_staar_R[:, :] = (
        F_star_R[:, :] + S_star_R[None, :]*(U_staar_R[:, :] - U_star_R[:, :])
    )

    # Final set up for HLLD flux vector and the conditions for flux calculation
    F_HLLD = np.empty_like(FL)

    mask_L = (SL >= 0.0)
    mask_R = (SR <= 0.0)
    mask_star_L = ((SL <= 0.0 ) &
                   (S_star_L >= 0.0))
    mask_star_R = ((SR >= 0.0 ) &
                   (S_star_R <= 0.0))
    mask_staar_L = ((S_star_L <= 0) &
                    (SM >= 0))
    mask_staar_R = ((S_star_R >= 0) &
                    (SM <= 0))

    # Final flux assignments
    F_HLLD[:, mask_L] = FL[:, mask_L]
    F_HLLD[:, mask_star_L] = F_star_L[:, mask_star_L]
    F_HLLD[:, mask_staar_L] = F_staar_L[:, mask_staar_L]
    F_HLLD[:, mask_staar_R] = F_staar_R[:, mask_staar_R]
    F_HLLD[:, mask_star_R] = F_star_R[:, mask_star_R]
    F_HLLD[:, mask_R] = FR[:, mask_R]

    return F_HLLD