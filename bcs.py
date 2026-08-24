from numba import njit

@njit
def apply_bcs(rho, mom_x, mom_y, mom_z, E, B_y, B_z, bc):
    rho[:3] = rho[3]
    rho[-3:] = rho[-4]
    mom_y[:3] = mom_y[3]
    mom_y[-3:] = mom_y[-4]
    mom_z[:3] = mom_z[3]
    mom_z[-3:] = mom_z[-4]
    E[:3] = E[3]
    E[-3:] = E[-4]
    B_y[:3] = B_y[3]
    B_y[-3:] = B_y[-4]
    B_z[:3] = B_z[3]
    B_z[-3:] = B_z[-4]

    if bc == "normal":
        mom_x[:3] = mom_x[3]
        mom_x[-3:] = mom_x[-4]

    elif bc == "reflect":
        mom_x[:3] = -mom_x[3]
        mom_x[-3:] = -mom_x[-4]

    return rho, mom_x, mom_y, mom_z, E, B_y, B_z