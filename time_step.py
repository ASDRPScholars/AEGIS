from numba import njit
from rhs import rhs
from bcs import apply_bcs
from config import time_integral

@njit
def time_integration(rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z, dt, bc):

    # Initial states
    rho0 = rho.copy()
    mom_x0 = mom_x.copy()
    mom_y0 = mom_y.copy()
    mom_z0 = mom_z.copy()
    E0 = E.copy()
    B_y0 = B_y.copy()
    B_z0 = B_z.copy()

    if time_integral == "RK4":
        # This is the simpler RK4 time-integration method
        # This is less expensive but stability-focused than the SSPRK4 method

        rho0, mom_x0, mom_y0, mom_z0, E0, B_y0, B_z0 = apply_bcs(
            rho0, mom_x0, mom_y0, mom_z0, E0, B_y0, B_z0, bc
        )

        # k1 state
        k1_rho, k1_mom_x, k1_mom_y, k1_mom_z, k1_E, k1_B_y, k1_B_z = rhs(
            rho0, mom_x0, mom_y0, mom_z0, E0, B_x, B_y0, B_z0
        )

        # Set up states after first time-step
        rho1 = rho0 + 0.5 * dt * k1_rho
        mom_x1 = mom_x0 + 0.5 * dt * k1_mom_x
        mom_y1 = mom_y0 + 0.5 * dt * k1_mom_y
        mom_z1 = mom_z0 + 0.5 * dt * k1_mom_z
        E1 = E0 + 0.5 * dt * k1_E
        B_y1 = B_y0 + 0.5 * dt * k1_B_y
        B_z1 = B_z0 + 0.5 * dt * k1_B_z

        rho1, mom_x1, mom_y1, mom_z1, E1, B_y1, B_z1 = apply_bcs(
            rho1, mom_x1, mom_y1, mom_z1, E1, B_y1, B_z1, bc
        )

        # k2 state
        k2_rho, k2_mom_x, k2_mom_y, k2_mom_z, k2_E, k2_B_y, k2_B_z = rhs(
            rho1, mom_x1, mom_y1, mom_z1, E1, B_x, B_y1, B_z1
        )

        # Set up states after second time-step
        rho2 = rho0 + 0.5 * dt * k2_rho
        mom_x2 = mom_x0 + 0.5 * dt * k2_mom_x
        mom_y2 = mom_y0 + 0.5 * dt * k2_mom_y
        mom_z2 = mom_z0 + 0.5 * dt * k2_mom_z
        E2 = E0 + 0.5 * dt * k2_E
        B_y2 = B_y0 + 0.5 * dt * k2_B_y
        B_z2 = B_z0 + 0.5 * dt * k2_B_z

        rho2, mom_x2, mom_y2, mom_z2, E2, B_y2, B_z2 = apply_bcs(
            rho2, mom_x2, mom_y2, mom_z2, E2, B_y2, B_z2, bc
        )

        # k3 state
        k3_rho, k3_mom_x, k3_mom_y, k3_mom_z, k3_E, k3_B_y, k3_B_z = rhs(
            rho2, mom_x2, mom_y2, mom_z2, E2, B_x, B_y2, B_z2
        )

        # Set up states after the third time-step
        rho3 = rho0 + dt * k3_rho
        mom_x3 = mom_x0 + dt * k3_mom_x
        mom_y3 = mom_y0 + dt * k3_mom_y
        mom_z3 = mom_z0 + dt * k3_mom_z
        E3 = E0 + dt * k3_E
        B_y3 = B_y0 + dt * k3_B_y
        B_z3 = B_z0 + dt * k3_B_z

        rho3, mom_x3, mom_y3, mom_z3, E3, B_y3, B_z3 = apply_bcs(
            rho3, mom_x3, mom_y3, mom_z3, E3, B_y3, B_z3, bc
        )

        # k4 state
        k4_rho, k4_mom_x, k4_mom_y, k4_mom_z, k4_E, k4_B_y, k4_B_z = rhs(
            rho3, mom_x3, mom_y3, mom_z3, E3, B_x, B_y3, B_z3
        )

        # Final update
        rho = rho0 + (dt / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho)
        mom_x = mom_x0 + (dt / 6.0) * (k1_mom_x + 2.0 * k2_mom_x + 2.0 * k3_mom_x + k4_mom_x)
        mom_y = mom_y0 + (dt / 6.0) * (k1_mom_y + 2.0 * k2_mom_y + 2.0 * k3_mom_y + k4_mom_y)
        mom_z = mom_z0 + (dt / 6.0) * (k1_mom_z + 2.0 * k2_mom_z + 2.0 * k3_mom_z + k4_mom_z)
        E = E0 + (dt / 6.0) * (k1_E + 2.0 * k2_E + 2.0 * k3_E + k4_E)
        B_y = B_y0 + (dt / 6.0) * (k1_B_y + 2.0 * k2_B_y + 2.0 * k3_B_y + k4_B_y)
        B_z = B_z0 + (dt / 6.0) * (k1_B_z + 2.0 * k2_B_z + 2.0 * k3_B_z + k4_B_z)

    elif time_integral == "SSPRK4":
        drho, dmom_x, dmom_y, dmom_z, dE, dB_y, dB_z = rhs(rho0, mom_x0, mom_y0, mom_z0, E0, B_x, B_y0, B_z0)
        rho1 = rho0 + 0.39175222700392*dt*drho
        mom_x1 = mom_x0 + 0.39175222700392*dt*dmom_x
        mom_y1 = mom_y0 + 0.39175222700392*dt*dmom_y
        mom_z1 = mom_z0 + 0.39175222700392*dt*dmom_z
        E1 = E0 + 0.39175222700392*dt*dE
        B_y1 = B_y0 + 0.39175222700392*dt*dB_y
        B_z1 = B_z0 + 0.39175222700392*dt*dB_z
        rho1, mom_x1, mom_y1, mom_z1, E1, B_y1, B_z1 = apply_bcs(rho1, mom_x1, mom_y1, mom_z1, E1, B_y1, B_z1, bc)

        drho, dmom_x, dmom_y, dmom_z, dE, dB_y, dB_z = rhs(rho1, mom_x1, mom_y1, mom_z1, E1, B_x, B_y1, B_z1)
        rho2 = 0.44437049406734*rho0 + 0.55562950593266*rho1 +  0.36841059262959*dt*drho
        mom_x2 = 0.44437049406734*mom_x0 + 0.55562950593266*mom_x1 +  0.36841059262959*dt*dmom_x
        mom_y2 = 0.44437049406734*mom_y0 + 0.55562950593266*mom_y1 +  0.36841059262959*dt*dmom_y
        mom_z2 = 0.44437049406734*mom_z0 + 0.55562950593266*mom_z1 +  0.36841059262959*dt*dmom_z
        E2 = 0.44437049406734*E0 + 0.55562950593266*E1 +  0.36841059262959*dt*dE
        B_y2 = 0.44437049406734*B_y0 + 0.55562950593266*B_y1 +  0.36841059262959*dt*dB_y
        B_z2 = 0.44437049406734*B_z0 + 0.55562950593266*B_z1 +  0.36841059262959*dt*dB_z
        rho2, mom_x2, mom_y2, mom_z2, E2, B_y2, B_z2 = apply_bcs(rho2, mom_x2, mom_y2, mom_z2, E2, B_y2, B_z2, bc)

        drho, dmom_x, dmom_y, dmom_z, dE, dB_y, dB_z = rhs(rho2, mom_x2, mom_y2, mom_z2, E2, B_x, B_y2, B_z2)
        rho3 = 0.62010185138540*rho0 + 0.37989814861460*rho2 +  0.25189177424738*dt*drho
        mom_x3 = 0.62010185138540*mom_x0 + 0.37989814861460*mom_x2 +  0.25189177424738*dt*dmom_x
        mom_y3 = 0.62010185138540*mom_y0 + 0.37989814861460*mom_y2 +  0.25189177424738*dt*dmom_y
        mom_z3 = 0.62010185138540*mom_z0 + 0.37989814861460*mom_z2 +  0.25189177424738*dt*dmom_z
        E3 = 0.62010185138540*E0 + 0.37989814861460*E2 +  0.25189177424738*dt*dE
        B_y3 = 0.62010185138540*B_y0 + 0.37989814861460*B_y2 +  0.25189177424738*dt*dB_y
        B_z3 = 0.62010185138540*B_z0 + 0.37989814861460*B_z2 +  0.25189177424738*dt*dB_z
        rho3, mom_x3, mom_y3, mom_z3, E3, B_y3, B_z3 = apply_bcs(rho3, mom_x3, mom_y3, mom_z3, E3, B_y3, B_z3, bc)

        drho, dmom_x, dmom_y, dmom_z, dE, dB_y, dB_z = rhs(rho3, mom_x3, mom_y3, mom_z3, E3, B_x, B_y3, B_z3)
        rho4 = 0.17807995410773*rho0 +  0.82192004589227*rho3 +  0.54497475021237*dt*drho
        mom_x4 = 0.17807995410773*mom_x0 +  0.82192004589227*mom_x3 +  0.54497475021237*dt*dmom_x
        mom_y4 = 0.17807995410773*mom_y0 +  0.82192004589227*mom_y3 +  0.54497475021237*dt*dmom_y
        mom_z4 = 0.17807995410773*mom_z0 +  0.82192004589227*mom_z3 +  0.54497475021237*dt*dmom_z
        E4 = 0.17807995410773*E0 +  0.82192004589227*E3 +  0.54497475021237*dt*dE
        B_y4 = 0.17807995410773*B_y0 +  0.82192004589227*B_y3 +  0.54497475021237*dt*dB_y
        B_z4 = 0.17807995410773*B_z0 +  0.82192004589227*B_z3 +  0.54497475021237*dt*dB_z
        rho4, mom_x4, mom_y4, mom_z4, E4, B_y4, B_z4 = apply_bcs(rho4, mom_x4, mom_y4, mom_z4, E4, B_y4, B_z4, bc)

        # Need both the previous and new rhs for this final update
        # So two different sets of variables to store the rhs stages
        drho_4, dmom_x_4, dmom_y_4, dmom_z_4, dE_4, dB_y_4, dB_z_4 = rhs(rho4, mom_x4, mom_y4, mom_z4, E4, B_x, B_y4, B_z4)
        rho5 = (
            0.00683325884039*rho0 +  0.51723167208978*rho2
            + 0.12759831133288*rho3 + 0.34833675773694*rho4
            + 0.08460416338212*dt*drho + 0.22600748319395*dt*drho_4
        )
        mom_x5 = (
            0.00683325884039*mom_x0 +  0.51723167208978*mom_x2
            + 0.12759831133288*mom_x3 + 0.34833675773694*mom_x4
            + 0.08460416338212*dt*dmom_x + 0.22600748319395*dt*dmom_x_4
        )
        mom_y5 = (
            0.00683325884039*mom_y0 +  0.51723167208978*mom_y2
            + 0.12759831133288*mom_y3 + 0.34833675773694*mom_y4
            + 0.08460416338212*dt*dmom_y + 0.22600748319395*dt*dmom_y_4
        )
        mom_z5 = (
            0.00683325884039*mom_z0 +  0.51723167208978*mom_z2
            + 0.12759831133288*mom_z3 + 0.34833675773694*mom_z4
            + 0.08460416338212*dt*dmom_z + 0.22600748319395*dt*dmom_z_4
        )
        E5 = (
            0.00683325884039*E0 +  0.51723167208978*E2
            + 0.12759831133288*E3 + 0.34833675773694*E4
            + 0.08460416338212*dt*dE + 0.22600748319395*dt*dE_4
        )
        B_y5 = (
            0.00683325884039*B_y0 + 0.51723167208978*B_y2
            + 0.12759831133288*B_y3 + 0.34833675773694*B_y4
            + 0.08460416338212*dt*dB_y + 0.22600748319395*dt*dB_y_4
        )
        B_z5 = (
            0.00683325884039*B_z0 + 0.51723167208978*B_z2
            + 0.12759831133288*B_z3 + 0.34833675773694*B_z4
            + 0.08460416338212*dt*dB_z + 0.22600748319395*dt*dB_z_4
        )

        rho, mom_x, mom_y, mom_z, E, B_y, B_z = (
            rho5, mom_x5, mom_y5, mom_z5, E5, B_y5, B_z5
        )

    rho, mom_x, mom_y, mom_z, E, B_y, B_z = apply_bcs(
        rho, mom_x, mom_y, mom_z, E, B_y, B_z, bc
    )

    return rho, mom_x, mom_y, mom_z, E, B_y, B_z