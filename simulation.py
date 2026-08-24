import os
import time
import numpy as np
import matplotlib.pyplot as plt
from ics import initialize_problems
from config import Nx, gamma, x, CFL, dx, IC_type, cons_to_prim
from time_step import time_integration

# Create output directory if it doesn't exist
output_dir = "results_lf"
os.makedirs(output_dir, exist_ok=True)

# Also have to initialize everything for the plots later
createPlot = True
fig = None
ax1 = ax2 = ax3 = ax4 = None
title_rho = title_u = title_p = title_B = None
line_rho = line_u = line_p = line_B = None

# Initialize the problem
rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z, t_final, bc = initialize_problems(IC_type, x)

t = 0.0
step = 0
total_time_AEGIS = 0.0
while t < t_final:
    # Convert conservative -> primitive
    rho, u, v, w, p, p_t, B_x, B_y, B_z = cons_to_prim(rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z)

    # Check for unphysical states
    if np.any(rho <= 0) or np.any(p <= 0):
      print("Unphysical starting state at time n", t, step)
      break

    # Compute speed of sound (adiabatic)
    c_s = np.sqrt(gamma * p / rho)

    # Compute Alfvenic speeds
    c_a = np.sqrt((B_x**2+B_y**2+B_z**2)/rho)
    c_ax = np.sqrt(B_x**2/rho)

    # Fastest magnetosonic wave
    c_f = np.sqrt(
        0.5*((c_a**2+c_s**2) + np.sqrt((c_a**2+c_s**2)**2-4*(c_s**2)*c_ax**2))
        )

    # Compute time step based on CFL condition
    dt = CFL * dx / np.max(np.abs(u) + c_f)

    dt = min(dt, t_final-t)

    print(f"Time: {t:.6f}, Timestep Num: {step}")

    # Time tracking enabled to check how long the method itself takes
    t_start = time.perf_counter()
    rho, mom_x, mom_y, mom_z, E, B_y, B_z = time_integration(rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z, dt, bc)
    total_time_AEGIS += time.perf_counter() - t_start

    if step % 50 == 0 or t >= t_final:
        # Convert conservative variables to primitive variables
        (
            rho_plot,
            u_plot,
            v_plot,
            w_plot,
            p_plot,
            p_t_plot,
            B_x_plot,
            B_y_plot,
            B_z_plot
        ) = cons_to_prim(
            rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z
        )

        if createPlot:

            # Creating the actual figure
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

            # Density plot
            line_rho, = ax1.plot(x, np.zeros_like(x), color='r', label="AEGIS-MHD")
            ax1.set_xlabel("x")
            ax1.set_ylabel("Density")
            title_rho = ax1.set_title("Density Profile (t=0.000)")
            ax1.grid()
            leg1 = ax1.legend()

            # Velocity plot
            line_u, = ax2.plot(x, np.zeros_like(x), color='r', label="AEGIS-MHD")
            ax2.set_xlabel("x")
            ax2.set_ylabel("X-Axis Velocity")
            title_u = ax2.set_title("X-Axis Velocity Profile (t=0.000)")
            ax2.grid()
            leg2 = ax2.legend()

            # Pressure plot
            line_p, = ax3.plot(x, np.zeros_like(x), color='r', label="AEGIS-MHD")
            ax3.set_xlabel("x")
            ax3.set_ylabel("Pressure")
            title_p = ax3.set_title("Pressure Profile (t=0.000)")
            ax3.grid()
            leg3 = ax3.legend()

            # Magnetic Field
            line_B, = ax4.plot(x, np.zeros_like(x), color='r', label="AEGIS-MHD")
            ax4.set_xlabel("x")
            ax4.set_ylabel("Y-Axis Magnetic Field")
            title_B = ax4.set_title("Y-Axis Magnetic Field Profile (t=0.000)")
            ax4.grid()
            leg4 = ax4.legend()

            # Not gonna close the graph to keep reusing it
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'step_{step:04d}.png'))

            createPlot = False

        else:
            line_rho.set_ydata(rho_plot)
            title_rho.set_text(f"Density Profile (t={t:.3f})")

            line_u.set_ydata(u_plot)
            title_u.set_text(f"X-Axis Velocity Profile (t={t:.3f})")

            line_p.set_ydata(p_plot)
            title_p.set_text(f"Pressure Profile (t={t:.3f})")

            line_B.set_ydata(B_y_plot)
            title_B.set_text(f"Y-Axis Magnetic Profile Profile (t={t:.3f})")

            ax1.relim(); ax1.autoscale_view()
            ax2.relim(); ax2.autoscale_view()
            ax3.relim(); ax3.autoscale_view()
            ax4.relim(); ax4.autoscale_view()

            fig.savefig(os.path.join(output_dir, f'step_{step:04d}.png'))

    t += dt
    step += 1
# Use the actual final step values
(rho_plot, u_plot, v_plot, w_plot, p_plot, p_t_plot,
B_x_plot, B_y_plot, B_z_plot) = cons_to_prim(
    rho, mom_x, mom_y, mom_z, E, B_x, B_y, B_z
)

line_rho.set_ydata(rho_plot)
title_rho.set_text(f"Density Profile (t={t:.3f})")

line_u.set_ydata(u_plot)
title_u.set_text(f"X-Axis Velocity Profile (t={t:.3f})")

line_p.set_ydata(p_plot)
title_p.set_text(f"Pressure Profile (t={t:.3f})")

line_B.set_ydata(B_y_plot)
title_B.set_text(f"Y-Axis Magnetic Profile Profile (t={t:.3f})")

ax1.relim(); ax1.autoscale_view()
ax2.relim(); ax2.autoscale_view()
ax3.relim(); ax3.autoscale_view()
ax4.relim(); ax4.autoscale_view()

# Final output is a pdf so that file resolutions are high
fig.savefig(os.path.join(output_dir, f'step_{step:04d}.pdf'))

print(f"Real time for AEGIS: {total_time_AEGIS:.4f} seconds")
