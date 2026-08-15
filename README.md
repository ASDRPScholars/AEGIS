<p align="center">
  <img src="https://github.com/user-attachments/assets/d2ddba91-d0ba-4a3d-9b25-5823d2166de5" width="200" alt="aegis logo">
</p>



---
<p align="center">
  <i align="center">An Adaptive Edge-Guided Blended-Flux Scheme for the 1D Euler Equations.</i>
</p>

![aegis banner 2](https://github.com/user-attachments/assets/d51d66a2-487d-48b4-b08b-aa47e16913e8)

---
**Built by the DeGrendele Simulation Lab.**


Aditya Kaul*, Naga Chintalapati*, Vunal Jinasundera*, Arnav Krishnan, Advika Singh, Claire Wang, Jasmine Lindstrorm, Vipanchi Rawat

## Abstract
This work presents the Adaptive Edge-Guided Interface Scheme (**AEGIS**), a high-order shock-capturing scheme for the 1D Euler equations. **AEGIS** adaptively blends fifth-order and low-order numerical fluxes based on local flow smoothness. The proposed edge-guided blending method combines a modified Jameson–Schmidt–Turkel discontinuity sensor with a Weibull mapping. This approach uses both pressure and density indicators to detect discontinuities, while the mapping converts the resulting interface indicators into flux weights. **AEGIS** is evaluated on the Sod, Lax, and Shu–Osher shock-tube problems and a smooth Gaussian density-pulse advection problem. The $L_1$ and $L_2$ errors of **AEGIS** are compared with those of a first-order Lax–Friedrichs solver and a component-wise finite-difference implementation of the fifth-order Jiang–Shu weighted essentially non-oscillatory scheme (WENO5-JS) for the tested problems. Across the shock-tube problems, **AEGIS** reduced density-profile $L_1$ errors by $15.71–31.59$% and $L_2$ errors by $0.15–25.71$% relative to WENO5-JS. For the smooth advection problem, **AEGIS** exhibited approximately fifth-order convergence. The results demonstrate that **AEGIS** preserves high-order accuracy in smooth regions while robustly capturing discontinuities.

Key content: computational fluid dynamics, high-order methods, finite-volume methods, shock capturing, 1D Euler equations.

## Plots
![shu plot](https://github.com/user-attachments/assets/7abb89da-ba07-44fd-990b-e79bd415a40f)
![sod plot](https://github.com/user-attachments/assets/44408ed4-4720-451a-84e0-ace8ce2fea70)
![einfeldt plot](https://github.com/user-attachments/assets/6c3e3a43-4f64-40cb-8782-7f97238b8fc5)
![lax plot](https://github.com/user-attachments/assets/0cce8697-b233-4837-9f3f-b5056181004a)
![l2_eror](https://github.com/user-attachments/files/31105335/L2error_plot.5.pdf)

Shown at timestep t=1.8.


## Paper
MIT URTC 2026 Submission: [AEGIS_Conservation_Laws.pdf](https://github.com/user-attachments/files/30963491/AEGIS_Conservation_Laws.pdf)

---

## Solver Overview

### `1D_Shock-Tube_Tests.py`
This script is designed to test and compare different numerical schemes against standard benchmark problems for the 1D Euler equations. It primarily is used for testing schemes for shock capturing, but contains other problems as well.

**Features:**
- **Numerical Methods Included:**
  - **First-Order Lax-Friedrichs:** A robust but dissipative first-order method.
  - **WENO5 (Weighted Essentially Non-Oscillatory):** A 5th-order accurate scheme utilizing global Lax-Friedrichs flux splitting paired with 4th-order Runge-Kutta (RK4) time integration.
  - **Blended Scheme:** A hybrid method that utilizes an extended JST edge sensor as a shock detector to dynamically switch/blend between high-order polynomial reconstructions (e.g., MUSCL-MC, 4th-order, and 5th-order) in smooth regions and dissipative methods near shocks.
- **Initial Conditions (Benchmarks):**
  Configurable via the `IC_type` variable, allowing simulation of:
  - `sod`: Sod Shock-Tube
  - `shu-osher`: Shu-Osher Shock-Tube
  - `lax_tube`: Lax Shock-Tube
  - `woodward`: Woodward-Colella blast wave (Work In Progress)
  - `Einfeldt`: Strong expansion/rarefaction test
  - `Gauss`: Smooth Gaussian density pulse
- **Output:** 
  Generates and saves profiles of Density, Velocity, and Pressure comparing the different methods against a high-resolution "exact" reference solution. Plots are saved to the `results_lf/` directory.

### `1D_Convergence_Plot.py`
This script focuses on performing a formal grid convergence study to determine the actual numerical order of accuracy of the implemented schemes.

**Features:**
- **Smooth Problem Testing:** Typically utilizes the `Gauss` smooth initial condition, as convergence rates for high-order schemes must be measured on continuous solutions without shocks.
- **Grid Refinement:** Iterates the solution over a series of progressively finer grids (e.g., `N = 32, 64, 128, 256, 512, 1024`).
- **Error Calculation:** Computes a highly resolved reference solution (e.g., `N = 10240`) to act as the "exact" solution. Interpolates the lower-resolution results onto the exact grid and calculates $L_1$ and $L_2$ norms of the error.
- **Convergence Plots:** Produces log-log plots of $L_1$ and $L_2$ error versus the number of grid points. These plots visually demonstrate the convergence rates (slopes) of the First-Order, WENO5, and Blended methods, allowing validation of their theoretical order of accuracy. Output plots are saved to `results_lf/`.

## Dependencies
To run these scripts, you will need the following Python packages:
- `numpy`: Array manipulations and math operations.
- `matplotlib`: Plotting the results and convergence graphs.
- `numba`: Just-In-Time (JIT) compilation (via `@njit`) heavily used throughout both scripts to drastically speed up the numerical fluxes and time-stepping loops.

## Usage
Simply execute either Python script directly:

```bash
python 1D_Shock-Tube_Tests.py
python 1D_Convergence_Plot.py
```

Outputs will automatically be generated and stored in a local `results_lf/` directory created at runtime.

---
![cse](https://github.com/user-attachments/assets/d91a05a4-8a67-4879-8e39-fde85355afc9#gh-dark-mode-only)
