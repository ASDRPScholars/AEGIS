![aegis logo](https://github.com/user-attachments/assets/72ae8cc6-a51a-44c3-8afa-9c06978b691e)

---
<p align="center">
  <i align="center">An Adaptive Edge-Guided Blended-Flux Scheme for the 1D Euler Equations.</i>
</p>

![aegis banner 2](https://github.com/user-attachments/assets/3830634e-8890-4e53-a2d7-55098efa84fa)

---
**Built by the DeGrendele Simulation Lab**
Aditya Kaul, Naga Chintalapati, Vunal Jinasundera, Arnav Krishnan, Advika Singh, Claire Wang, Jasmine Lindstrorm, Vipanchi Rawat


## Summary
This repository contains numerical solvers for the 1D Euler equations of gas dynamics. It implements multiple numerical methods and includes scripts for both standard benchmark testing (shock tube problems) and formal convergence studies.

## Files Overview

### `1D_Shock-Tube_Tests.py`
This script is designed to test and compare different numerical schemes against standard benchmark problems for the 1D Euler equations. It primarily is used for testing schemes for shock capturing, but contains other problems as well.

**Features:**
- **Numerical Methods Included:**
  - **First-Order Lax-Friedrichs:** A robust but dissipative first-order method.
  - **WENO5 (Weighted Essentially Non-Oscillatory):** A 5th-order accurate scheme utilizing global Lax-Friedrichs flux splitting paired with 4th-order Runge-Kutta (RK4) time integration.
  - **Blended Scheme:** A hybrid method that utilizes an extended JST edge sensor as a shock detector to dynamically switch/blend between high-order polynomial reconstructions (e.g., MUSCL-MC, 4th-order, and 5th-order) in smooth regions and dissipative methods near shocks.
- **Initial Conditions (Benchmarks):**
  Configurable via the `IC_type` variable, allowing simulation of:
  - `sod`: Sod Shock Tube
  - `shu-osher`: Shu-Osher Problem (shock interacting with density perturbations)
  - `lax_tube`: Lax Shock Tube
  - `woodward`: Woodward-Colella blast wave (Work In Progress)
  - `Einfeldt`: Strong expansion/rarefaction test
  - `Gauss`: Smooth Gaussian density pulse
- **Output:** 
  Generates and saves profiles of Density, Velocity, and Pressure comparing the different methods against a high-resolution "exact" reference solution. Plots are saved to the `results_lf/` directory.

### `1dEulerCONVERGENCE.py`
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
python 1dEulerTESTING.py
python 1dEulerCONVERGENCE.py
```

Outputs will automatically be generated and stored in a local `results_lf/` directory created at runtime.
