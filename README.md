# AEGIS-MHD

AEGIS-MHD is a one-dimensional magnetohydrodynamics solver with three-dimensional velocity and magnetic-field components (1D3V). It applies the Adaptive Edge-Guided Interface Scheme (AEGIS) to capture shocks and other discontinuities while retaining higher-order accuracy in smooth regions.

This project builds upon the original one-dimensional AEGIS code I developed through the Aspiring Scholars Directed Research Program (ASDRP). I gratefully acknowledge ASDRP for providing the research environment and mentorship that supported the development of the original method.

## Numerical Methods

AEGIS-MHD currently uses:

- Fifth-order reconstruction in smooth regions
- MUSCL reconstruction with a monotonized-central limiter near discontinuities
- A JST-inspired edge sensor for shocks and magnetic discontinuities
- Adaptive blending between high- and low-order numerical fluxes
- The HLLD approximate Riemann solver for ideal MHD
- A local Lax–Friedrichs/Rusanov fallback flux
- Fourth-order Runge–Kutta time integration
- CFL-based adaptive timesteps
- Numba compilation for improved performance

The solver evolves the conservative variables:

$$
U =
\begin{bmatrix}
\rho &
\rho u &
\rho v &
\rho w &
E &
B_y &
B_z
\end{bmatrix}^{T}
$$

The longitudinal magnetic-field component, $B_x$, is treated as constant throughout the one-dimensional domain.

## Included Problems

The available initial conditions include:

- **Brio–Wu shock tube**
- **Shu–Osher shock interaction**

Additional MHD benchmark problems, including the Ryu–Jones shock tube, are planned.

Select the problem in `config.py`:

```python
IC_type = "brio-wu"
```

For the Brio–Wu problem, the solver automatically uses the standard ratio of specific heats:

```python
gamma = 2.0
```

## Project Structure

```text
AEGIS-MHD/
├── simulation.py   # Main simulation loop and result plotting
├── config.py       # Grid, physical constants, and variable conversions
├── ics.py          # Initial-condition definitions
├── rhs.py          # Reconstruction, shock detection, and RHS evaluation
├── flux.py         # Physical flux and HLLD flux calculations
├── time_step.py    # Runge–Kutta time integration
├── bcs.py          # Boundary-condition handling
└── README.md
```

## Requirements

AEGIS-MHD requires Python 3 and the following packages:

- NumPy
- Matplotlib
- Numba

Install them with:

```bash
python -m pip install numpy matplotlib numba
```

## Running the Solver

Configure the problem and numerical parameters in `config.py`, then run:

```bash
python simulation.py
```

Simulation plots are written to:

```text
results_lf/
```

Generated results and Python bytecode caches are excluded from version control.

## Current Status

AEGIS-MHD is under active development. The current implementation is intended for experimentation, verification, and continued development of the AEGIS method for ideal magnetohydrodynamics.

Current areas of development include:

- Validation against reference MHD benchmark solutions
- Additional MHD shock-tube problems
- Improved robustness for degenerate HLLD states
- Convergence and conservation testing
- Additional boundary conditions
- Performance optimization

## Acknowledgment

The original one-dimensional AEGIS method that motivated this project was developed by the author through ASDRP. AEGIS-MHD extends that work to the ideal magnetohydrodynamic equations.

ASDRP is gratefully acknowledged for supporting the research environment and mentorship through which the original project was developed.
