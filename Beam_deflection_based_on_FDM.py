# This code solves for deflection of a beam under UDL with given physical parameters (Problem 10 in :https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter23.06-Summary-and-Problems.html)
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Given parameters
L = 5  # Length of the beam (m)
y0 = 0  # Deflection at x=0
yL = 0  # Deflection at x=L
w0 = 15e3  # Uniformly distributed load (N/m)
EI = 1.8e7  # Flexural Rigidity (N-m^2)

# Discretization
N = 10  # Number of grid points
dx = L / (N - 1)
x = np.linspace(0, L, N)

# Initial guess for deflection
y_initial = np.zeros(N)

# Function for the system of equations
def equations(y):
    y_new = np.zeros_like(y)
    for i in range(1, N - 1):
        # Finite difference approximation of d^2y/dx^2
        d2y_dx2 = (y[i - 1] - 2 * y[i] + y[i + 1]) / dx**2
        # Nonlinear term: [1 + (dy/dx)^2]^(3/2)
        dy_dx = (y[i + 1] - y[i - 1]) / (2 * dx)
        nonlinear_term = (1 + dy_dx**2)**(3 / 2)
        # Discretized equation
        y_new[i] = EI * d2y_dx2 - 12 * w0 * (L * x[i] - x[i]**2) * nonlinear_term
    # Boundary conditions
    y_new[0] = y[0] - y0
    y_new[-1] = y[-1] - yL
    return y_new

# Solve the nonlinear system using fsolve
y_solution = fsolve(equations, y_initial)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_solution, label="Deflection", color="blue", linestyle="dashed", marker="o")
plt.title("Deflection of the Beam under Uniform Load")
plt.xlabel("x [m]")
plt.ylabel("Deflection [m]")
plt.legend()
plt.grid(True)
plt.show()
