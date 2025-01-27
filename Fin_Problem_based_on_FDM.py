# This code solves a heat transfer (temperature distribution) from a fin with following given physical parameters (Problem 9 in :https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter23.06-Summary-and-Problems.html)
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Given parameters
L = 0.2  # Length of the fin
T0 = 475  # Temperature at x=0
TL = 290  # Temperature at x=L
Ts = 290  # Surrounding/Ambient temperature
hc = 40  # Convective heat transfer coefficient
P = 0.015  # Perimeter
epsilon = 0.4  # Emissivity
k = 240  # Thermal conductivity
Ac = 1.55e-5  # Cross-sectional area
sigma_SB = 5.67e-8  # Stefan-Boltzmann constant

# Derived parameters
alpha1 = hc * P / (k * Ac)
alpha2 = epsilon * sigma_SB * P / (k * Ac)

# Discretization
N = 100  # Number of grid points
dx = L / (N - 1)
x = np.linspace(0, L, N)

# Initialize temperature array
T = np.linspace(T0, TL, N)  # Initial guess for temperature distribution

# Function for the system of equations
def equations(T):
    T_new = T.copy()
    for i in range(1, N - 1):
        T_new[i] = (
            T[i - 1] - 2 * T[i] + T[i + 1]
        ) / dx**2 - alpha1 * (T[i] - Ts) - alpha2 * (T[i] ** 4 - Ts ** 4)
    T_new[0] = T[0] - T0  # Boundary condition at x=0
    T_new[-1] = T[-1] - TL  # Boundary condition at x=L
    return T_new

# Solve the nonlinear system
T_solution = fsolve(equations, T)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, T_solution, label="Temperature Distribution", color="blue",linestyle ="dashed",marker="x" )
plt.axhline(y=Ts, color="red", linestyle="--", label="Ambient Temp (Ts)")
plt.title("Temperature Distribution in the Pin Fin")
plt.xlabel("x [m]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.grid(True)
plt.show()
