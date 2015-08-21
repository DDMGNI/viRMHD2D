
import numpy as np

rho = 30.
r   = 0.5


def magnetic_A(x, y, Lx, Ly):
    return 0.0
 
def velocity_P(x, y, Lx, Ly):
    if y <= r: 
        return 0.05 * np.cos(2. * np.pi * x) / np.cosh(rho * (y - 0.25))**2
    else:
        return 0.05 * np.cos(2. * np.pi * x) / np.cosh(rho * (0.75 - y))**2


def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
