
import numpy as np

u0  = 0.001
A0  = 1.
eps = 1.
k   = 1.


def magnetic_A(x, y, Lx, Ly):
    return A0 * np.log( np.cosh(2 * np.pi * k * x) + eps * np.cosh(2 * np.pi * k * y) )

def velocity_P(x, y, Lx, Ly):
    return u0 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
