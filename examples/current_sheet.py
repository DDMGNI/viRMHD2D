
import numpy as np


def magnetic_A(x, y, Lx, Ly):
    return - 1.29 / np.cosh(x)**2

def velocity_P(x, y, Lx, Ly):
    return 1.E-3 * (np.cos(x+y) - np.cos(x-y))


def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
