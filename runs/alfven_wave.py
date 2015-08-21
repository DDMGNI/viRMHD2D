
import numpy as np


def magnetic_A(x, y, Lx, Ly):
    return - np.sin(np.pi * x) + np.cos(np.pi * y)
 
def velocity_P(x, y, Lx, Ly):
    return - np.sin(np.pi * x)



def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
