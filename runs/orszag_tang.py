
import numpy as np


def magnetic_A(x, y, Lx, Ly):
    return np.cos(2.*y) - 2. * np.cos(x) 
     
def velocity_P(x, y, Lx, Ly):
    return 2. * np.sin(y) - 2. * np.cos(x) 



def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
