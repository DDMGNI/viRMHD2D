
import numpy as np


def magnetic_A(x, y, Lx, Ly):
    return 2. * np.cos(x) - np.cos(2.*y)  
     
def velocity_P(x, y, Lx, Ly):
    return 2. * np.cos(x) - 2. * np.sin(y) 



def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
