
import numpy as np
from scipy.special import jv

lambdaR = 3.83170597020751231561
R       = 0.1
U       = 1.0
lam     = lambdaR / R

def magnetic_A(x, y, Lx, Ly):
    return 0.0
 
def velocity_P(x, y, Lx, Ly):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if r < R: 
        return 2. * lam * U * jv(1, lam * r) / jv(0, lambdaR) * np.cos(theta)
    else:
        return 0.


def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
