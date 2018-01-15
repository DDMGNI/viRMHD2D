
import numpy as np

def magnetic_J(x, y, Lx, Ly):
    return velocity_O(y, x, Lx, Ly)

def velocity_O(x, y, Lx, Ly):
    th = np.arctan2(y,x)
    r02 = 1. + 0.6 * np.cos(2*th)
    r2 = x*x + y*y
    
    if r2 < r02:
        return 1. - r2/r02
    else:
        return 0.
