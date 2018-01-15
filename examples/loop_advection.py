
import numpy as np

u0 = np.sqrt(5.)
A0 = 1.E-3
R0 = 0.3
th = np.arctan(0.5)

vx = u0 * np.cos(th)
vy = u0 * np.sin(th)


def magnetic_A(x, y, Lx, Ly):
    r = np.sqrt(x**2 + y**2)
     
    if r < R0:
        A = A0 * (R0 - r)
    else:
        A = 0.
      
    return A
 
 
def velocity_P(x, y, Lx, Ly):
    return vx*y - vy*x


def current_perturbation(x, y, Lx, Ly):
    return 0.

def vorticity_perturbation(x, y, Lx, Ly):
    return 0.
