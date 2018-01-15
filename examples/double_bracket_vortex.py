
import numpy as np

omega_x0, omega_y0 = (np.pi, np.pi)
omega_ax, omega_ay = (0.6, 1.)
omega_amp = 1.        

psi_x0, psi_y0 = (np.pi, np.pi)
psi_ax, psi_ay = (1., 1.)
psi_amp = 1.


def magnetic_A(x, y, Lx, Ly):
    return psi_amp * np.exp( 2. * (np.cos(x - psi_x0) - 1.) / psi_ax**2 
                           + 2. * (np.cos(y - psi_y0) - 1.) / psi_ay**2 )

def velocity_O(x, y, Lx, Ly):
    return omega_amp * np.exp( 2. * (np.cos(x - omega_x0) - 1.) / omega_ax**2 
                             + 2. * (np.cos(y - omega_y0) - 1.) / omega_ay**2 )
