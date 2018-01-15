'''
Created on Jul 2, 2012

@author: mkraus
'''

import h5py
import numpy as np


class Diagnostics(object):
    '''
    classdocs
    '''


    def __init__(self, hdf5_file):
        '''
        Constructor
        '''

        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        assert self.hdf5 != None
        
        self.eps_plot = 1E-12

        
        self.tGrid = self.hdf5['t'][:].flatten()
        self.xGrid = self.hdf5['x'][:]
        self.yGrid = self.hdf5['y'][:]
        
        self.nt = len(self.tGrid)-1
        
        self.nx = self.hdf5.attrs["grid.nx"]
        self.ny = self.hdf5.attrs["grid.ny"]
        self.n  = self.nx * self.ny
        
        self.ht = self.hdf5.attrs["grid.ht"]
        self.hx = self.hdf5.attrs["grid.hx"]
        self.hy = self.hdf5.attrs["grid.hy"]

        self.Lx = self.hdf5.attrs["grid.Lx"]
        self.Ly = self.hdf5.attrs["grid.Ly"]
        
        self.de = self.hdf5.attrs["initial_data.skin_depth"]
        
        assert self.nx == len(self.xGrid)
        assert self.ny == len(self.yGrid)
        
        assert np.allclose(self.Lx, (self.xGrid[-1] - self.xGrid[0]) + (self.xGrid[1] - self.xGrid[0]), rtol=1e-08, atol=1e-10)
        assert np.allclose(self.Ly, (self.yGrid[-1] - self.yGrid[0]) + (self.yGrid[1] - self.yGrid[0]), rtol=1e-08, atol=1e-10)
        
        assert np.allclose(self.hx, self.xGrid[1] - self.xGrid[0], rtol=1e-08, atol=1e-10)
        assert np.allclose(self.hy, self.yGrid[1] - self.yGrid[0], rtol=1e-08, atol=1e-10)
        
        self.tMin = self.tGrid[ 1]
        self.tMax = self.tGrid[-1]
        self.xMin = self.xGrid[ 0]
        self.xMax = self.xGrid[-1]
        self.yMin = self.yGrid[ 0]
        self.yMax = self.yGrid[-1]
        
        
        print("")
        print("nt = %i" % (self.nt))
        print("nx = %i" % (self.nx))
        print("ny = %i" % (self.ny))
        print("")
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hy = %f" % (self.hy))
        print("")
        print("de = %f" % (self.de))
        print("")
        print("")
        print("tGrid:")
        print(self.tGrid)
        print("")
        print("xGrid:")
        print(self.xGrid)
        print("")
        print("yGrid:")
        print(self.yGrid)
        print("")
        
        
        self.A  = np.zeros((self.nx, self.ny))
        self.J  = np.zeros((self.nx, self.ny))
        self.P  = np.zeros((self.nx, self.ny))
        self.O  = np.zeros((self.nx, self.ny))
        self.X  = np.zeros((self.nx, self.ny))
        
        self.Bx = np.zeros((self.nx, self.ny))
        self.By = np.zeros((self.nx, self.ny))
        self.Vx = np.zeros((self.nx, self.ny))
        self.Vy = np.zeros((self.nx, self.ny))
        
        self.m_energy   = 0.0
        self.k_energy   = 0.0
        self.i_energy   = 0.0
        self.energy     = 0.0
        self.psi_l2     = 0.0
        self.c_helicity = 0.0
        self.m_helicity = 0.0
        self.circulation= 0.0
        
        self.energy_init      = 0.0
        self.psi_l2_init      = 0.0
        self.c_helicity_init  = 0.0
        self.m_helicity_init  = 0.0
        self.circulation_init = 0.0
        
        self.energy_error     = 0.0
        self.psi_l2_error     = 0.0
        self.c_helicity_error = 0.0
        self.m_helicity_error = 0.0
        self.circulation_error= 0.0
        
        self.plot_energy     = False
        self.plot_psi_l2     = False
        self.plot_c_helicity = False
        self.plot_m_helicity = False
        self.plot_circulation= False
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
        
    def read_from_hdf5(self, iTime):
        self.A[:,:]  = self.hdf5['A' ][iTime,:,:].T
        self.J[:,:]  = self.hdf5['J' ][iTime,:,:].T
        self.P[:,:]  = self.hdf5['P' ][iTime,:,:].T
        self.O[:,:]  = self.hdf5['O' ][iTime,:,:].T

        self.X[:,:]  = self.A + self.J * self.de**2
        
        self.Bx[:,:] = self.hdf5['Bx'][iTime,:,:].T
        self.By[:,:] = self.hdf5['By'][iTime,:,:].T
        self.Vx[:,:] = self.hdf5['Vx'][iTime,:,:].T
        self.Vy[:,:] = self.hdf5['Vy'][iTime,:,:].T
        
    
    def update_invariants(self, iTime):
        
        
        self.m_energy   = np.sum( self.A * self.J )
        self.k_energy   = np.sum( self.P * self.O )
        self.i_energy   = np.sum( self.J * self.J )
        self.psi_l2     = np.sum( self.X * self.X )
        self.c_helicity = np.sum( self.X * self.O )
        self.m_helicity = np.sum( self.X )
        self.circulation= np.sum( self.O )
        
        self.m_energy *= 0.5 * self.hx * self.hy
        self.k_energy *= 0.5 * self.hx * self.hy
        self.i_energy *= 0.5 * self.hx * self.hy * self.de**2
        self.psi_l2         *= self.hx * self.hy
        self.c_helicity     *= self.hx * self.hy
        self.m_helicity     *= self.hx * self.hy
        self.circulation    *= self.hx * self.hy
        
        self.energy   = self.m_energy + self.k_energy + self.i_energy
    
        
        if iTime == 0:
            self.energy_init      = self.energy
            self.psi_l2_init      = self.psi_l2
            self.c_helicity_init  = self.c_helicity
            self.m_helicity_init  = self.m_helicity
            self.circulation_init = self.circulation
            
            self.energy_error     = 0.0
            self.psi_l2_error     = 0.0
            self.c_helicity_error = 0.0
            self.m_helicity_error = 0.0
            self.circulation_error= 0.0
        
            if np.abs(self.energy_init) < self.eps_plot:
                self.plot_energy = True
            
            if np.abs(self.psi_l2_init) < self.eps_plot:
                self.plot_psi_l2 = True
            
            if np.abs(self.c_helicity_init) < self.eps_plot:
                self.plot_c_helicity = True
            
            if np.abs(self.m_helicity_init) < self.eps_plot:
                self.plot_m_helicity = True
            
            if np.abs(self.circulation_init) < self.eps_plot:
                self.plot_circulation= True
            
            print("")
            print("Initial Energy:            %e" % self.energy_init)
            print("Initial L2 Norm of Psi:    %e" % self.psi_l2_init)
            print("Initial Circulation:       %e" % self.circulation_init)
            print("Initial Magnetic Helicity: %e" % self.m_helicity_init)
            print("Initial Cross Helicity:    %e" % self.c_helicity_init)
            print("")
            
        else:
            self.energy_error     = ( self.energy     - self.energy_init     ) / self.energy_init
            self.psi_l2_error     = ( self.psi_l2     - self.psi_l2_init     ) / self.psi_l2_init
            self.c_helicity_error = ( self.c_helicity - self.c_helicity_init ) / self.c_helicity_init
            self.m_helicity_error = ( self.m_helicity - self.m_helicity_init ) / self.m_helicity_init
            self.circulation_error= ( self.circulation- self.circulation_init) / self.circulation_init
        
