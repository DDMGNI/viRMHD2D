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
        
        
        self.tGrid = self.hdf5['t'][:,0,0]
        self.xGrid = self.hdf5['x'][:]
        self.yGrid = self.hdf5['y'][:]
        
        self.nt = len(self.tGrid)-1
        
        if self.nt == 1:
            self.ht = 0.
        else:
            self.ht = self.tGrid[2] - self.tGrid[1]
        
        self.Lx = (self.xGrid[-1] - self.xGrid[0]) + (self.xGrid[1] - self.xGrid[0])
        self.Ly = (self.yGrid[-1] - self.yGrid[0]) + (self.yGrid[1] - self.yGrid[0])
        
        self.nx = len(self.xGrid)
        self.ny = len(self.yGrid)
        self.n  = self.nx * self.ny
        
        self.hx = self.xGrid[1] - self.xGrid[0]
        self.hy = self.yGrid[1] - self.yGrid[0]
        
        self.tMin = self.tGrid[ 1]
        self.tMax = self.tGrid[-1]
        self.xMin = self.xGrid[ 0]
        self.xMax = self.xGrid[-1]
        self.yMin = self.yGrid[ 0]
        self.yMax = self.yGrid[-1]
        
        
        print("nt = %i (%i)" % (self.nt, len(self.tGrid)) )
        print("nx = %i" % (self.nx))
        print("ny = %i" % (self.ny))
        print
        print("ht = %f" % (self.ht))
        print("hx = %f" % (self.hx))
        print("hy = %f" % (self.hy))
        print
        print("tGrid:")
        print(self.tGrid)
        print
        print("xGrid:")
        print(self.xGrid)
        print
        print("yGrid:")
        print(self.yGrid)
        print
        
        
        self.A  = np.zeros((self.nx, self.ny))
        self.J  = np.zeros((self.nx, self.ny))
        self.P  = np.zeros((self.nx, self.ny))
        self.O  = np.zeros((self.nx, self.ny))
        
        self.Bx = np.zeros((self.nx, self.ny))
        self.By = np.zeros((self.nx, self.ny))
        self.Vx = np.zeros((self.nx, self.ny))
        self.Vy = np.zeros((self.nx, self.ny))
        
        self.m_energy   = 0.0
        self.k_energy   = 0.0
        self.energy     = 0.0
        self.c_helicity = 0.0
        self.m_helicity = 0.0
        
        self.energy_init      = 0.0
        self.c_helicity_init  = 0.0
        self.m_helicity_init  = 0.0
        
        self.energy_error     = 0.0
        self.c_helicity_error = 0.0
        self.m_helicity_error = 0.0
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
        
    def read_from_hdf5(self, iTime):
        self.A  = self.hdf5['A' ][iTime,:,:].T
        self.J  = self.hdf5['J' ][iTime,:,:].T
        self.P  = 0.5 * (self.hdf5['P' ][iTime,:,:] + self.hdf5['P' ][iTime+1,:,:]).T
        self.O  = 0.5 * (self.hdf5['O' ][iTime,:,:] + self.hdf5['O' ][iTime+1,:,:]).T
        
        self.Bx = self.hdf5['Bx'][iTime,:,:].T
        self.By = self.hdf5['By'][iTime,:,:].T
        self.Vx = self.hdf5['Vx'][iTime,:,:].T
        self.Vy = self.hdf5['Vy'][iTime,:,:].T
        
    
    def update_invariants(self, iTime):
        
        self.m_energy   = 0.0
        self.k_energy   = 0.0
        self.c_helicity = 0.0
        self.m_helicity = 0.0
        
        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                
                self.m_energy += self.A[ix,iy] * self.J[ix,iy]
                self.k_energy += self.P[ix,iy] * self.O[ix,iy]

                self.c_helicity += self.A[ix,iy] * self.O[ix,iy]
                self.m_helicity += self.A[ix,iy]
                
        
        self.m_energy *= 0.5 * self.hx * self.hy
        self.k_energy *= 0.5 * self.hx * self.hy
        
        self.c_helicity  *= self.hx * self.hy
        self.m_helicity  *= self.hx * self.hy
        
        self.energy   = self.m_energy + self.k_energy 
    
        
        if iTime == 0:
            self.energy_init      = self.energy
            self.c_helicity_init  = self.c_helicity
            self.m_helicity_init  = self.m_helicity
            
            self.energy_error     = 0.0
            self.c_helicity_error = 0.0
            self.m_helicity_error = 0.0
        
        else:
            if self.energy_init < 1E-10:
                self.energy_error = (self.energy   - self.energy_init)
            else:
                self.energy_error = (self.energy   - self.energy_init) / self.energy_init
            
            if self.c_helicity_init < 1E-10:
                self.c_helicity_error = (self.c_helicity - self.c_helicity_init)
            else:
                self.c_helicity_error = (self.c_helicity - self.c_helicity_init) / self.c_helicity_init
            
            if self.m_helicity_init < 1E-10:
                self.m_helicity_error = (self.m_helicity - self.m_helicity_init)
            else:
                self.m_helicity_error = (self.m_helicity - self.m_helicity_init) / self.m_helicity_init
        
