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
        
        self.E_magnetic  = 0.0
        self.E_velocity  = 0.0
        
        self.energy   = 0.0
        self.helicity = 0.0
        
        self.E0       = 0.0
        self.H0       = 0.0
        
        self.L1_magnetic_0 = 0.0
        self.L1_velocity_0 = 0.0
        self.L2_magnetic_0 = 0.0
        self.L2_velocity_0 = 0.0
        
        self.E_error  = 0.0
        self.H_error  = 0.0
        
        
        self.read_from_hdf5(0)
        self.update_invariants(0)
        
        
        
    def read_from_hdf5(self, iTime):
        self.A  = self.hdf5['A' ][iTime,:,:].T
        self.J  = self.hdf5['J' ][iTime,:,:].T
        self.P  = self.hdf5['P' ][iTime,:,:].T
        self.O  = self.hdf5['O' ][iTime,:,:].T
        
        self.Bx = self.hdf5['Bx'][iTime,:,:].T
        self.By = self.hdf5['By'][iTime,:,:].T
        self.Vx = self.hdf5['Vx'][iTime,:,:].T
        self.Vy = self.hdf5['Vy'][iTime,:,:].T
        
    
    def update_invariants(self, iTime):
        
        self.E_magnetic = 0.0
        self.E_velocity = 0.0
        helicity = 0.0
        
        for ix in range(0, self.nx):
            ixp = (ix+1) % self.nx
            
            for iy in range(0, self.ny):
                iyp = (iy+1) % self.ny
                
                self.E_magnetic += self.A[ix,iy] * self.J[ix,iy]

# #                 self.E_magnetic += self.Bx[ixp,iy]**2 + self.Bx[ix,iy]**2 + self.Bx[ixp,iyp]**2 + self.Bx[ix,iyp]**2 \
# #                                  + self.By[ixp,iy]**2 + self.By[ix,iy]**2 + self.By[ixp,iyp]**2 + self.By[ix,iyp]**2
#                                 
# #                 self.E_magnetic += (self.Bx[ixp,iy] + self.Bx[ix,iy] + self.Bx[ixp,iyp] + self.Bx[ix,iyp])**2 \
# #                                  + (self.By[ixp,iy] + self.By[ix,iy] + self.By[ixp,iyp] + self.By[ix,iyp])**2
                                
#                 self.E_magnetic += (self.A[ixp,iy] - self.A[ix,iy] + self.A[ixp,iyp] - self.A[ix,iyp])**2 / self.hx**2 \
#                                  + (self.A[ix,iyp] - self.A[ix,iy] + self.A[ixp,iyp] - self.A[ixp,iy])**2 / self.hy**2
                
#                 self.E_magnetic += (self.A[ixp,iy ] - self.A[ix ,iy ])**2 / self.hx**2 \
#                                  + (self.A[ixp,iyp] - self.A[ix ,iyp])**2 / self.hx**2 \
#                                  + (self.A[ix, iyp] - self.A[ix ,iy ])**2 / self.hy**2 \
#                                  + (self.A[ixp,iyp] - self.A[ixp,iy ])**2 / self.hy**2
                                
                self.E_velocity += self.P[ix,iy] * self.O[ix,iy]

# #                 self.E_velocity += self.Vx[ixp,iy]**2 + self.Vx[ix,iy]**2 + self.Vx[ixp,iyp]**2 + self.Vx[ix,iyp]**2 \
# #                                  + self.Vy[ixp,iy]**2 + self.Vy[ix,iy]**2 + self.Vy[ixp,iyp]**2 + self.Vy[ix,iyp]**2
# 
# #                 self.E_velocity += (self.Vx[ixp,iy] + self.Vx[ix,iy] + self.Vx[ixp,iyp] + self.Vx[ix,iyp])**2 \
# #                                  + (self.Vy[ixp,iy] + self.Vy[ix,iy] + self.Vy[ixp,iyp] + self.Vy[ix,iyp])**2

#                 self.E_velocity += (self.P[ixp,iy] - self.P[ix,iy] + self.P[ixp,iyp] - self.P[ix,iyp])**2 / self.hx**2 \
#                                  + (self.P[ix,iyp] - self.P[ix,iy] + self.P[ixp,iyp] - self.P[ixp,iy])**2 / self.hy**2
                
#                 self.E_velocity += (self.P[ixp,iy ] - self.P[ix ,iy ])**2 / self.hx**2 \
#                                  + (self.P[ixp,iyp] - self.P[ix ,iyp])**2 / self.hx**2 \
#                                  + (self.P[ix, iyp] - self.P[ix ,iy ])**2 / self.hy**2 \
#                                  + (self.P[ixp,iyp] - self.P[ixp,iy ])**2 / self.hy**2
                
                helicity += self.A[ix,iy] * self.O[ix,iy]
#                 helicity += self.A[ix,iy] # cross helicity
                
#                 helicity += (self.A[ixp,iy] - self.A[ix,iy] + self.A[ixp,iyp] - self.A[ix,iyp]) / self.hx \
#                           * (self.P[ixp,iy] - self.P[ix,iy] + self.P[ixp,iyp] - self.P[ix,iyp]) / self.hx
#                            
#                 helicity += (self.A[ix,iyp] - self.A[ix,iy] + self.A[ixp,iyp] - self.A[ixp,iy]) / self.hy \
#                           * (self.P[ix,iyp] - self.P[ix,iy] + self.P[ixp,iyp] - self.P[ixp,iy]) / self.hy
                
#                 helicity += (self.A[ixp,iy ] - self.A[ix,iy ]) * (self.P[ixp,iy ] - self.P[ix,iy ]) / self.hx**2 \
#                           + (self.A[ixp,iyp] - self.A[ix,iyp]) * (self.P[ixp,iyp] - self.P[ix,iyp]) / self.hx**2
#                           
#                 helicity += (self.A[ix,iyp ] - self.A[ix,iy ]) * (self.P[ix,iyp ] - self.P[ix,iy ]) / self.hy**2 \
#                           + (self.A[ixp,iyp] - self.A[ixp,iy]) * (self.P[ixp,iyp] - self.P[ixp,iy]) / self.hy**2
                
                
#         self.E_magnetic *= 0.5 * 0.25 * self.hx * self.hy
#         self.E_velocity *= 0.5 * 0.25 * self.hx * self.hy
        
        self.E_magnetic *= 0.5 * self.hx * self.hy
        self.E_velocity *= 0.5 * self.hx * self.hy
        
        self.energy   = self.E_magnetic + self.E_velocity 
        self.helicity = helicity * self.hx * self.hy
    
        
        if iTime == 0:
            self.E0 = self.energy
            self.H0 = self.helicity
            
            self.E_error  = 0.0
            self.H_error  = 0.0
        
        else:
            self.E_error = (self.energy   - self.E0) / self.E0
            
            if self.H0 < 1E-10:
                self.H_error = (self.helicity - self.H0)
            else:
                self.H_error = (self.helicity - self.H0) / self.H0
        
