'''
Created on Apr 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from PETScDerivatives import PETScDerivatives


cdef class PETScOhmsLaw(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, object da1, int nx, int ny,
                       double ht, double hx, double hy):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        self.arakawa_fac = 0.5 * self.ht * self.hx_inv * self.hy_inv / 12.
        
        # create history vector
        self.Ah = self.da1.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        
        # create local vectors
        self.localAp = da1.createLocalVec()
        self.localAh = da1.createLocalVec()
        self.localPh = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, nx, ny, ht, hx, hy)
        
        
    
    def update_history(self, Vec A, Vec P):
        A.copy(self.Ah)
        P.copy(self.Ph)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        self.da1.globalToLocal(self.Ph, self.localPh)
        
        cdef double[:,:] P_ave = self.da1.getVecArray(self.localPh)[...]
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        row.field = 0
        col.field = 0
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                
                row.index = (i,j)
                
                # dA/dt + [P, dA] 
                for index, value in [
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * self.arakawa_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * self.arakawa_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * self.arakawa_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * self.arakawa_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * self.arakawa_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * self.arakawa_fac),
                        ((i,   j  ), self.ht_inv),
                        ((i,   j+1), + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * self.arakawa_fac \
                                     + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * self.arakawa_fac),
                        ((i+1, j-1), - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * self.arakawa_fac),
                        ((i+1, j  ), - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * self.arakawa_fac \
                                     - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * self.arakawa_fac),
                        ((i+1, j+1), - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * self.arakawa_fac),
                    ]:
  
                    col.index = index
                    A.setValueStencil(row, col, value)
                  
        
        A.assemble()
        
    
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.function(X, Y)
        
    
    def function(self, Vec X, Vec Y):
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        self.da1.globalToLocal(X,       self.localAp)
        self.da1.globalToLocal(self.Ah, self.localAh)
        self.da1.globalToLocal(self.Ph, self.localPh)
        
        cdef double[:,:] Ah = self.da1.getVecArray(self.localAh)[...]
        cdef double[:,:] Ap = self.da1.getVecArray(self.localAp)[...]
        
        cdef double[:,:] A_ave = 0.5 * (self.da1.getVecArray(self.localAp)[:,:] + self.da1.getVecArray(self.localAh)[:,:])
        cdef double[:,:] P_ave = self.da1.getVecArray(self.localPh)[...]
        cdef double[:,:] y     = self.da1.getVecArray(Y)[...]
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                jy = j-ys
                
                y[iy, jy] = \
                          + (Ap[ix,jx] - Ah[ix,jx] ) * self.ht_inv \
                          + self.derivatives.arakawa(P_ave, A_ave, ix, jx)

