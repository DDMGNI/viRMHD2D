'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport DMDA, Mat, Vec

from PETScDerivatives import PETScDerivatives


cdef class PETScVorticity(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DMDA da1, np.uint64_t nx, np.uint64_t ny,
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
        
        
        # create history vector
        self.Oh = self.da1.createGlobalVec()
        self.Pp = self.da1.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        self.Ah = self.da1.createGlobalVec()
        self.Jh = self.da1.createGlobalVec()
        
        # create local vectors
        self.localOp = da1.createLocalVec()
        self.localOh = da1.createLocalVec()
        self.localPp = da1.createLocalVec()
        self.localPh = da1.createLocalVec()
        self.localAh = da1.createLocalVec()
        self.localJh = da1.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, nx, ny, ht, hx, hy)
        
        
    
    def update_history(self, Vec O, Vec P, Vec A, Vec J):
        O.copy(self.Oh)
        P.copy(self.Ph)
        A.copy(self.Ah)
        J.copy(self.Jh)
        
    
    def update_streaming_function(self, Vec P):
        P.copy(self.Pp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(self.Pp, self.localPp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = self.da1.getVecArray(self.localPh)[...]
        
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        
        cdef double arak_fac = 0.5 * self.hx_inv * self.hy_inv / 12.
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        row.field = 0
        col.field = 0
        
        for i in range(xs, xe):
            ix = i-xs+2
            
            for j in range(ys, ye):
                jx = j-ys+2
                
                row.index = (i,j)
                
                # dO/dt + [P, dO] 
                for index, value in [
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), self.ht_inv),
                        ((i,   j+1), + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
    
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.function(X, Y)
        
    
    def function(self, Vec X, Vec Y):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X,       self.localOp)
        self.da1.globalToLocal(self.Oh, self.localOh)
        self.da1.globalToLocal(self.Pp, self.localPp)
        self.da1.globalToLocal(self.Ph, self.localPh)
        self.da1.globalToLocal(self.Ah, self.localAh)
        self.da1.globalToLocal(self.Jh, self.localJh)
        
        cdef np.ndarray[np.float64_t, ndim=2] Op = self.da1.getVecArray(self.localOp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Oh = self.da1.getVecArray(self.localOh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = self.da1.getVecArray(self.localPp)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = self.da1.getVecArray(self.localPh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Ah = self.da1.getVecArray(self.localAh)[...]
        cdef np.ndarray[np.float64_t, ndim=2] Jh = self.da1.getVecArray(self.localJh)[...]
        
        cdef double[:,:] A_ave = Ah
        cdef double[:,:] J_ave = Jh
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        
        cdef double[:,:] y     = self.da1.getVecArray(Y)[...]
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                y[iy, jy] = (Op[ix,jx] - Oh[ix,jx] ) * self.ht_inv \
                          + self.derivatives.arakawa(P_ave, O_ave, ix, jx) \
                          + self.derivatives.arakawa(J_ave, A_ave, ix, jx)

