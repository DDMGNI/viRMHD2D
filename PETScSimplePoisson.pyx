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


cdef class PETScPoisson(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DMDA da1,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.hx = hx
        self.hy = hy
        
        
        # create local vectors
        self.localB  = da1.createLocalVec()
        self.localX  = da1.createLocalVec()
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        cdef np.float64_t lapx_fac = 1. / self.hx**2
        cdef np.float64_t lapy_fac = 1. / self.hy**2
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                
                row.index = (i,j)
                
                if i == 0 and j == 0:
                    A.setValueStencil(row, row, 1.)

                else:
                    for index, value in [
                        ((i,   j-1),                 - 1. * lapy_fac),
                        ((i-1, j  ), - 1. * lapx_fac                ),
                        ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                        ((i+1, j  ), - 1. * lapx_fac                ),
                        ((i,   j+1),                 - 1. * lapy_fac),
                        ]:
                        
                        col.index = index
                        A.setValueStencil(row, col, value)

        A.assemble()
        

    @cython.boundscheck(False)
    def formRHS(self, Vec X, Vec B):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        
        self.da1.globalToLocal(X, self.localX)
        
        cdef np.ndarray[np.float64_t, ndim=2] b = self.da1.getVecArray(B)[...]
        cdef np.ndarray[np.float64_t, ndim=2] x = self.da1.getVecArray(self.localX)[...]
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                b[iy,jy] = x[ix,jx]
                
    