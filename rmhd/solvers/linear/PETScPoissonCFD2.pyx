'''
Created on Apr 10, 2012

@author: Michael Kraus (michael.kraus@ipp.mpg.de)
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from rmhd.solvers.common.PETScDerivatives import PETScDerivatives


cdef class PETScPoisson(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, object da1,
                 int nx, int ny,
                 double hx, double hy):
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
        
        self.lapx_fac = 1. / self.hx**2
        self.lapy_fac = 1. / self.hy**2
        
        
        # create local vectors
        self.localX  = da1.createLocalVec()
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef int i, j, stencil
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da1.getRanges()
        stencil = self.da1.getStencilWidth()
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            ix = i-xs+stencil
            
            for j in range(ys, ye):
                jx = j-ys+stencil
                
                row.index = (i,j)
                
                for index, value in [
                    ((i,   j-1),                      - 1. * self.lapy_fac),
                    ((i-1, j  ), - 1. * self.lapx_fac                ),
                    ((i,   j  ), + 2. * self.lapx_fac + 2. * self.lapy_fac),
                    ((i+1, j  ), - 1. * self.lapx_fac                ),
                    ((i,   j+1),                      - 1. * self.lapy_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)

        A.assemble()
        

    @cython.boundscheck(False)
    def formRHS(self, Vec X, Vec B):
        X.copy(B)
        
#         cdef int i, j, stencil
#         cdef int ix, iy, jx, jy
#         cdef int xe, xs, ye, ys
#         
#         (xs, xe), (ys, ye) = self.da1.getRanges()
#         stencil = self.da1.getStencilWidth()
#         
#         self.da1.globalToLocal(X, self.localX)
#         
#         cdef double[:,:] b = self.da1.getVecArray(B)[...]
#         cdef double[:,:] x = self.da1.getVecArray(self.localX)[...]
#         
#         
#         for i in range(xs, xe):
#             ix = i-xs+stencil
#             iy = i-xs
#             
#             for j in range(ys, ye):
#                 jx = j-ys+stencil
#                 jy = j-ys
#                 
#                 b[iy,jy] = x[ix,jx]
                
    