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


cdef class PETScMatrix(object):
    '''
    The ScipySparse class implements a couple of solvers for the Vlasov-Poisson system
    built on top of the SciPy Sparse package.
    '''
    
    def __init__(self, DMDA da1, DMDA da4,
                 np.uint64_t nx, np.uint64_t ny,
                 np.float64_t ht, np.float64_t hx, np.float64_t hy):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        self.da4 = da4
        
        # grid
        self.nx = nx
        self.ny = ny
        
        self.ht = ht
        self.hx = hx
        self.hy = hy
        
        
        # create history vector
        self.Xh = self.da4.createGlobalVec()
        
        # create local vectors
        self.localB  = da4.createLocalVec()
        self.localX  = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, da4, nx, ny, ht, hx, hy)
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ah = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Jh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Oh = xh[...][:,:,3]
        
        
        cdef np.float64_t ave_fac  = 1.0 / 16.
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hy)
#        cdef np.float64_t lapx_fac = 0.25 / self.hx**2
#        cdef np.float64_t lapy_fac = 0.25 / self.hy**2
        cdef np.float64_t lapx_fac = self.hx**2
        cdef np.float64_t lapy_fac = self.hy**2
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+1
            
            for j in np.arange(ys, ye):
                jx = j-ys+1
                
                row.index = (i,j)
                
                # magnetic potential
                # dA_t + 0.5 * ([Ph, A] + [P, Ah]) = 0
                row.field = 0
                
                # dA_t + [Ph, A]
                col.field = 0
                for index, value in [
                        ((i-1, j-1), 1. * time_fac - (Ph[ix-1, jx  ] - Ph[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), 2. * time_fac - (Ph[ix,   jx+1] - Ph[ix,   jx-1]) * arak_fac \
                                                   - (Ph[ix-1, jx+1] - Ph[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), 1. * time_fac - (Ph[ix,   jx+1] - Ph[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), 2. * time_fac + (Ph[ix+1, jx  ] - Ph[ix-1, jx  ]) * arak_fac \
                                                   + (Ph[ix+1, jx-1] - Ph[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 4. * time_fac),
                        ((i,   j+1), 2. * time_fac - (Ph[ix+1, jx  ] - Ph[ix-1, jx  ]) * arak_fac \
                                                   - (Ph[ix+1, jx+1] - Ph[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), 1. * time_fac + (Ph[ix+1, jx  ] - Ph[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), 2. * time_fac + (Ph[ix,   jx+1] - Ph[ix,   jx-1]) * arak_fac \
                                                   + (Ph[ix+1, jx+1] - Ph[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), 1. * time_fac + (Ph[ix,   jx+1] - Ph[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # + [P, Ah]
                col.field = 2
                for index, value in [
                        ((i-1, j-1), + (Ah[ix-1, jx  ] - Ah[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (Ah[ix,   jx+1] - Ah[ix,   jx-1]) * arak_fac \
                                     + (Ah[ix-1, jx+1] - Ah[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (Ah[ix,   jx+1] - Ah[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (Ah[ix+1, jx  ] - Ah[ix-1, jx  ]) * arak_fac \
                                     - (Ah[ix+1, jx-1] - Ah[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), + (Ah[ix+1, jx  ] - Ah[ix-1, jx  ]) * arak_fac \
                                     + (Ah[ix+1, jx+1] - Ah[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (Ah[ix+1, jx  ] - Ah[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (Ah[ix,   jx+1] - Ah[ix,   jx-1]) * arak_fac \
                                     - (Ah[ix+1, jx+1] - Ah[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (Ah[ix,   jx+1] - Ah[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # current density
                # J - Delta A = 0
                row.field = 1
                
                # - Delta A 
                col.field = 0
#                for index, value in [
#                        ((i-1, j-1), - 1. * lapx_fac - 1. * lapy_fac),
#                        ((i,   j-1), + 2. * lapx_fac - 2. * lapy_fac),
#                        ((i+1, j-1), - 1. * lapx_fac - 1. * lapy_fac),
#                        ((i-1, j  ), - 2. * lapx_fac + 2. * lapy_fac),
#                        ((i,   j  ), + 4. * lapx_fac + 4. * lapy_fac),
#                        ((i+1, j  ), - 2. * lapx_fac + 2. * lapy_fac),
#                        ((i-1, j+1), - 1. * lapx_fac - 1. * lapy_fac),
#                        ((i,   j+1), + 2. * lapx_fac - 2. * lapy_fac),
#                        ((i+1, j+1), - 1. * lapx_fac - 1. * lapy_fac),
#                    ]:
                for index, value in [
                        ((i-1, j  ), - 1. * lapx_fac),
                        ((i+1, j  ), - 1. * lapx_fac),
                        ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                        ((i,   j-1), - 1. * lapy_fac),
                        ((i,   j+1), - 1. * lapy_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # J
                col.field = 1
#                for index, value in [
#                        ((i-1, j-1), 1. * ave_fac),
#                        ((i,   j-1), 2. * ave_fac),
#                        ((i+1, j-1), 1. * ave_fac),
#                        ((i-1, j  ), 2. * ave_fac),
#                        ((i,   j  ), 4. * ave_fac),
#                        ((i+1, j  ), 2. * ave_fac),
#                        ((i-1, j+1), 1. * ave_fac),
#                        ((i,   j+1), 2. * ave_fac),
#                        ((i+1, j+1), 1. * ave_fac),
#                    ]:
                for index, value in [
                        ((i,   j  ), 1.),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # streaming function
                # O - Delta P = - R_P
                row.field = 2
                
                # - Delta P
                col.field = 2
#                for index, value in [
#                        ((i-1, j-1), - 1. * lapx_fac - 1. * lapy_fac),
#                        ((i,   j-1), + 2. * lapx_fac - 2. * lapy_fac),
#                        ((i+1, j-1), - 1. * lapx_fac - 1. * lapy_fac),
#                        ((i-1, j  ), - 2. * lapx_fac + 2. * lapy_fac),
#                        ((i,   j  ), + 4. * lapx_fac + 4. * lapy_fac),
#                        ((i+1, j  ), - 2. * lapx_fac + 2. * lapy_fac),
#                        ((i-1, j+1), - 1. * lapx_fac - 1. * lapy_fac),
#                        ((i,   j+1), + 2. * lapx_fac - 2. * lapy_fac),
#                        ((i+1, j+1), - 1. * lapx_fac - 1. * lapy_fac),
#                    ]:
                for index, value in [
                        ((i-1, j  ), - 1. * lapx_fac),
                        ((i+1, j  ), - 1. * lapx_fac),
                        ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                        ((i,   j-1), - 1. * lapy_fac),
                        ((i,   j+1), - 1. * lapy_fac),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # O
                col.field = 3
#                for index, value in [
#                        ((i-1, j-1), 1. * ave_fac),
#                        ((i,   j-1), 2. * ave_fac),
#                        ((i+1, j-1), 1. * ave_fac),
#                        ((i-1, j  ), 2. * ave_fac),
#                        ((i,   j  ), 4. * ave_fac),
#                        ((i+1, j  ), 2. * ave_fac),
#                        ((i-1, j+1), 1. * ave_fac),
#                        ((i,   j+1), 2. * ave_fac),
#                        ((i+1, j+1), 1. * ave_fac),
#                    ]:
                for index, value in [
                        ((i,   j  ), 1.),
                    ]:
                    
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # vorticity
                # dO_t + 0.5 * ([Ph, O] + [P, Oh]) + 0.5 * ([J, Ah] + [Jh, A]) = 0
                row.field = 3
                
                # dO_t + [Ph, O] 
                col.field = 3
                for index, value in [
                        ((i-1, j-1), 1. * time_fac - (Ph[ix-1, jx  ] - Ph[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), 2. * time_fac - (Ph[ix,   jx+1] - Ph[ix,   jx-1]) * arak_fac \
                                                   - (Ph[ix-1, jx+1] - Ph[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), 1. * time_fac - (Ph[ix,   jx+1] - Ph[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), 2. * time_fac + (Ph[ix+1, jx  ] - Ph[ix-1, jx  ]) * arak_fac \
                                                   + (Ph[ix+1, jx-1] - Ph[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 4. * time_fac),
                        ((i,   j+1), 2. * time_fac - (Ph[ix+1, jx  ] - Ph[ix-1, jx  ]) * arak_fac \
                                                   - (Ph[ix+1, jx+1] - Ph[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), 1. * time_fac + (Ph[ix+1, jx  ] - Ph[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), 2. * time_fac + (Ph[ix,   jx+1] - Ph[ix,   jx-1]) * arak_fac \
                                                   + (Ph[ix+1, jx+1] - Ph[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), 1. * time_fac + (Ph[ix,   jx+1] - Ph[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # + [P, Oh]
                col.field = 2
                for index, value in [
                        ((i-1, j-1), + (Oh[ix-1, jx  ] - Oh[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (Oh[ix,   jx+1] - Oh[ix,   jx-1]) * arak_fac \
                                     + (Oh[ix-1, jx+1] - Oh[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (Oh[ix,   jx+1] - Oh[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (Oh[ix+1, jx  ] - Oh[ix-1, jx  ]) * arak_fac \
                                     - (Oh[ix+1, jx-1] - Oh[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), + (Oh[ix+1, jx  ] - Oh[ix-1, jx  ]) * arak_fac \
                                     + (Oh[ix+1, jx+1] - Oh[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (Oh[ix+1, jx  ] - Oh[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (Oh[ix,   jx+1] - Oh[ix,   jx-1]) * arak_fac \
                                     - (Oh[ix+1, jx+1] - Oh[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (Oh[ix,   jx+1] - Oh[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # + [J, Ah]
                col.field = 1
                for index, value in [
                        ((i-1, j-1), + (Ah[ix-1, jx  ] - Ah[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (Ah[ix,   jx+1] - Ah[ix,   jx-1]) * arak_fac \
                                     + (Ah[ix-1, jx+1] - Ah[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (Ah[ix,   jx+1] - Ah[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (Ah[ix+1, jx  ] - Ah[ix-1, jx  ]) * arak_fac \
                                     - (Ah[ix+1, jx-1] - Ah[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), + (Ah[ix+1, jx  ] - Ah[ix-1, jx  ]) * arak_fac \
                                     + (Ah[ix+1, jx+1] - Ah[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (Ah[ix+1, jx  ] - Ah[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (Ah[ix,   jx+1] - Ah[ix,   jx-1]) * arak_fac \
                                     - (Ah[ix+1, jx+1] - Ah[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (Ah[ix,   jx+1] - Ah[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # + [Jh, A]
                col.field = 0
                for index, value in [
                        ((i-1, j-1), - (Jh[ix-1, jx  ] - Jh[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (Jh[ix,   jx+1] - Jh[ix,   jx-1]) * arak_fac \
                                     - (Jh[ix-1, jx+1] - Jh[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (Jh[ix,   jx+1] - Jh[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (Jh[ix+1, jx  ] - Jh[ix-1, jx  ]) * arak_fac \
                                     + (Jh[ix+1, jx-1] - Jh[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (Jh[ix+1, jx  ] - Jh[ix-1, jx  ]) * arak_fac \
                                     - (Jh[ix+1, jx+1] - Jh[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (Jh[ix+1, jx  ] - Jh[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (Jh[ix,   jx+1] - Jh[ix,   jx-1]) * arak_fac \
                                     + (Jh[ix+1, jx+1] - Jh[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (Jh[ix,   jx+1] - Jh[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
