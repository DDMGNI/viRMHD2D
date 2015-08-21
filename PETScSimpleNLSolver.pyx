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


cdef class PETScSolver(object):
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
        
        self.ht_inv = 1. / ht
        self.hx_inv = 1. / hx
        self.hy_inv = 1. / hy
        
        
        # create history vector
        self.Xp = self.da4.createGlobalVec()
        self.Xh = self.da4.createGlobalVec()
        
        # create local vectors
        self.localXp = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, da4, nx, ny, ht, hx, hy)
        
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ap = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Jp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Op = xp[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ah = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Jh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Oh = xh[...][:,:,3]
        
        cdef double[:,:] A_ave = 0.5 * (Ap + Ah)
        cdef double[:,:] J_ave = 0.5 * (Jp + Jh)
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        
        
        cdef np.float64_t time_fac = 1.0 / (16. * self.ht)
        cdef np.float64_t arak_fac = 0.5 / (12. * self.hx * self.hy)
        
        cdef np.float64_t lapx_fac = 1.0 / self.hx**2
        cdef np.float64_t lapy_fac = 1.0 / self.hy**2
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                
                row.index = (i,j)
                
                # magnetic potential
                # dA/dt + [P, A] = - R_J
                row.field = 0
                
                # dA/dt + [P, dA] 
                col.field = 0
                for index, value in [
                        ((i-1, j-1), 1. * time_fac + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), 2. * time_fac + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                                   + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), 1. * time_fac + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), 2. * time_fac - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                                   - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 4. * time_fac),
                        ((i,   j+1), 2. * time_fac + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                                   + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), 1. * time_fac - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), 2. * time_fac - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                                   - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), 1. * time_fac - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
                    ]:
  
                    col.index = index
                    A.setValueStencil(row, col, value)
                  
                  
                # + [dP, A]
                col.field = 2
                for index, value in [
                        ((i-1, j-1), - (A_ave[ix-1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     - (A_ave[ix-1, jx+1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (A_ave[ix,   jx+1] - A_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     + (A_ave[ix+1, jx-1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     - (A_ave[ix+1, jx+1] - A_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (A_ave[ix+1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     + (A_ave[ix+1, jx+1] - A_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (A_ave[ix,   jx+1] - A_ave[ix+1, jx  ]) * arak_fac),
                    ]:
  
                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # current density
                # - Delta A - J = - R_A
                row.field = 1
                
                if i == 0 and j == 0:
                    A.setValueStencil(row, row, 1.)
                    
                else:
                    # - Delta A 
                    col.field = 0
                    for index, value in [
                        ((i,   j-1),                 - 1. * lapy_fac),
                        ((i-1, j  ), - 1. * lapx_fac                ),
                        ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                        ((i+1, j  ), - 1. * lapx_fac                ),
                        ((i,   j+1),                 - 1. * lapy_fac),
                        ]:
                        
                        col.index = index
                        A.setValueStencil(row, col, value)
                
                    # - J
                    col.field = 1
                    for index, value in [
                            ((i,   j  ), 1.),
                        ]:
                        
                        col.index = index
                        A.setValueStencil(row, col, - value)
                
                
                
                # streaming function
                # - Delta P - O = - R_P
                row.field = 2
                
                if i == 0 and j == 0:
                    A.setValueStencil(row, row, 1.)

                else:
                    # - Delta P
                    col.field = 2
                    for index, value in [
                        ((i,   j-1),                 - 1. * lapy_fac),
                        ((i-1, j  ), - 1. * lapx_fac                ),
                        ((i,   j  ), + 2. * lapx_fac + 2. * lapy_fac),
                        ((i+1, j  ), - 1. * lapx_fac                ),
                        ((i,   j+1),                 - 1. * lapy_fac),
                        ]:
                        
                        col.index = index
                        A.setValueStencil(row, col, value)
                
                    # - O
                    col.field = 3
                    for index, value in [
                            ((i,   j  ), 1.),
                        ]:
                        
                        col.index = index
                        A.setValueStencil(row, col, - value)
                
                
                
                # vorticity
                # dO/dt + [P, O] + [J, A] = - R_O
                row.field = 3
                
                # + [J, dA]
                col.field = 0
                for index, value in [
                        ((i-1, j-1), + (J_ave[ix-1, jx  ] - J_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (J_ave[ix,   jx+1] - J_ave[ix,   jx-1]) * arak_fac \
                                     + (J_ave[ix-1, jx+1] - J_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (J_ave[ix,   jx+1] - J_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (J_ave[ix+1, jx  ] - J_ave[ix-1, jx  ]) * arak_fac \
                                     - (J_ave[ix+1, jx-1] - J_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), + (J_ave[ix+1, jx  ] - J_ave[ix-1, jx  ]) * arak_fac \
                                     + (J_ave[ix+1, jx+1] - J_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (J_ave[ix+1, jx  ] - J_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (J_ave[ix,   jx+1] - J_ave[ix,   jx-1]) * arak_fac \
                                     - (J_ave[ix+1, jx+1] - J_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (J_ave[ix,   jx+1] - J_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                
                # + [dJ, A]
                col.field = 1
                for index, value in [
                        ((i-1, j-1), - (A_ave[ix-1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     - (A_ave[ix-1, jx+1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (A_ave[ix,   jx+1] - A_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     + (A_ave[ix+1, jx-1] - A_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (A_ave[ix+1, jx  ] - A_ave[ix-1, jx  ]) * arak_fac \
                                     - (A_ave[ix+1, jx+1] - A_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (A_ave[ix+1, jx  ] - A_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (A_ave[ix,   jx+1] - A_ave[ix,   jx-1]) * arak_fac \
                                     + (A_ave[ix+1, jx+1] - A_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (A_ave[ix,   jx+1] - A_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # + [dP, O]
                col.field = 2
                for index, value in [
                        ((i-1, j-1), - (O_ave[ix-1, jx  ] - O_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), - (O_ave[ix,   jx+1] - O_ave[ix,   jx-1]) * arak_fac \
                                     - (O_ave[ix-1, jx+1] - O_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), - (O_ave[ix,   jx+1] - O_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), + (O_ave[ix+1, jx  ] - O_ave[ix-1, jx  ]) * arak_fac \
                                     + (O_ave[ix+1, jx-1] - O_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j+1), - (O_ave[ix+1, jx  ] - O_ave[ix-1, jx  ]) * arak_fac \
                                     - (O_ave[ix+1, jx+1] - O_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), + (O_ave[ix+1, jx  ] - O_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), + (O_ave[ix,   jx+1] - O_ave[ix,   jx-1]) * arak_fac \
                                     + (O_ave[ix+1, jx+1] - O_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), + (O_ave[ix,   jx+1] - O_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
                # dO/dt + [P, dO] 
                col.field = 3
                for index, value in [
                        ((i-1, j-1), 1. * time_fac + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), 2. * time_fac + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                                   + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), 1. * time_fac + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), 2. * time_fac - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                                   - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 4. * time_fac),
                        ((i,   j+1), 2. * time_fac + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                                   + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), 1. * time_fac - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), 2. * time_fac - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                                   - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), 1. * time_fac - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
                    ]:

                    col.index = index
                    A.setValueStencil(row, col, value)
                
                
        A.assemble()
        
    
    
    def snes_mult(self, SNES snes, Vec X, Vec Y):
        self.mult(X, Y)
        
    
    def mult(self, Vec X, Vec Y):
        cdef np.int64_t i, j
        cdef np.int64_t ix, iy, jx, jy
        cdef np.int64_t xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,       self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[np.float64_t, ndim=3] y  = self.da4.getVecArray(Y)[...]
        
        cdef np.ndarray[np.float64_t, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[np.float64_t, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ap = xp[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Jp = xp[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Pp = xp[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Op = xp[...][:,:,3]
        
        cdef np.ndarray[np.float64_t, ndim=2] Ah = xh[...][:,:,0]
        cdef np.ndarray[np.float64_t, ndim=2] Jh = xh[...][:,:,1]
        cdef np.ndarray[np.float64_t, ndim=2] Ph = xh[...][:,:,2]
        cdef np.ndarray[np.float64_t, ndim=2] Oh = xh[...][:,:,3]
        
        cdef double[:,:] A_ave = 0.5 * (Ap + Ah)
        cdef double[:,:] J_ave = 0.5 * (Jp + Jh)
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        
        
        
        
        for i in np.arange(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in np.arange(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # magnetic potential
                y[iy, jy, 0] = \
                             + self.derivatives.dt(Ap, ix, jx) \
                             - self.derivatives.dt(Ah, ix, jx) \
                             + self.derivatives.arakawa(P_ave, A_ave, ix, jx)
                
                # current density
                y[iy, jy, 1] = \
                             - self.derivatives.laplace(Ap, ix, jx) \
                             - Jp[ix, jx]
                
                # streaming function
                if i == 0 and j == 0:
                    y[iy, jy, 2] = Pp[ix,jx]
                
                else:
                    y[iy, jy, 2] = \
                                 - self.derivatives.laplace(Pp, ix, jx) \
                                 - Op[ix, jx]
                
                # vorticity
                y[iy, jy, 3] = \
                             + self.derivatives.dt(Op, ix, jx) \
                             - self.derivatives.dt(Oh, ix, jx) \
                             + self.derivatives.arakawa(P_ave, O_ave, ix, jx) \
                             + self.derivatives.arakawa(J_ave, A_ave, ix, jx)

