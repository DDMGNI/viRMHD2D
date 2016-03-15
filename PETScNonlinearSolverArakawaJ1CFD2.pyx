# cython: profile=True
'''
Created on Apr 10, 2012

@author: mkraus
'''

cimport cython

import  numpy as np
cimport numpy as np

from petsc4py import PETSc

from petsc4py.PETSc cimport Mat, Vec

from PETScDerivatives import PETScDerivatives


cdef class PETScSolver(object):
    '''
    The PETScSolver class implements a nonlinear solver for the reduced MHD system
    built on top of the PETSc SNES module.
    '''
    
    def __init__(self, object da1, object da4,
                 int nx, int ny,
                 double ht, double hx, double hy,
                 object pc=None):
        '''
        Constructor
        '''
        
        # distributed arrays
        self.da1 = da1
        self.da4 = da4
        
        # preconditioner
        self.pc = pc
        
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
        self.Yd = self.da4.createGlobalVec()
        
        self.Ap = self.da1.createGlobalVec()
        self.Jp = self.da1.createGlobalVec()
        self.Pp = self.da1.createGlobalVec()
        self.Op = self.da1.createGlobalVec()
        
        self.Ah = self.da1.createGlobalVec()
        self.Jh = self.da1.createGlobalVec()
        self.Ph = self.da1.createGlobalVec()
        self.Oh = self.da1.createGlobalVec()
        
        # create working vector
        self.YA = self.da1.createGlobalVec()
        self.YJ = self.da1.createGlobalVec()
        self.YP = self.da1.createGlobalVec()
        self.YO = self.da1.createGlobalVec()
        
        self.Ad = self.da1.createGlobalVec()
        self.Jd = self.da1.createGlobalVec()
        self.Pd = self.da1.createGlobalVec()
        self.Od = self.da1.createGlobalVec()
        
        # create temporary vectors
        self.T  = self.da1.createGlobalVec()
        self.T1 = self.da1.createGlobalVec()
        self.T2 = self.da1.createGlobalVec()
        
        # create local vectors
        self.localXd = da4.createLocalVec()
        self.localXp = da4.createLocalVec()
        self.localXh = da4.createLocalVec()
        
        # create derivatives object
        self.derivatives = PETScDerivatives(da1, nx, ny, ht, hx, hy)
        
        
    
    def update_history(self, Vec X):
        X.copy(self.Xh)
        
#         x = self.da4.getVecArray(self.Xh)
#         a = self.da1.getVecArray(self.Ah)
#         j = self.da1.getVecArray(self.Jh)
#         p = self.da1.getVecArray(self.Ph)
#         o = self.da1.getVecArray(self.Oh)
#         
#         a[:,:] = x[:,:,0]
#         j[:,:] = x[:,:,1]
#         p[:,:] = x[:,:,2]
#         o[:,:] = x[:,:,3]
        
    
    def update_previous(self, Vec X):
        X.copy(self.Xp)
        
#         x = self.da4.getVecArray(self.Xp)
#         a = self.da1.getVecArray(self.Ap)
#         j = self.da1.getVecArray(self.Jp)
#         p = self.da1.getVecArray(self.Pp)
#         o = self.da1.getVecArray(self.Op)
#         
#         a[:,:] = x[:,:,0]
#         j[:,:] = x[:,:,1]
#         p[:,:] = x[:,:,2]
#         o[:,:] = x[:,:,3]
        
    
    @cython.boundscheck(False)
    def formMat(self, Mat A):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[double, ndim=2] Ap = xp[...][:,:,0]
        cdef np.ndarray[double, ndim=2] Jp = xp[...][:,:,1]
        cdef np.ndarray[double, ndim=2] Pp = xp[...][:,:,2]
        cdef np.ndarray[double, ndim=2] Op = xp[...][:,:,3]
        
        cdef np.ndarray[double, ndim=2] Ah = xh[...][:,:,0]
        cdef np.ndarray[double, ndim=2] Jh = xh[...][:,:,1]
        cdef np.ndarray[double, ndim=2] Ph = xh[...][:,:,2]
        cdef np.ndarray[double, ndim=2] Oh = xh[...][:,:,3]
        
        cdef double[:,:] A_ave = 0.5 * (Ap + Ah)
        cdef double[:,:] J_ave = 0.5 * (Jp + Jh)
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        
        
        cdef double arak_fac = 0.5 * self.ht * self.hx_inv * self.hy_inv / 12.
        cdef double lapx_fac = self.hx_inv**2
        cdef double lapy_fac = self.hy_inv**2
        
        
        A.zeroEntries()
        
        row = Mat.Stencil()
        col = Mat.Stencil()
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            
            for j in range(ys, ye):
                jx = j-ys+2
                
                row.index = (i,j)
                
                # magnetic potential
                # dA/dt + [P, A] = - R_J
                row.field = 0
                
                # dA/dt + [P, dA] 
                col.field = 0
                for index, value in [
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 1.),
                        ((i,   j+1), + (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     + (P_ave[ix+1, jx+1] - P_ave[ix-1, jx+1]) * arak_fac),
                        ((i+1, j-1), - (P_ave[ix+1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i+1, j  ), - (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     - (P_ave[ix+1, jx+1] - P_ave[ix+1, jx-1]) * arak_fac),
                        ((i+1, j+1), - (P_ave[ix,   jx+1] - P_ave[ix+1, jx  ]) * arak_fac),
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
                    A.setValueStencil(row, col, - value)
            
                # - J
                col.field = 1
                col.index = (i,j)
                A.setValueStencil(row, col, 1.)
                
                
                
                # streaming function
                # - Delta P - O = - R_P
                row.field = 2
                
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
                    A.setValueStencil(row, col, - value)
            
                # - O
                col.field = 3
                col.index = (i,j)
                A.setValueStencil(row, col, 1.)
                
                
                
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
                        ((i-1, j-1), + (P_ave[ix-1, jx  ] - P_ave[ix,   jx-1]) * arak_fac),
                        ((i-1, j  ), + (P_ave[ix,   jx+1] - P_ave[ix,   jx-1]) * arak_fac \
                                     + (P_ave[ix-1, jx+1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i-1, j+1), + (P_ave[ix,   jx+1] - P_ave[ix-1, jx  ]) * arak_fac),
                        ((i,   j-1), - (P_ave[ix+1, jx  ] - P_ave[ix-1, jx  ]) * arak_fac \
                                     - (P_ave[ix+1, jx-1] - P_ave[ix-1, jx-1]) * arak_fac),
                        ((i,   j  ), 1.),
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
        
    
    def mult(self, Mat mat, Vec X, Vec Y):
        self.matrix_mult(X, Y)
        
    
    @cython.boundscheck(False)
    def matrix_mult(self, Vec X, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(self.Xp, self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
         
        cdef double[:,:] Ad
        cdef double[:,:] Jd
        cdef double[:,:] Pd
        cdef double[:,:] Od
        
#         Ad = self.da1.getVecArray(self.Ad)#[...]
#         Jd = self.da1.getVecArray(self.Jd)#[...]
#         Pd = self.da1.getVecArray(self.Pd)#[...]
#         Od = self.da1.getVecArray(self.Od)#[...]
        
        cdef np.ndarray[double, ndim=3] y  = self.da4.getVecArray(Y)[...]
        
        cdef np.ndarray[double, ndim=3] xd
        cdef np.ndarray[double, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
          
        cdef np.ndarray[double, ndim=2] Ap = xp[...][:,:,0]
        cdef np.ndarray[double, ndim=2] Jp = xp[...][:,:,1]
        cdef np.ndarray[double, ndim=2] Pp = xp[...][:,:,2]
        cdef np.ndarray[double, ndim=2] Op = xp[...][:,:,3]
          
        cdef np.ndarray[double, ndim=2] Ah = xh[...][:,:,0]
        cdef np.ndarray[double, ndim=2] Jh = xh[...][:,:,1]
        cdef np.ndarray[double, ndim=2] Ph = xh[...][:,:,2]
        cdef np.ndarray[double, ndim=2] Oh = xh[...][:,:,3]
          
        cdef double[:,:] A_ave = 0.5 * (Ap + Ah)
        cdef double[:,:] J_ave = 0.5 * (Jp + Jh)
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        
        if self.pc == None:
            self.da4.globalToLocal(X, self.localXd)
#             xd = self.da4.getVecArray(X)#[...]
        else:
            self.pc.solve(X, self.Yd)
            self.da4.globalToLocal(self.Yd, self.localXd)
#             xd = self.da4.getVecArray(self.Yd)#[...]
        
        xd = self.da4.getVecArray(self.localXd)[...]
        
        Ad = xd[:,:,0]
        Jd = xd[:,:,1]
        Pd = xd[:,:,2]
        Od = xd[:,:,3]
        
#         Ad[:,:] = xd[:,:,0]
#         Jd[:,:] = xd[:,:,1]
#         Pd[:,:] = xd[:,:,2]
#         Od[:,:] = xd[:,:,3]
#         
#         
#         # magnetic potential
#         self.Ad.copy(self.YA)
#          
#         self.derivatives.arakawa_vec(self.Pp, self.Ad, self.T1)
#         self.derivatives.arakawa_vec(self.Ph, self.Ad, self.T2)
#         self.YA.axpy(0.25*self.ht, self.T1)
#         self.YA.axpy(0.25*self.ht, self.T2)
#  
#         self.derivatives.arakawa_vec(self.Pd, self.Ap, self.T1)
#         self.derivatives.arakawa_vec(self.Pd, self.Ah, self.T2)
#         self.YA.axpy(0.25*self.ht, self.T1)
#         self.YA.axpy(0.25*self.ht, self.T2)
#         
#         # current density
#         self.derivatives.laplace_vec(self.Ad, self.YJ, +1.)
#         self.YJ.axpy(1., self.Jd)
#         
#         # streaming function
#         self.derivatives.laplace_vec(self.Pd, self.YP, +1.)
#         self.YP.axpy(1., self.Od)
# 
#         # vorticity
#         self.Od.copy(self.YO)
#          
#         self.derivatives.arakawa_vec(self.Pp, self.Od, self.T1)
#         self.derivatives.arakawa_vec(self.Ph, self.Od, self.T2)
#         self.YO.axpy(0.25*self.ht, self.T1)
#         self.YO.axpy(0.25*self.ht, self.T2)
#  
#         self.derivatives.arakawa_vec(self.Pd, self.Op, self.T1)
#         self.derivatives.arakawa_vec(self.Pd, self.Oh, self.T2)
#         self.YO.axpy(0.25*self.ht, self.T1)
#         self.YO.axpy(0.25*self.ht, self.T2)
#          
#         self.derivatives.arakawa_vec(self.Jp, self.Ad, self.T1)
#         self.derivatives.arakawa_vec(self.Jh, self.Ad, self.T2)
#         self.YO.axpy(0.25*self.ht, self.T1)
#         self.YO.axpy(0.25*self.ht, self.T2)
#  
#         self.derivatives.arakawa_vec(self.Jd, self.Ap, self.T1)
#         self.derivatives.arakawa_vec(self.Jd, self.Ah, self.T2)
#         self.YO.axpy(0.25*self.ht, self.T1)
#         self.YO.axpy(0.25*self.ht, self.T2)
#         
#         
#         YA = self.da1.getVecArray(self.YA)
#         YJ = self.da1.getVecArray(self.YJ)
#         YP = self.da1.getVecArray(self.YP)
#         YO = self.da1.getVecArray(self.YO)
#         
#         y  = self.da4.getVecArray(Y)
#         
#         y[:,:,0] = YA[:,:]
#         y[:,:,1] = YJ[:,:]
#         y[:,:,2] = YP[:,:]
#         y[:,:,3] = YO[:,:]
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
             
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                 
                # magnetic potential
                y[iy, jy, 0] = Ad[ix,jx] \
                             + 0.5 * self.ht * self.derivatives.arakawa(P_ave, Ad, ix, jx) \
                             + 0.5 * self.ht * self.derivatives.arakawa(Pd, A_ave, ix, jx)
                 
                # current density
                y[iy, jy, 1] = Jd[ix, jx] \
                             + ( \
                                   + 1. * Ad[ix-1, jx] \
                                   - 2. * Ad[ix,   jx] \
                                   + 1. * Ad[ix+1, jx] \
                               ) * self.hx_inv**2 \
                             + ( \
                                   + 1. * Ad[ix, jx-1] \
                                   - 2. * Ad[ix, jx  ] \
                                   + 1. * Ad[ix, jx+1] \
                               ) * self.hy_inv**2
                
#                 y[iy, jy, 1] = Jd[ix, jx] \
#                              + self.derivatives.laplace(Ad, ix, jx)
                 
                # streaming function
                y[iy, jy, 2] = Od[ix, jx] \
                             + ( \
                                   + 1. * Pd[ix-1, jx] \
                                   - 2. * Pd[ix,   jx] \
                                   + 1. * Pd[ix+1, jx] \
                               ) * self.hx_inv**2 \
                             + ( \
                                   + 1. * Pd[ix, jx-1] \
                                   - 2. * Pd[ix, jx  ] \
                                   + 1. * Pd[ix, jx+1] \
                               ) * self.hy_inv**2
                 
#                 y[iy, jy, 2] = Od[ix, jx] \
#                              + self.derivatives.laplace(Pd, ix, jx)
                 
                # vorticity
                y[iy, jy, 3] = Od[ix,jx] \
                             + 0.5 * self.ht * self.derivatives.arakawa(P_ave, Od, ix, jx) \
                             + 0.5 * self.ht * self.derivatives.arakawa(Pd, O_ave, ix, jx) \
                             + 0.5 * self.ht * self.derivatives.arakawa(J_ave, Ad, ix, jx) \
                             + 0.5 * self.ht * self.derivatives.arakawa(Jd, A_ave, ix, jx)


   
    def snes_function(self, SNES snes, Vec X, Vec Y):
        self.function(X, Y)
        
    
    def function(self, Vec X, Vec Y):
        cdef int i, j
        cdef int ix, iy, jx, jy
        cdef int xe, xs, ye, ys
        
        (xs, xe), (ys, ye) = self.da4.getRanges()
        
        self.da4.globalToLocal(X,       self.localXp)
        self.da4.globalToLocal(self.Xh, self.localXh)
        
        cdef np.ndarray[double, ndim=3] y  = self.da4.getVecArray(Y)[...]
        
        cdef np.ndarray[double, ndim=3] xp = self.da4.getVecArray(self.localXp)[...]
        cdef np.ndarray[double, ndim=3] xh = self.da4.getVecArray(self.localXh)[...]
        
        cdef np.ndarray[double, ndim=2] Ap = xp[...][:,:,0]
        cdef np.ndarray[double, ndim=2] Jp = xp[...][:,:,1]
        cdef np.ndarray[double, ndim=2] Pp = xp[...][:,:,2]
        cdef np.ndarray[double, ndim=2] Op = xp[...][:,:,3]
        
        cdef np.ndarray[double, ndim=2] Ah = xh[...][:,:,0]
        cdef np.ndarray[double, ndim=2] Jh = xh[...][:,:,1]
        cdef np.ndarray[double, ndim=2] Ph = xh[...][:,:,2]
        cdef np.ndarray[double, ndim=2] Oh = xh[...][:,:,3]
        
        cdef double[:,:] A_ave = 0.5 * (Ap + Ah)
        cdef double[:,:] J_ave = 0.5 * (Jp + Jh)
        cdef double[:,:] P_ave = 0.5 * (Pp + Ph)
        cdef double[:,:] O_ave = 0.5 * (Op + Oh)
        
        
        for i in range(xs, xe):
            ix = i-xs+2
            iy = i-xs
            
            for j in range(ys, ye):
                jx = j-ys+2
                jy = j-ys
                
                # magnetic potential
                y[iy, jy, 0] = Ap[ix,jx] - Ah[ix,jx] \
                             + self.ht * self.derivatives.arakawa(P_ave, A_ave, ix, jx)
                
                # current density
                y[iy, jy, 1] = Jp[ix, jx] \
                             + self.derivatives.laplace(Ap, ix, jx)
                
                # streaming function
                y[iy, jy, 2] = Op[ix, jx] \
                             + self.derivatives.laplace(Pp, ix, jx)
                
                # vorticity
                y[iy, jy, 3] = Op[ix,jx] - Oh[ix,jx] \
                             + self.ht * self.derivatives.arakawa(P_ave, O_ave, ix, jx) \
                             + self.ht * self.derivatives.arakawa(J_ave, A_ave, ix, jx)

